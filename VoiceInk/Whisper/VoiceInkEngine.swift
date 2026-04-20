import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import AppKit
import os

@MainActor
class VoiceInkEngine: NSObject, ObservableObject {
    @Published var recordingState: RecordingState = .idle
    @Published var shouldCancelRecording = false
    var partialTranscript: String = ""
    var currentSession: TranscriptionSession?

    let recorder = Recorder()
    var recordedFile: URL? = nil
    let recordingsDirectory: URL

    // Injected managers
    let whisperModelManager: WhisperModelManager
    let transcriptionModelManager: TranscriptionModelManager
    weak var recorderUIManager: RecorderUIManager?

    let modelContext: ModelContext
    internal let serviceRegistry: TranscriptionServiceRegistry
    private let pipeline: TranscriptionPipeline

    let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "VoiceInkEngine")

    /// Monotonic counter bumped every time the user starts or finishes a
    /// transcription. The idle-unload task captures the counter at schedule
    /// time and bails out if it has moved — means another recording raced in
    /// and we shouldn't evict the model out from under it.
    private var activityGeneration: UInt64 = 0
    private var idleUnloadTask: Task<Void, Never>?

    init(
        modelContext: ModelContext,
        whisperModelManager: WhisperModelManager,
        transcriptionModelManager: TranscriptionModelManager
    ) {
        self.modelContext = modelContext
        self.whisperModelManager = whisperModelManager
        self.transcriptionModelManager = transcriptionModelManager

        self.recordingsDirectory = AppStoragePaths.recordingsDirectory

        self.serviceRegistry = TranscriptionServiceRegistry(
            modelProvider: whisperModelManager
        )
        self.pipeline = TranscriptionPipeline(
            modelContext: modelContext,
            serviceRegistry: serviceRegistry
        )

        super.init()

        setupNotifications()
        createRecordingsDirectoryIfNeeded()
    }

    private func createRecordingsDirectoryIfNeeded() {
        do {
            try AppStoragePaths.createDirectoryIfNeeded(at: recordingsDirectory)
        } catch {
            logger.error("❌ Error creating recordings directory: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Toggle Record

    func toggleRecord() async {
        logger.notice("toggleRecord called – state=\(String(describing: self.recordingState), privacy: .public)")

        if recordingState == .recording {
            partialTranscript = ""
            recordingState = .transcribing
            await recorder.stopRecording()

            if let recordedFile {
                if !shouldCancelRecording {
                    await runPipeline(audioURL: recordedFile, recordedAt: Date())
                } else {
                    currentSession?.cancel()
                    currentSession = nil
                    try? FileManager.default.removeItem(at: recordedFile)
                    recordingState = .idle
                    await cleanupResources()
                }
            } else {
                logger.error("❌ No recorded file found after stopping recording")
                currentSession?.cancel()
                currentSession = nil
                recordingState = .idle
                await cleanupResources()
            }
        } else {
            logger.notice("toggleRecord: entering start-recording branch")
            guard let preflightModel = transcriptionModelManager.currentTranscriptionModel else {
                NotificationManager.shared.showNotification(title: "No AI Model Selected", type: .error)
                return
            }

            // Thermal / low-power advisory before we spin up heavyweight
            // local inference (Voxtral, Cohere, Whisper). We only surface
            // a notification when the advisory recommends a fallback —
            // softer warnings go to the log to avoid notification spam.
            let workload = SystemResourceGuard.workload(for: preflightModel.provider)
            let advisory = SystemResourceGuard.evaluate(workload: workload)
            if let message = advisory.message {
                logger.notice("Thermal advisory: \(advisory.logDescription, privacy: .public) — \(message, privacy: .public)")
                if advisory.recommendedFallback {
                    NotificationManager.shared.showNotification(title: message, type: .info)
                }
            }

            shouldCancelRecording = false
            partialTranscript = ""

            // Start of a fresh recording — invalidate any pending idle-unload
            // so we don't race against ourselves while the user is recording.
            activityGeneration &+= 1
            idleUnloadTask?.cancel()
            idleUnloadTask = nil

            requestRecordPermission { [self] granted in
                if granted {
                    Task {
                        do {
                            let fileName = "\(UUID().uuidString).wav"
                            let permanentURL = self.recordingsDirectory.appendingPathComponent(fileName)
                            self.recordedFile = permanentURL

                            guard let model = self.transcriptionModelManager.currentTranscriptionModel else {
                                throw VoiceInkEngineError.transcriptionFailed
                            }

                            let session = try self.serviceRegistry.createSession(
                                for: model,
                                onPartialTranscript: { [weak self] partial in
                                    Task { @MainActor in
                                        self?.partialTranscript = partial
                                    }
                                }
                            )
                            self.currentSession = session

                            let startupForwarder = BufferedPCMChunkForwarder(
                                capacityBytes: BoundedPCMChunkBuffer.defaultCapacityBytes,
                                logger: self.logger,
                                label: "Streaming startup"
                            )
                            let sessionIdentity = ObjectIdentifier(session as AnyObject)
                            let prepareTask = Task<(@Sendable (Data) -> Void)?, Never> {
                                do {
                                    return try await session.prepare(model: model)
                                } catch {
                                    await MainActor.run {
                                        self.logger.error("❌ Session preparation failed: \(error.localizedDescription, privacy: .public)")
                                    }
                                    return nil
                                }
                            }

                            self.recorder.onAudioChunk = { data in
                                startupForwarder.send(data)
                            }

                            try await self.recorder.startRecording(toOutputFile: permanentURL)

                            guard self.recorderUIManager?.isMiniRecorderVisible ?? false, !self.shouldCancelRecording else {
                                await self.recorder.stopRecording()
                                session.cancel()
                                prepareTask.cancel()
                                startupForwarder.finish()
                                self.currentSession = nil
                                self.recordedFile = nil
                                return
                            }

                            self.recordingState = .recording
                            self.logger.notice("toggleRecord: recording started successfully, state=recording")

                            Task { @MainActor [weak self] in
                                let realCallback = await prepareTask.value

                                guard let self else {
                                    startupForwarder.finish()
                                    session.cancel()
                                    return
                                }

                                let isCurrentSession = self.currentSession.map {
                                    ObjectIdentifier($0 as AnyObject) == sessionIdentity
                                } ?? false

                                guard isCurrentSession,
                                      self.recordingState == .recording,
                                      !self.shouldCancelRecording else {
                                    startupForwarder.finish()
                                    session.cancel()
                                    return
                                }

                                if let realCallback {
                                    startupForwarder.installConsumer(realCallback)
                                } else {
                                    startupForwarder.finish()
                                    self.recorder.onAudioChunk = nil
                                }
                            }

                            // Raise QoS to .userInitiated: this task loads / prewarms the
                                // model while the user waits with the mini-recorder open. The
                                // default detached priority is .utility, which lets unrelated
                                // background work starve this out on a busy Mac.
                            Task.detached(priority: .userInitiated) { [weak self] in
                                guard let self else { return }

                                if model.provider == .local {
                                    let availableModels = await self.whisperModelManager.availableModels
                                    let loadedModelName = await self.whisperModelManager.loadedLocalModel?.name
                                    let isModelLoaded = await self.whisperModelManager.isModelLoaded

                                    if let localWhisperModel = availableModels.first(where: { $0.name == model.name }),
                                       loadedModelName != localWhisperModel.name || !isModelLoaded {
                                        do {
                                            try await self.whisperModelManager.loadModel(localWhisperModel)
                                        } catch {
                                            self.logger.error("❌ Model loading failed: \(error.localizedDescription, privacy: .public)")
                                        }
                                    }
                                } else if let parakeetModel = model as? ParakeetModel {
                                    try? await self.serviceRegistry.parakeetTranscriptionService.loadModel(for: parakeetModel)
                                } else if model.provider == .localVoxtral {
                                    _ = try? await VoxtralNativeRuntime.shared.warmupModel(
                                        modelReference: LocalVoxtralConfiguration.modelName,
                                        autoDownload: true
                                    )
                                } else if let cohereModel = model as? LocalCohereTranscribeModel {
                                    try? await self.serviceRegistry.cohereTranscribeTranscriptionService.prepareModel(for: cohereModel)
                                }

                            }

                        } catch {
                            self.logger.error("❌ Failed to start recording: \(error.localizedDescription, privacy: .public)")
                            self.currentSession?.cancel()
                            self.currentSession = nil
                            NotificationManager.shared.showNotification(title: "Recording failed to start", type: .error)
                            self.logger.notice("toggleRecord: calling dismissMiniRecorder from error handler")
                            await self.recorderUIManager?.dismissMiniRecorder()
                            self.recordedFile = nil
                        }
                    }
                } else {
                    logger.error("❌ Recording permission denied.")
                }
            }
        }
    }

    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
        response(true)
    }

    // MARK: - Pipeline Dispatch

    private func runPipeline(audioURL: URL, recordedAt: Date) async {
        guard let model = transcriptionModelManager.currentTranscriptionModel else {
            recordingState = .idle
            return
        }

        activityGeneration &+= 1
        idleUnloadTask?.cancel()
        idleUnloadTask = nil

        let session = currentSession
        currentSession = nil

        await pipeline.run(
            audioURL: audioURL,
            recordedAt: recordedAt,
            model: model,
            session: session,
            onStateChange: { [weak self] state in self?.recordingState = state },
            shouldCancel: { [weak self] in self?.shouldCancelRecording ?? false },
            onCleanup: { [weak self] in await self?.cleanupResources() },
            onDismiss: { [weak self] in await self?.recorderUIManager?.dismissMiniRecorder() }
        )

        shouldCancelRecording = false
        if recordingState != .idle {
            recordingState = .idle
        }

        scheduleIdleUnloadIfNeeded()
    }

    /// Schedules a one-shot task that unloads large local models after the
    /// configured idle threshold on RAM-constrained Macs. No-op on Macs with
    /// plenty of memory — the cost of reloading the model outweighs the RAM
    /// we'd save.
    private func scheduleIdleUnloadIfNeeded() {
        guard SystemResourceGuard.shouldAutoUnloadIdleModels() else { return }

        let scheduledGeneration = activityGeneration
        let threshold = SystemResourceGuard.idleUnloadThreshold
        idleUnloadTask?.cancel()
        idleUnloadTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(threshold * 1_000_000_000))
            guard !Task.isCancelled, let self else { return }
            await self.performIdleUnloadIfStillIdle(expectedGeneration: scheduledGeneration)
        }
    }

    private func performIdleUnloadIfStillIdle(expectedGeneration: UInt64) async {
        // Bail if another recording/transcription happened after we were
        // scheduled, or if the user is currently recording right now.
        guard activityGeneration == expectedGeneration,
              recordingState == .idle else {
            return
        }
        logger.notice("Idle unload: releasing local model resources after \(Int(SystemResourceGuard.idleUnloadThreshold), privacy: .public)s of inactivity")
        await cleanupResources()
    }

    // MARK: - Resource Cleanup

    func cleanupResources() async {
        logger.notice("cleanupResources: releasing model resources")
        await retainOnlyLocalModelResources(for: nil)
        logger.notice("cleanupResources: completed")
    }

    func retainOnlyLocalModelResources(for model: (any TranscriptionModel)?) async {
        if model?.provider != .local || whisperModelManager.loadedLocalModel?.name != model?.name {
            await whisperModelManager.cleanupResources()
        }

        if model?.provider != .parakeet {
            await serviceRegistry.parakeetTranscriptionService.cleanup()
        }

        let keptVoxtralModels: Set<String> =
            model?.provider == .localVoxtral ? [LocalVoxtralConfiguration.modelName] : []
        await VoxtralNativeRuntime.shared.unloadAllUnusedPreparedStates(keeping: keptVoxtralModels)

        if model?.provider != .cohereTranscribe {
            await serviceRegistry.cohereTranscribeTranscriptionService.cleanup()
        }
    }

    func prewarmCurrentLocalModel(using audioURL: URL) async throws {
        guard let model = transcriptionModelManager.currentTranscriptionModel else {
            return
        }

        switch model.provider {
        case .local:
            guard let localWhisperModel = whisperModelManager.availableModels.first(where: { $0.name == model.name }) else {
                throw VoiceInkEngineError.modelLoadFailed
            }

            if whisperModelManager.loadedLocalModel?.name != localWhisperModel.name || !whisperModelManager.isModelLoaded {
                try await whisperModelManager.loadModel(localWhisperModel)
            }
        case .parakeet:
            guard let parakeetModel = model as? ParakeetModel else { return }
            try await serviceRegistry.parakeetTranscriptionService.loadModel(for: parakeetModel)
        case .localVoxtral:
            _ = try await VoxtralNativeRuntime.shared.warmupModel(
                modelReference: LocalVoxtralConfiguration.modelName,
                autoDownload: true
            )
        case .cohereTranscribe:
            guard let cohereModel = model as? LocalCohereTranscribeModel else { return }
            try await serviceRegistry.cohereTranscribeTranscriptionService.warmup(for: cohereModel)
        default:
            return
        }

        if model.provider != .localVoxtral && model.provider != .cohereTranscribe {
            _ = try await serviceRegistry.transcribe(audioURL: audioURL, model: model)
        }
    }

    func setupNotifications() {}
}
