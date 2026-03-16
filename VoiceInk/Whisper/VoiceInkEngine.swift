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
    let enhancementService: AIEnhancementService?
    private let pipeline: TranscriptionPipeline

    let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "VoiceInkEngine")

    init(
        modelContext: ModelContext,
        whisperModelManager: WhisperModelManager,
        transcriptionModelManager: TranscriptionModelManager,
        enhancementService: AIEnhancementService? = nil
    ) {
        self.modelContext = modelContext
        self.whisperModelManager = whisperModelManager
        self.transcriptionModelManager = transcriptionModelManager
        self.enhancementService = enhancementService

        self.recordingsDirectory = AppStoragePaths.recordingsDirectory

        self.serviceRegistry = TranscriptionServiceRegistry(
            modelProvider: whisperModelManager,
            modelsDirectory: whisperModelManager.modelsDirectory
        )
        self.pipeline = TranscriptionPipeline(
            modelContext: modelContext,
            serviceRegistry: serviceRegistry,
            enhancementService: enhancementService
        )

        super.init()

        if let enhancementService {
            PowerModeSessionManager.shared.configure(engine: self, enhancementService: enhancementService)
        }

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

    func getEnhancementService() -> AIEnhancementService? {
        return enhancementService
    }

    // MARK: - Toggle Record

    func toggleRecord(powerModeId: UUID? = nil) async {
        logger.notice("toggleRecord called – state=\(String(describing: self.recordingState), privacy: .public)")

        if recordingState == .recording {
            partialTranscript = ""
            recordingState = .transcribing
            recorder.stopRecording()

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
            guard transcriptionModelManager.currentTranscriptionModel != nil else {
                NotificationManager.shared.showNotification(title: "No AI Model Selected", type: .error)
                return
            }
            shouldCancelRecording = false
            partialTranscript = ""

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

                            let session = self.serviceRegistry.createSession(
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
                                self.recorder.stopRecording()
                                session.cancel()
                                prepareTask.cancel()
                                startupForwarder.finish()
                                self.currentSession = nil
                                self.recordedFile = nil
                                return
                            }

                            self.recordingState = .recording
                            self.logger.notice("toggleRecord: recording started successfully, state=recording")

                            await ActiveWindowService.shared.applyConfiguration(powerModeId: powerModeId)

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

                            Task.detached { [weak self] in
                                guard let self else { return }

                                if model.provider == .local {
                                    if let localWhisperModel = await self.whisperModelManager.availableModels.first(where: { $0.name == model.name }),
                                       await self.whisperModelManager.whisperContext == nil {
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
                                }

                                if let enhancementService = self.enhancementService {
                                    await MainActor.run {
                                        enhancementService.captureClipboardContext()
                                    }
                                    await enhancementService.captureScreenContext()
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
            serviceRegistry.parakeetTranscriptionService.cleanup()
        }

        let keptVoxtralModels: Set<String> =
            model?.provider == .localVoxtral ? [LocalVoxtralConfiguration.modelName] : []
        await VoxtralNativeRuntime.shared.unloadAllUnusedPreparedStates(keeping: keptVoxtralModels)
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

            if whisperModelManager.whisperContext == nil
                || whisperModelManager.loadedLocalModel?.name != localWhisperModel.name {
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
        default:
            return
        }

        if model.provider != .localVoxtral {
            _ = try await serviceRegistry.transcribe(audioURL: audioURL, model: model)
        }
    }

    // MARK: - Notification Handling

    func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleLicenseStatusChanged),
            name: .licenseStatusChanged,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handlePromptChange),
            name: .promptDidChange,
            object: nil
        )
    }

    @objc func handleLicenseStatusChanged() {
        pipeline.licenseViewModel = LicenseViewModel()
    }

    @objc func handlePromptChange() {
        Task {
            let currentPrompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")
                ?? whisperModelManager.whisperPrompt.transcriptionPrompt
            if let context = whisperModelManager.whisperContext {
                await context.setPrompt(currentPrompt)
            }
        }
    }
}
