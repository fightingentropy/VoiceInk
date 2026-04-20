import Foundation
import SwiftUI
import os

@MainActor
class RecorderUIManager: ObservableObject {
    @Published var miniRecorderError: String?

    @Published var isMiniRecorderVisible = false {
        didSet {
            Task { @MainActor in
                if isMiniRecorderVisible {
                    showRecorderPanel()
                } else {
                    hideRecorderPanel()
                }
            }
        }
    }

    var miniWindowManager: MiniWindowManager?

    private weak var engine: VoiceInkEngine?
    private var recorder: Recorder?

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "RecorderUIManager")

    init() {}

    /// Call after VoiceInkEngine is created to break the circular init dependency.
    func configure(engine: VoiceInkEngine, recorder: Recorder) {
        self.engine = engine
        self.recorder = recorder
        setupNotifications()
    }

    // MARK: - Recorder Panel Management

    func showRecorderPanel() {
        guard let engine = engine, let recorder = recorder else { return }
        logger.notice("Showing mini recorder")

        if miniWindowManager == nil {
            miniWindowManager = MiniWindowManager(engine: engine, recorder: recorder)
        }
        miniWindowManager?.show()
    }

    func hideRecorderPanel() {
        miniWindowManager?.hide()
    }

    // MARK: - Mini Recorder Management

    func toggleMiniRecorder() async {
        guard let engine = engine else { return }
        logger.notice("toggleMiniRecorder called – visible=\(self.isMiniRecorderVisible, privacy: .public), state=\(String(describing: engine.recordingState), privacy: .public)")

        if isMiniRecorderVisible {
            if engine.recordingState == .recording {
                logger.notice("toggleMiniRecorder: stopping recording (was recording)")
                await engine.toggleRecord()
            } else {
                logger.notice("toggleMiniRecorder: cancelling (was not recording)")
                await cancelRecording()
            }
        } else {
            SoundManager.shared.playStartSound()
            await MainActor.run { isMiniRecorderVisible = true }
            await engine.toggleRecord()
        }
    }

    func dismissMiniRecorder() async {
        guard let engine = engine, let recorder = recorder else { return }
        logger.notice("dismissMiniRecorder called – state=\(String(describing: engine.recordingState), privacy: .public)")

        if engine.recordingState == .busy {
            logger.notice("dismissMiniRecorder: early return, state is busy")
            return
        }

        let wasRecording = engine.recordingState == .recording

        await MainActor.run {
            engine.recordingState = .busy
        }

        // Cancel and release any active streaming session to prevent resource leaks.
        engine.currentSession?.cancel()
        engine.currentSession = nil

        if wasRecording {
            await recorder.stopRecording()
        }

        hideRecorderPanel()

        await MainActor.run {
            isMiniRecorderVisible = false
        }

        let retainedModel = retainableLocalModel(from: engine)
        await engine.retainOnlyLocalModelResources(for: retainedModel)
        if retainedModel != nil {
            NotificationCenter.default.post(name: .localModelDidUse, object: nil)
        }

        await MainActor.run {
            engine.recordingState = .idle
        }
        logger.notice("dismissMiniRecorder completed")
    }

    func resetOnLaunch() async {
        guard let engine = engine, let recorder = recorder else { return }
        logger.notice("Resetting recording state on launch")
        await recorder.stopRecording()
        hideRecorderPanel()
        await MainActor.run {
            isMiniRecorderVisible = false
            engine.shouldCancelRecording = false
            miniRecorderError = nil
            engine.recordingState = .idle
        }
        await engine.cleanupResources()
    }

    func cancelRecording() async {
        guard let engine = engine else { return }
        logger.notice("cancelRecording called")
        SoundManager.shared.playEscSound()
        engine.shouldCancelRecording = true
        await dismissMiniRecorder()
    }

    // MARK: - Notification Handling

    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleToggleMiniRecorder),
            name: .toggleMiniRecorder,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDismissMiniRecorder),
            name: .dismissMiniRecorder,
            object: nil
        )
    }

    @objc public func handleToggleMiniRecorder() {
        logger.notice("handleToggleMiniRecorder: .toggleMiniRecorder notification received")
        Task {
            await toggleMiniRecorder()
        }
    }

    @objc public func handleDismissMiniRecorder() {
        logger.notice("handleDismissMiniRecorder: .dismissMiniRecorder notification received")
        Task {
            await dismissMiniRecorder()
        }
    }

    private func retainableLocalModel(from engine: VoiceInkEngine) -> (any TranscriptionModel)? {
        guard let model = engine.transcriptionModelManager.currentTranscriptionModel else {
            return nil
        }

        switch model.provider {
        case .local, .parakeet, .localVoxtral, .cohereTranscribe:
            return model
        default:
            return nil
        }
    }
}
