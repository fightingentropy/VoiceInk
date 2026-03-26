import Foundation
import os
import AppKit

enum LocalModelWarmRetention: Int, CaseIterable, Identifiable {
    case fiveMinutes = 300
    case fifteenMinutes = 900
    case thirtyMinutes = 1800
    case oneHour = 3600
    case never = -1

    var id: Int { rawValue }

    var displayName: String {
        switch self {
        case .fiveMinutes:
            return "5 minutes"
        case .fifteenMinutes:
            return "15 minutes"
        case .thirtyMinutes:
            return "30 minutes"
        case .oneHour:
            return "1 hour"
        case .never:
            return "Until quit"
        }
    }

    var interval: TimeInterval? {
        switch self {
        case .never:
            return nil
        default:
            return TimeInterval(rawValue)
        }
    }
}

@MainActor
final class ModelPrewarmService: ObservableObject {
    private let engine: VoiceInkEngine
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "ModelPrewarm")
    private let prewarmAudioURL = Bundle.main.url(forResource: "esc", withExtension: "wav")
    private let prewarmEnabledKey = "PrewarmModelOnWake"
    private let warmRetentionKey = "LocalModelWarmRetentionSeconds"
    private var unloadTask: Task<Void, Never>?
    private var modelChangeTask: Task<Void, Never>?

    init(engine: VoiceInkEngine) {
        self.engine = engine
        setupNotifications()
        schedulePrewarmOnAppLaunch()
    }

    // MARK: - Notification Setup

    private func setupNotifications() {
        let center = NSWorkspace.shared.notificationCenter

        // Trigger on wake from sleep
        center.addObserver(
            self,
            selector: #selector(schedulePrewarm),
            name: NSWorkspace.didWakeNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleLocalModelDidUse),
            name: .localModelDidUse,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleSettingsDidChange),
            name: .AppSettingsDidChange,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleModelDidChange),
            name: .didChangeModel,
            object: nil
        )

        logger.notice("ModelPrewarmService initialized - listening for wake and app launch")
    }

    // MARK: - Trigger Handlers

    /// Trigger on app launch (cold start)
    private func schedulePrewarmOnAppLaunch() {
        logger.notice("App launched, scheduling prewarm")
        Task {
            try? await Task.sleep(for: .seconds(3))
            await performPrewarm()
        }
    }

    /// Trigger on wake from sleep or screen unlock
    @objc private func schedulePrewarm() {
        logger.notice("Mac activity detected (wake/unlock), scheduling prewarm")
        Task {
            try? await Task.sleep(for: .seconds(3))
            await performPrewarm()
        }
    }

    @objc private func handleLocalModelDidUse() {
        rescheduleIdleUnload()
    }

    @objc private func handleSettingsDidChange() {
        rescheduleIdleUnload()
    }

    @objc private func handleModelDidChange() {
        scheduleModelChangePrewarm()
    }

    // MARK: - Core Prewarming Logic

    private func performPrewarm() async {
        guard shouldPrewarm() else { return }
        unloadTask?.cancel()

        guard let audioURL = prewarmAudioURL else {
            logger.error("❌ Prewarm audio file (esc.wav) not found")
            return
        }

        guard let currentModel = engine.transcriptionModelManager.currentTranscriptionModel else {
            logger.notice("No model selected, skipping prewarm")
            return
        }

        logger.notice("Prewarming \(currentModel.displayName, privacy: .public)")
        let startTime = Date()

        do {
            try await engine.prewarmCurrentLocalModel(using: audioURL)
            let duration = Date().timeIntervalSince(startTime)

            logger.notice("Prewarm completed in \(String(format: "%.2f", duration), privacy: .public)s")
            rescheduleIdleUnload()

        } catch {
            logger.error("❌ Prewarm failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Validation

    private func shouldPrewarm() -> Bool {
        // Check if user has enabled prewarming
        let isEnabled = UserDefaults.standard.bool(forKey: prewarmEnabledKey)
        guard isEnabled else {
            logger.notice("Prewarm disabled by user")
            return false
        }

        // Only prewarm local models.
        guard let model = engine.transcriptionModelManager.currentTranscriptionModel else {
            return false
        }

        switch model.provider {
        case .local, .parakeet, .localVoxtral, .cohereTranscribe:
            return true
        default:
            logger.notice("Skipping prewarm - cloud models don't need it")
            return false
        }
    }

    func handleWindowDidDisappear() {
        rescheduleIdleUnload()
    }

    private func currentWarmRetention() -> LocalModelWarmRetention {
        let storedValue = UserDefaults.standard.integer(forKey: warmRetentionKey)
        return LocalModelWarmRetention(rawValue: storedValue) ?? .fifteenMinutes
    }

    private func rescheduleIdleUnload() {
        unloadTask?.cancel()

        guard let model = engine.transcriptionModelManager.currentTranscriptionModel else {
            return
        }

        guard isLocalModel(model) else {
            return
        }

        guard let interval = currentWarmRetention().interval else {
            logger.notice("Local model retention set to until quit")
            return
        }

        unloadTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(interval))
                await MainActor.run {
                    guard let self else { return }
                    if self.engine.recordingState != .idle {
                        self.logger.notice("Skipping local model unload while recording is active")
                        self.rescheduleIdleUnload()
                        return
                    }

                    self.logger.notice("Unloading local model after \(interval, privacy: .public)s of inactivity")
                }

                guard let self else { return }
                if self.engine.recordingState == .idle {
                    await self.engine.cleanupResources()
                }
            } catch is CancellationError {
                return
            } catch {
                return
            }
        }
    }

    private func scheduleModelChangePrewarm() {
        modelChangeTask?.cancel()

        modelChangeTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .milliseconds(500))

                guard let self else { return }
                guard self.engine.recordingState == .idle else { return }

                let currentModel = self.engine.transcriptionModelManager.currentTranscriptionModel
                let retainedModel = currentModel.flatMap { self.isLocalModel($0) ? $0 : nil }
                await self.engine.retainOnlyLocalModelResources(for: retainedModel)

                guard self.shouldPrewarm() else {
                    self.rescheduleIdleUnload()
                    return
                }

                await self.performPrewarm()
            } catch is CancellationError {
                return
            } catch {
                return
            }
        }
    }

    private func isLocalModel(_ model: any TranscriptionModel) -> Bool {
        switch model.provider {
        case .local, .parakeet, .localVoxtral, .cohereTranscribe:
            return true
        default:
            return false
        }
    }

    deinit {
        unloadTask?.cancel()
        modelChangeTask?.cancel()
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        NotificationCenter.default.removeObserver(self)
        logger.notice("ModelPrewarmService deinitialized")
    }
}
