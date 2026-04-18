import Foundation
import AVFoundation
import SwiftData
import os

/// Handles the full post-recording pipeline:
/// transcribe → filter → format → word-replace → prompt-detect → AI enhance → paste/dismiss → save/history/benchmark
@MainActor
class TranscriptionPipeline {
    private let modelContext: ModelContext
    private let serviceRegistry: TranscriptionServiceRegistry
    private let enhancementService: AIEnhancementService?
    private let promptDetectionService = PromptDetectionService()
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionPipeline")

    private struct PersistencePayload: Sendable {
        let timestamp: Date
        let text: String
        let enhancedText: String?
        let duration: TimeInterval
        let audioFileURL: String
        let transcriptionModelName: String?
        let aiEnhancementModelName: String?
        let promptName: String?
        let transcriptionDuration: TimeInterval?
        let enhancementDuration: TimeInterval?
        let aiRequestSystemMessage: String?
        let aiRequestUserMessage: String?
        let powerModeName: String?
        let powerModeEmoji: String?
        let transcriptionStatus: TranscriptionStatus

        var isCompleted: Bool {
            transcriptionStatus == .completed
        }

        func makeTranscription() -> Transcription {
            let transcription = Transcription(
                text: text,
                duration: duration,
                enhancedText: enhancedText,
                audioFileURL: audioFileURL,
                transcriptionModelName: transcriptionModelName,
                aiEnhancementModelName: aiEnhancementModelName,
                promptName: promptName,
                transcriptionDuration: transcriptionDuration,
                enhancementDuration: enhancementDuration,
                aiRequestSystemMessage: aiRequestSystemMessage,
                aiRequestUserMessage: aiRequestUserMessage,
                powerModeName: powerModeName,
                powerModeEmoji: powerModeEmoji,
                transcriptionStatus: transcriptionStatus
            )
            transcription.timestamp = timestamp
            return transcription
        }
    }

    init(
        modelContext: ModelContext,
        serviceRegistry: TranscriptionServiceRegistry,
        enhancementService: AIEnhancementService?
    ) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
        self.enhancementService = enhancementService
    }

    /// Run the full pipeline for a given transcription record.
    /// - Parameters:
    ///   - audioURL: The recorded audio file.
    ///   - recordedAt: Timestamp captured when recording ended.
    ///   - model: The transcription model to use.
    ///   - session: An active streaming session if one was prepared, otherwise nil.
    ///   - onStateChange: Called when the pipeline moves to a new recording state (e.g. `.enhancing`).
    ///   - shouldCancel: Returns true if the user requested cancellation.
    ///   - onCleanup: Called when cancellation is detected to release model resources.
    ///   - onDismiss: Called at the end to dismiss the recorder panel.
    func run(
        audioURL: URL,
        recordedAt: Date,
        model: any TranscriptionModel,
        session: TranscriptionSession?,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: () -> Bool,
        onCleanup: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void
    ) async {
        if shouldCancel() {
            await onCleanup()
            return
        }

        Task {
            let isSystemMuteEnabled = UserDefaults.standard.bool(forKey: "isSystemMuteEnabled")
            if isSystemMuteEnabled {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            SoundManager.shared.playStopSound()
        }

        var finalPastedText: String?
        var promptDetectionResult: PromptDetectionService.PromptDetectionResult?
        var persistencePayload: PersistencePayload?

        logger.notice("🔄 Starting transcription...")

        do {
            let transcriptionStart = Date()
            var text: String
            if let session {
                text = try await session.transcribe(audioURL: audioURL)
            } else {
                text = try await serviceRegistry.transcribe(audioURL: audioURL, model: model)
            }
            logger.notice("📝 Transcript: \(text, privacy: .public)")
            text = TranscriptionOutputFilter.filter(text)
            logger.notice("📝 Output filter result: \(text, privacy: .public)")
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)

            let powerModeManager = PowerModeManager.shared
            let activePowerModeConfig = powerModeManager.currentActiveConfiguration
            let powerModeName = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.name : nil
            let powerModeEmoji = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.emoji : nil

            if shouldCancel() { await onCleanup(); return }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)

            if UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled") {
                text = WhisperTextFormatter.format(text)
                logger.notice("📝 Formatted transcript: \(text, privacy: .public)")
            }

            text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)
            logger.notice("📝 WordReplacement: \(text, privacy: .public)")

            let audioAsset = AVURLAsset(url: audioURL)
            let actualDuration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0
            finalPastedText = text

            if let enhancementService, enhancementService.isConfigured {
                let detectionResult = promptDetectionService.analyzeText(text, with: enhancementService)
                promptDetectionResult = detectionResult
                await promptDetectionService.applyDetectionResult(detectionResult, to: enhancementService)
            }

            if let enhancementService,
               enhancementService.isEnhancementEnabled,
               enhancementService.isConfigured {
                if shouldCancel() { await onCleanup(); return }

                onStateChange(.enhancing)
                let textForAI = promptDetectionResult?.processedText ?? text

                do {
                    let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(textForAI)
                    logger.notice("📝 AI enhancement: \(enhancedText, privacy: .public)")
                    finalPastedText = enhancedText
                    persistencePayload = PersistencePayload(
                        timestamp: recordedAt,
                        text: text,
                        enhancedText: enhancedText,
                        duration: actualDuration,
                        audioFileURL: audioURL.absoluteString,
                        transcriptionModelName: model.displayName,
                        aiEnhancementModelName: enhancementService.getAIService()?.currentModel,
                        promptName: promptName,
                        transcriptionDuration: transcriptionDuration,
                        enhancementDuration: enhancementDuration,
                        aiRequestSystemMessage: enhancementService.lastSystemMessageSent,
                        aiRequestUserMessage: enhancementService.lastUserMessageSent,
                        powerModeName: powerModeName,
                        powerModeEmoji: powerModeEmoji,
                        transcriptionStatus: .completed
                    )
                } catch {
                    if shouldCancel() { await onCleanup(); return }
                    persistencePayload = PersistencePayload(
                        timestamp: recordedAt,
                        text: text,
                        enhancedText: nil,
                        duration: actualDuration,
                        audioFileURL: audioURL.absoluteString,
                        transcriptionModelName: model.displayName,
                        aiEnhancementModelName: nil,
                        promptName: nil,
                        transcriptionDuration: transcriptionDuration,
                        enhancementDuration: nil,
                        aiRequestSystemMessage: nil,
                        aiRequestUserMessage: nil,
                        powerModeName: powerModeName,
                        powerModeEmoji: powerModeEmoji,
                        transcriptionStatus: .completed
                    )
                }
            } else {
                persistencePayload = PersistencePayload(
                    timestamp: recordedAt,
                    text: text,
                    enhancedText: nil,
                    duration: actualDuration,
                    audioFileURL: audioURL.absoluteString,
                    transcriptionModelName: model.displayName,
                    aiEnhancementModelName: nil,
                    promptName: nil,
                    transcriptionDuration: transcriptionDuration,
                    enhancementDuration: nil,
                    aiRequestSystemMessage: nil,
                    aiRequestUserMessage: nil,
                    powerModeName: powerModeName,
                    powerModeEmoji: powerModeEmoji,
                    transcriptionStatus: .completed
                )
            }

        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let recoverySuggestion = (error as? LocalizedError)?.recoverySuggestion ?? ""
            let fullErrorText = recoverySuggestion.isEmpty ? errorDescription : "\(errorDescription) \(recoverySuggestion)"

            let audioAsset = AVURLAsset(url: audioURL)
            let actualDuration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0
            persistencePayload = PersistencePayload(
                timestamp: recordedAt,
                text: "Transcription Failed: \(fullErrorText)",
                enhancedText: nil,
                duration: actualDuration,
                audioFileURL: audioURL.absoluteString,
                transcriptionModelName: model.displayName,
                aiEnhancementModelName: nil,
                promptName: nil,
                transcriptionDuration: nil,
                enhancementDuration: nil,
                aiRequestSystemMessage: nil,
                aiRequestUserMessage: nil,
                powerModeName: nil,
                powerModeEmoji: nil,
                transcriptionStatus: .failed
            )
        }

        if shouldCancel() { await onCleanup(); return }

        if let textToPaste = finalPastedText,
           persistencePayload?.isCompleted == true {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                let appendSpace = UserDefaults.standard.bool(forKey: "AppendTrailingSpace")
                CursorPaster.pasteAtCursor(textToPaste + (appendSpace ? " " : ""))

                let powerMode = PowerModeManager.shared
                if let activeConfig = powerMode.currentActiveConfiguration, activeConfig.isAutoSendEnabled {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        CursorPaster.pressEnter()
                    }
                }
            }
        }

        if let result = promptDetectionResult,
           let enhancementService,
           result.shouldEnableAI {
            await promptDetectionService.restoreOriginalSettings(result, to: enhancementService)
        }

        await onDismiss()

        if let persistencePayload {
            persistTranscription(
                payload: persistencePayload,
                audioURL: audioURL
            )
        }
    }

    private func persistTranscription(payload: PersistencePayload, audioURL: URL) {
        let modelContainer = modelContext.container
        let logger = self.logger

        Task.detached(priority: .utility) {
            let backgroundContext = ModelContext(modelContainer)
            let transcription = payload.makeTranscription()
            backgroundContext.insert(transcription)

            do {
                try backgroundContext.save()

                if payload.isCompleted {
                    RecentBenchmarkCorpusService.shared.captureCompletedRecording(
                        transcription: transcription,
                        audioURL: audioURL
                    )
                }

                await MainActor.run {
                    NotificationCenter.default.post(name: .transcriptionCompleted, object: nil)
                }
            } catch {
                logger.error("❌ Failed to persist transcription: \(error.localizedDescription, privacy: .public)")
            }
        }
    }
}
