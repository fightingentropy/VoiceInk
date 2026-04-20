import Foundation
import AVFoundation
import SwiftData
import os

/// Handles the full post-recording pipeline:
/// transcribe -> filter -> format -> word-replace -> paste/dismiss -> save/history/benchmark
@MainActor
class TranscriptionPipeline {
    private let modelContext: ModelContext
    private let serviceRegistry: TranscriptionServiceRegistry
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionPipeline")

    private struct PersistencePayload: Sendable {
        let timestamp: Date
        let text: String
        let duration: TimeInterval
        let audioFileURL: String
        let transcriptionModelName: String?
        let transcriptionDuration: TimeInterval?
        let transcriptionStatus: TranscriptionStatus

        var isCompleted: Bool {
            transcriptionStatus == .completed
        }

        func makeTranscription() -> Transcription {
            let transcription = Transcription(
                text: text,
                duration: duration,
                audioFileURL: audioFileURL,
                transcriptionModelName: transcriptionModelName,
                transcriptionDuration: transcriptionDuration,
                transcriptionStatus: transcriptionStatus
            )
            transcription.timestamp = timestamp
            return transcription
        }
    }

    init(
        modelContext: ModelContext,
        serviceRegistry: TranscriptionServiceRegistry
    ) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
    }

    /// Run the full pipeline for a given transcription record.
    /// - Parameters:
    ///   - audioURL: The recorded audio file.
    ///   - recordedAt: Timestamp captured when recording ended.
    ///   - model: The transcription model to use.
    ///   - session: An active streaming session if one was prepared, otherwise nil.
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

            persistencePayload = PersistencePayload(
                timestamp: recordedAt,
                text: text,
                duration: actualDuration,
                audioFileURL: audioURL.absoluteString,
                transcriptionModelName: model.displayName,
                transcriptionDuration: transcriptionDuration,
                transcriptionStatus: .completed
            )

        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let recoverySuggestion = (error as? LocalizedError)?.recoverySuggestion ?? ""
            let fullErrorText = recoverySuggestion.isEmpty ? errorDescription : "\(errorDescription) \(recoverySuggestion)"

            let audioAsset = AVURLAsset(url: audioURL)
            let actualDuration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0
            persistencePayload = PersistencePayload(
                timestamp: recordedAt,
                text: "Transcription Failed: \(fullErrorText)",
                duration: actualDuration,
                audioFileURL: audioURL.absoluteString,
                transcriptionModelName: model.displayName,
                transcriptionDuration: nil,
                transcriptionStatus: .failed
            )
        }

        if shouldCancel() { await onCleanup(); return }

        if let textToPaste = finalPastedText,
           persistencePayload?.isCompleted == true {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                let appendSpace = UserDefaults.standard.bool(forKey: "AppendTrailingSpace")
                CursorPaster.pasteAtCursor(textToPaste + (appendSpace ? " " : ""))

            }
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
