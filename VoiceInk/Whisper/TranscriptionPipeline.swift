import Foundation
import AppKit
import SwiftData
import os

/// Handles the full post-recording pipeline:
/// transcribe -> filter -> format -> word-replace -> paste/dismiss
@MainActor
class TranscriptionPipeline {
    private let modelContext: ModelContext
    private let serviceRegistry: TranscriptionServiceRegistry
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionPipeline")

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
        recordedAt _: Date,
        model: any TranscriptionModel,
        session: TranscriptionSession?,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: () -> Bool,
        onCleanup: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void
    ) async {
        defer {
            let discarded = EphemeralTranscriptionPolicy.discardRecording(at: audioURL)
            if discarded {
                logger.notice("Discarded the temporary recording")
            } else if FileManager.default.fileExists(atPath: audioURL.path) {
                logger.error("Could not discard the temporary recording")
            }
        }

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

        logger.notice("🔄 Starting transcription...")

        do {
            var text: String
            if let session {
                text = try await session.transcribe(audioURL: audioURL)
            } else {
                text = try await serviceRegistry.transcribe(audioURL: audioURL, model: model)
            }
            logger.notice("📝 Transcript received (\(text.count, privacy: .public) characters)")
            text = TranscriptionOutputFilter.filter(text)
            logger.notice("📝 Output filter completed (\(text.count, privacy: .public) characters)")

            if shouldCancel() { await onCleanup(); return }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)

            if UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled") {
                text = WhisperTextFormatter.format(text)
                logger.notice("📝 Transcript formatting completed")
            }

            let frontmostAppContext = Self.frontmostAppContext()

            if UserDefaults.standard.bool(forKey: "ConvertSpokenPunctuation") {
                text = SpokenPunctuationFormatter.apply(text, frontmostAppContext: frontmostAppContext)
                logger.notice("📝 Spoken punctuation conversion completed")
            }

            if UserDefaults.standard.bool(forKey: "ConvertLiteralDictationTokens") {
                text = DictationLiteralFormatter.apply(text, frontmostAppContext: frontmostAppContext)
                logger.notice("📝 Literal dictation conversion completed")
            }

            text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)
            logger.notice("📝 Word replacement completed")
            finalPastedText = text

        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let recoverySuggestion = (error as? LocalizedError)?.recoverySuggestion ?? ""
            let fullErrorText = recoverySuggestion.isEmpty ? errorDescription : "\(errorDescription) \(recoverySuggestion)"
            logger.error("❌ Transcription failed: \(fullErrorText, privacy: .public)")
        }

        if shouldCancel() { await onCleanup(); return }

        if let textToPaste = finalPastedText {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                let appendSpace = UserDefaults.standard.bool(forKey: "AppendTrailingSpace")
                var pasteText = textToPaste + (appendSpace ? " " : "")
                if UserDefaults.standard.bool(forKey: "ConvertLiteralDictationTokens") {
                    pasteText = DictationLiteralFormatter.applyTerminalLiteralAutocompleteSpacing(
                        pasteText,
                        frontmostAppContext: Self.frontmostAppContext()
                    )
                }
                CursorPaster.pasteAtCursor(pasteText)
            }
        }

        await onDismiss()
        logger.notice("Transcription completed without history persistence")
    }

    /// Gating haystack for app-aware formatting rules: app name plus bundle
    /// ID, because some bundle IDs don't contain the product name (Cursor is
    /// com.todesktop.230313mzl4w4u92, ChatGPT is com.openai.chat).
    nonisolated private static func frontmostAppContext() -> String? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        return [app.localizedName, app.bundleIdentifier].compactMap { $0 }.joined(separator: " ")
    }
}
