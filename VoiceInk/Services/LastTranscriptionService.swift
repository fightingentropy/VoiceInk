import Foundation
import SwiftData
import os

class LastTranscriptionService: ObservableObject {
    private static let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "LastTranscriptionService")

    static func getLastTranscription(from modelContext: ModelContext) -> Transcription? {
        var descriptor = FetchDescriptor<Transcription>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = 1
        
        do {
            let transcriptions = try modelContext.fetch(descriptor)
            return transcriptions.first
        } catch {
            logger.error("Error fetching last transcription: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }
    
    static func copyLastTranscription(from modelContext: ModelContext) {
        guard let lastTranscription = getLastTranscription(from: modelContext) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No transcription available",
                    type: .error
                )
            }
            return
        }
        
        let textToCopy = lastTranscription.text
        
        let success = ClipboardManager.copyToClipboard(textToCopy)
        
        Task { @MainActor in
            if success {
                NotificationManager.shared.showNotification(
                    title: "Last transcription copied",
                    type: .success
                )
            } else {
                NotificationManager.shared.showNotification(
                    title: "Failed to copy transcription",
                    type: .error
                )
            }
        }
    }

    static func pasteLastTranscription(from modelContext: ModelContext) {
        guard let lastTranscription = getLastTranscription(from: modelContext) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No transcription available",
                    type: .error
                )
            }
            return
        }
        
        let textToPaste = lastTranscription.text

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            CursorPaster.pasteAtCursor(textToPaste)
        }
    }
    
    @MainActor
    static func retryLastTranscription(from modelContext: ModelContext, transcriptionModelManager: TranscriptionModelManager, serviceRegistry: TranscriptionServiceRegistry) {
        guard let lastTranscription = getLastTranscription(from: modelContext),
              let audioURLString = lastTranscription.audioFileURL,
              let audioURL = URL(string: audioURLString),
              FileManager.default.fileExists(atPath: audioURL.path) else {
            NotificationManager.shared.showNotification(
                title: "Cannot retry: Audio file not found",
                type: .error
            )
            return
        }

        guard let currentModel = transcriptionModelManager.currentTranscriptionModel else {
            NotificationManager.shared.showNotification(
                title: "No transcription model selected",
                type: .error
            )
            return
        }
        guard currentModel.supportsAudioFileTranscription else {
            NotificationManager.shared.showNotification(
                title: TranscriptionCapabilityError.audioFileInputUnsupported(modelName: currentModel.displayName).localizedDescription,
                type: .error
            )
            return
        }

        let transcriptionService = AudioTranscriptionService(
            modelContext: modelContext,
            serviceRegistry: serviceRegistry
        )

        Task {
            do {
                let newTranscription = try await transcriptionService.retranscribeAudio(from: audioURL, using: currentModel)

                _ = ClipboardManager.copyToClipboard(newTranscription.text)

                NotificationManager.shared.showNotification(
                    title: "Copied to clipboard",
                    type: .success
                )
            } catch {
                NotificationManager.shared.showNotification(
                    title: "Retry failed: \(error.localizedDescription)",
                    type: .error
                )
            }
        }
    }
}
