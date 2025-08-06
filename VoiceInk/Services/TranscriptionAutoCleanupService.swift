import Foundation
import SwiftData
import OSLog

/// A service that automatically deletes transcriptions when "Do Not Maintain Transcript History" is enabled
class TranscriptionAutoCleanupService {
    static let shared = TranscriptionAutoCleanupService()
    
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "TranscriptionAutoCleanupService")
    private var modelContext: ModelContext?
    
    private init() {}
    
    /// Start monitoring for new transcriptions and auto-delete if needed
    func startMonitoring(modelContext: ModelContext) {
        self.modelContext = modelContext
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleTranscriptionCreated(_:)),
            name: .transcriptionCreated,
            object: nil
        )
        
        logger.info("TranscriptionAutoCleanupService started monitoring")
    }
    
    /// Stop monitoring for transcriptions
    func stopMonitoring() {
        NotificationCenter.default.removeObserver(self, name: .transcriptionCreated, object: nil)
        logger.info("TranscriptionAutoCleanupService stopped monitoring")
    }
    
    @objc private func handleTranscriptionCreated(_ notification: Notification) {
        // Check if no-retention mode is enabled
        guard UserDefaults.standard.bool(forKey: "DoNotMaintainTranscriptHistory") else {
            return
        }
        
        guard let transcription = notification.object as? Transcription,
              let modelContext = self.modelContext else {
            logger.error("Invalid transcription or missing model context")
            return
        }
        
        logger.info("Auto-deleting transcription for zero data retention")
        
        // Delete the audio file if it exists
        if let urlString = transcription.audioFileURL,
           let url = URL(string: urlString) {
            do {
                try FileManager.default.removeItem(at: url)
                logger.debug("Deleted audio file: \(url.lastPathComponent)")
            } catch {
                logger.error("Failed to delete audio file: \(error.localizedDescription)")
            }
        }
        
        // Delete the transcription from the database
        modelContext.delete(transcription)
        
        do {
            try modelContext.save()
            logger.debug("Successfully deleted transcription from database")
        } catch {
            logger.error("Failed to save after transcription deletion: \(error.localizedDescription)")
        }
    }
}