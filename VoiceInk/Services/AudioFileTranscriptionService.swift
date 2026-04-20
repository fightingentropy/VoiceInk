import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import os

@MainActor
class AudioTranscriptionService: ObservableObject {
    @Published var isTranscribing = false
    @Published var currentError: TranscriptionError?

    private let modelContext: ModelContext
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "AudioTranscriptionService")
    private let serviceRegistry: TranscriptionServiceRegistry

    enum TranscriptionError: Error {
        case noAudioFile
        case transcriptionFailed
        case modelNotLoaded
        case invalidAudioFormat
    }

    init(modelContext: ModelContext, engine: VoiceInkEngine) {
        self.modelContext = modelContext
        self.serviceRegistry = TranscriptionServiceRegistry(modelProvider: engine.whisperModelManager)
    }

    init(modelContext: ModelContext, serviceRegistry: TranscriptionServiceRegistry) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
    }
    
    func retranscribeAudio(from url: URL, using model: any TranscriptionModel) async throws -> Transcription {
        guard model.supportsAudioFileTranscription else {
            throw TranscriptionCapabilityError.audioFileInputUnsupported(modelName: model.displayName)
        }

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TranscriptionError.noAudioFile
        }
        
        await MainActor.run {
            isTranscribing = true
        }
        
        do {
            let transcriptionStart = Date()
            var text = try await serviceRegistry.transcribe(audioURL: url, model: model)
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)
            text = TranscriptionOutputFilter.filter(text)
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)

            if UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled") {
                text = WhisperTextFormatter.format(text)
            }

            text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)
            logger.notice("✅ Word replacements applied")

            let audioAsset = AVURLAsset(url: url)
            let duration = CMTimeGetSeconds(try await audioAsset.load(.duration))
            let recordingsDirectory = AppStoragePaths.recordingsDirectory

            let fileName = "retranscribed_\(UUID().uuidString).wav"
            let permanentURL = recordingsDirectory.appendingPathComponent(fileName)

            do {
                try AppStoragePaths.createDirectoryIfNeeded(at: recordingsDirectory)
                try FileManager.default.copyItem(at: url, to: permanentURL)
            } catch {
                logger.error("❌ Failed to create permanent copy of audio: \(error.localizedDescription, privacy: .public)")
                isTranscribing = false
                throw error
            }
            
            let permanentURLString = permanentURL.absoluteString

            let newTranscription = Transcription(
                text: text,
                duration: duration,
                audioFileURL: permanentURLString,
                transcriptionModelName: model.displayName,
                transcriptionDuration: transcriptionDuration
            )
            modelContext.insert(newTranscription)
            do {
                try modelContext.save()
                NotificationCenter.default.post(name: .transcriptionCreated, object: newTranscription)
                NotificationCenter.default.post(name: .transcriptionCompleted, object: newTranscription)
            } catch {
                logger.error("❌ Failed to save transcription: \(error.localizedDescription, privacy: .public)")
            }

            await MainActor.run {
                isTranscribing = false
            }

            return newTranscription
        } catch {
            logger.error("❌ Transcription failed: \(error.localizedDescription, privacy: .public)")
            currentError = .transcriptionFailed
            isTranscribing = false
            throw error
        }
    }
}
