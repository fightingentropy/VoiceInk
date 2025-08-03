import Foundation
import AVFoundation
import FluidAudio
import os.log



class ParakeetTranscriptionService: TranscriptionService {
    private var asrManager: AsrManager?
    private let customModelsDirectory: URL?
    @Published var isModelLoaded = false
    
    // Logger for Parakeet transcription service
    private let logger = Logger(subsystem: "com.voiceink.app", category: "ParakeetTranscriptionService")
    
    init(customModelsDirectory: URL? = nil) {
        self.customModelsDirectory = customModelsDirectory
        logger.notice("🦜 ParakeetTranscriptionService initialized with directory: \(customModelsDirectory?.path ?? "default")")
    }

    func loadModel() async throws {
        if isModelLoaded {
            return
        }

        logger.notice("🦜 Starting Parakeet model loading")
        
        do {
            let asrConfig = ASRConfig(
                maxSymbolsPerFrame: 3,
                realtimeMode: true,
                chunkSizeMs: 1500,
                tdtConfig: TdtConfig(
                    durations: [0, 1, 2, 3, 4],
                    maxSymbolsPerStep: 3
                )
            )
            asrManager = AsrManager(config: asrConfig)
            
            let models: AsrModels
            if let customDirectory = customModelsDirectory {
                logger.notice("🦜 Loading models from custom directory: \(customDirectory.path)")
                models = try await AsrModels.downloadAndLoad(to: customDirectory)
            } else {
                logger.notice("🦜 Loading models from default directory")
                models = try await AsrModels.downloadAndLoad()
            }
            
            // Check vocabulary file before initialization
            let vocabPath = getVocabularyPath()
            let vocabExists = FileManager.default.fileExists(atPath: vocabPath.path)
            logger.notice("🦜 Vocabulary file exists at \(vocabPath.lastPathComponent): \(vocabExists)")
            
            if vocabExists {
                do {
                    let vocabData = try Data(contentsOf: vocabPath)
                    let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: String] ?? [:]
                    logger.notice("🦜 Vocabulary loaded with \(vocabDict.count) entries")
                } catch {
                    logger.notice("🦜 Failed to parse vocabulary file: \(error.localizedDescription)")
                }
            }
            
            try await asrManager?.initialize(models: models)
            isModelLoaded = true
            logger.notice("🦜 Parakeet model loaded successfully")
            
        } catch {
            logger.notice("🦜 Failed to load Parakeet model: \(error.localizedDescription)")
            isModelLoaded = false
            asrManager = nil
            throw error
        }
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        do {

            if !isModelLoaded {
                try await loadModel()
            }
            
            guard let asrManager = asrManager else {
                logger.notice("🦜 ASR manager is nil after model loading")
                throw NSError(domain: "ParakeetTranscriptionService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize ASR manager."])
            }

            logger.notice("🦜 Starting Parakeet transcription")
            let audioSamples = try readAudioSamples(from: audioURL)
            logger.notice("🦜 Audio samples loaded: \(audioSamples.count) samples")
            
            let result = try await asrManager.transcribe(audioSamples)
            logger.notice("🦜 Parakeet transcription completed")
            
            // Check for empty results (vocabulary issue indicator)
            if result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                logger.notice("🦜 Warning: Empty transcription result for \(audioSamples.count) samples - possible vocabulary issue")
            }
            
            if UserDefaults.standard.object(forKey: "IsTextFormattingEnabled") as? Bool ?? true {
                return WhisperTextFormatter.format(result.text)
            }
            return result.text
        } catch {
            logger.notice("🦜 Parakeet transcription failed: \(error.localizedDescription)")
            let errorMessage = error.localizedDescription
            await MainActor.run {
                NotificationManager.shared.showNotification(
                    title: "Transcription Failed: \(errorMessage)",
                    type: .error
                )
            }
            return ""
        }
    }

    private func readAudioSamples(from url: URL) throws -> [Float] {
        logger.notice("🦜 Reading audio file: \(url.lastPathComponent)")
        let data = try Data(contentsOf: url)
        logger.notice("🦜 Audio file size: \(data.count) bytes")
        
        // A basic check, assuming a more robust check happens elsewhere.
        guard data.count > 44 else { 
            logger.notice("🦜 Warning: Audio file too small (\(data.count) bytes), expected > 44 bytes")
            return [] 
        }

        let floats = stride(from: 44, to: data.count, by: 2).map {
            return data[$0..<$0 + 2].withUnsafeBytes {
                let short = Int16(littleEndian: $0.load(as: Int16.self))
                return max(-1.0, min(Float(short) / 32767.0, 1.0))
            }
        }
        
        logger.notice("🦜 Processed audio: \(floats.count) samples from \(data.count) bytes")
        
        // Check if we have enough samples for transcription (minimum 16,000 samples = 1 second at 16kHz)
        if floats.count < 16000 {
            logger.notice("🦜 Warning: Audio too short (\(floats.count) samples), minimum 16,000 required")
        }
        
        return floats
    }
    
    // Helper function to get vocabulary path based on model directory
    private func getVocabularyPath() -> URL {
        if let customDirectory = customModelsDirectory {
            return customDirectory.appendingPathComponent("parakeet_vocab.json")
        } else {
            let applicationSupportURL = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!
            return applicationSupportURL
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
                .appendingPathComponent("parakeet-tdt-0.6b-v2-coreml", isDirectory: true)
                .appendingPathComponent("parakeet_vocab.json")
        }
    }
} 