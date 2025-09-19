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
    }

    func loadModel() async throws {
        if isModelLoaded {
            return
        }

		
        
        do {
         
            asrManager = AsrManager(config: .default) 
            let models: AsrModels
			if let customDirectory = customModelsDirectory {
				logger.notice("🦜 Loading Parakeet models from: \(customDirectory.path)")
				models = try await AsrModels.downloadAndLoad(to: customDirectory)
			} else {
				logger.notice("🦜 Loading Parakeet models from default directory")
				models = try await AsrModels.downloadAndLoad()
			}
            
            try await asrManager?.initialize(models: models)
			isModelLoaded = true
			logger.notice("🦜 Parakeet model loaded successfully")
            
		} catch {
            let description = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            logger.error("🦜 Failed to load Parakeet model: \(description)")
            isModelLoaded = false
            asrManager = nil
            throw error
        }
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        if asrManager == nil || !isModelLoaded {
            try await loadModel()
        }

		guard let asrManager = asrManager else {
			throw ASRError.notInitialized
		}
        
        let audioSamples = try readAudioSamples(from: audioURL)

        // Use full audio for transcription
        let speechAudio: [Float] = audioSamples

        let result = try await asrManager.transcribe(speechAudio)
		
        
        // Reset decoder state and cleanup after transcription to avoid blocking the transcription start
		Task {
			asrManager.cleanup()
			isModelLoaded = false
			logger.notice("🦜 Parakeet ASR models cleaned up from memory")
		}
        
        let text = result.text

        return text
    }

    private func readAudioSamples(from url: URL) throws -> [Float] {
        do {
            let data = try Data(contentsOf: url)
            
			// Check minimum file size for valid WAV header
			guard data.count > 44 else {
				throw ASRError.invalidAudioData
			}

            let floats = stride(from: 44, to: data.count, by: 2).map {
                return data[$0..<$0 + 2].withUnsafeBytes {
                    let short = Int16(littleEndian: $0.load(as: Int16.self))
                    return max(-1.0, min(Float(short) / 32767.0, 1.0))
                }
            }
            
            return floats
		} catch {
			throw ASRError.invalidAudioData
		}
    }

}
