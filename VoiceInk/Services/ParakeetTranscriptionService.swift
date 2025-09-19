import Foundation
import CoreML
import AVFoundation
import FluidAudio
import os.log



class ParakeetTranscriptionService: TranscriptionService {
    private var asrManager: AsrManager?
    private var vadManager: VadManager?
    private let customModelsDirectory: URL?
    @Published var isModelLoaded = false
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
				models = try await AsrModels.load(from: customDirectory)
			} else {
				logger.notice("🦜 Loading Parakeet models from default directory")
				let defaultDir = AsrModels.defaultCacheDirectory()
				models = try await AsrModels.load(from: defaultDir)
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

        let sampleRate = 16000.0
        let durationSeconds = Double(audioSamples.count) / sampleRate

        let speechAudio: [Float]
        if durationSeconds < 20.0 {
            speechAudio = audioSamples
        } else {
            let vadConfig = VadConfig(threshold: 0.7)
            if vadManager == nil {
                if let bundledVadURL = Bundle.main.url(forResource: ModelNames.VAD.sileroVad, withExtension: "mlmodelc") {
                    do {
                        let bundledModel = try MLModel(contentsOf: bundledVadURL)
                        vadManager = VadManager(config: vadConfig, vadModel: bundledModel)
                    } catch {
                    }
                } else {
                }
            }

            do {
                if let vadManager {
                    let segments = try await vadManager.segmentSpeechAudio(audioSamples)
                    if segments.isEmpty {
                        speechAudio = audioSamples
                    } else {
                        speechAudio = segments.flatMap { $0 }
                    }
                } else {
                    speechAudio = audioSamples
                }
            } catch {
                speechAudio = audioSamples
            }
        }

        let result = try await asrManager.transcribe(speechAudio)
		
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
