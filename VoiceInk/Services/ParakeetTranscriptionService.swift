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
         
            asrManager = AsrManager(config: .default) 
            let models: AsrModels
            if let customDirectory = customModelsDirectory {
                logger.notice("🦜 Loading models from custom directory: \(customDirectory.path)")
                models = try await AsrModels.downloadAndLoad(to: customDirectory)
            } else {
                logger.notice("🦜 Loading models from default directory")
                models = try await AsrModels.downloadAndLoad()
            }
            
            try await asrManager?.initialize(models: models)
            isModelLoaded = true
            logger.notice("🦜 Parakeet model loaded successfully")
            
        } catch let error as ASRError {
            logger.notice("🦜 Parakeet-specific error loading model: \(error.localizedDescription)")
            isModelLoaded = false
            asrManager = nil
            throw error
        } catch let error as AsrModelsError {
            logger.notice("🦜 Parakeet model management error loading model: \(error.localizedDescription)")
            isModelLoaded = false
            asrManager = nil
            throw error
        } catch {
            logger.notice("🦜 Unexpected error loading Parakeet model: \(error.localizedDescription)")
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
            logger.notice("🦜 Parakeet manager is still nil after attempting to load the model.")
            throw ASRError.notInitialized
        }
        
        let audioSamples = try readAudioSamples(from: audioURL)
        
        // Validate audio data before VAD
        guard !audioSamples.isEmpty else {
            logger.notice("🦜 Audio is empty, skipping transcription.")
            throw ASRError.invalidAudioData
        }

        // Use VAD to get speech segments
        let speechAudio: [Float]
        if let modelPath = await VADModelManager.shared.getModelPath() {
            if let vad = VoiceActivityDetector(modelPath: modelPath) {
                speechAudio = vad.process(audioSamples: audioSamples)
                logger.notice("🦜 VAD processed audio, resulting in \(speechAudio.count) samples.")
            } else {
                logger.warning("🦜 VAD could not be initialized. Transcribing original audio.")
                speechAudio = audioSamples
            }
        } else {
            logger.warning("🦜 VAD model path not found. Transcribing original audio.")
            speechAudio = audioSamples
        }
        
        // Validate audio data after VAD
        guard speechAudio.count >= 16000 else {
            logger.notice("🦜 Audio too short for transcription after VAD: \(speechAudio.count) samples")
            throw ASRError.invalidAudioData
        }
        
        let result = try await asrManager.transcribe(speechAudio)
        
        // Reset decoder state and cleanup after transcription to avoid blocking the transcription start
        Task {
            asrManager.cleanup()
            isModelLoaded = false
            logger.notice("🦜 Parakeet ASR models cleaned up from memory")
        }
        
        // Check for empty results (vocabulary issue indicator)
        if result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            logger.notice("🦜 Warning: Empty transcription result for \(audioSamples.count) samples - possible vocabulary issue")
        }
        
        if UserDefaults.standard.object(forKey: "IsTextFormattingEnabled") as? Bool ?? true {
            return WhisperTextFormatter.format(result.text)
        }
        return result.text
    }

    private func readAudioSamples(from url: URL) throws -> [Float] {
        do {
            let data = try Data(contentsOf: url)
            
            // Check minimum file size for valid WAV header
            guard data.count > 44 else {
                logger.notice("🦜 Audio file too small (\(data.count) bytes), expected > 44 bytes")
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
            logger.notice("🦜 Failed to read audio file: \(error.localizedDescription)")
            throw ASRError.invalidAudioData
        }
    }

}
