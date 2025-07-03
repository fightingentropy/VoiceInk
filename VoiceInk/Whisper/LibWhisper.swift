import Foundation
#if canImport(whisper)
import whisper
#else
#error("Unable to import whisper module. Please check your project configuration.")
#endif
import os

enum WhisperError: Error {
    case couldNotInitializeContext
}

// Meet Whisper C++ constraint: Don't access from more than one thread at a time.
actor WhisperContext {
    private var context: OpaquePointer?
    private var languageCString: [CChar]?
    private var prompt: String?
    private var promptCString: [CChar]?
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "WhisperContext")

    private init() {
        // Private initializer without context
    }

    init(context: OpaquePointer) {
        self.context = context
    }

    deinit {
        if let context = context {
            whisper_free(context)
        }
    }

    func fullTranscribe(samples: [Float]) async {
        guard let context = context else { return }
        
        // Leave 2 processors free (i.e. the high-efficiency cores).
        let maxThreads = max(1, min(8, cpuCount() - 2))
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        
        // Read language directly from UserDefaults
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"
        if selectedLanguage != "auto" {
            languageCString = Array(selectedLanguage.utf8CString)
            params.language = languageCString?.withUnsafeBufferPointer { ptr in
                ptr.baseAddress
            }
            logger.notice("🌐 Using language: \(selectedLanguage)")
        } else {
            languageCString = nil
            params.language = nil
            logger.notice("🌐 Using auto language detection")
        }
        
        if prompt != nil {
            promptCString = Array(prompt!.utf8CString)
            params.initial_prompt = promptCString?.withUnsafeBufferPointer { ptr in
                ptr.baseAddress
            }
            logger.notice("💬 Using prompt for transcription in language: \(selectedLanguage)")
        } else {
            promptCString = nil
            params.initial_prompt = nil
        }
        
        params.print_realtime   = true
        params.print_progress   = false
        params.print_timestamps = true
        params.print_special    = false
        params.translate        = false
        params.n_threads        = Int32(maxThreads)
        params.offset_ms        = 0
        params.no_context       = true
        params.single_segment   = false

        whisper_reset_timings(context)
        logger.notice("⚙️ Starting whisper transcription with VAD: \(params.vad ? "ENABLED" : "DISABLED")")
        
        if let vadModelPath = await VADModelManager.shared.getModelPath() {
            logger.notice("🎤 VAD is ENABLED - Successfully retrieved VAD model path: \(vadModelPath)")
            params.vad = true
            params.vad_model_path = (vadModelPath as NSString).utf8String
            
            var vadParams = whisper_vad_default_params()
            vadParams.min_speech_duration_ms = 500
            vadParams.min_silence_duration_ms = 500
            vadParams.samples_overlap = 0.1
            params.vad_params = vadParams
            
            logger.notice("🎤 VAD configured with parameters: min_speech=500ms, min_silence=500ms, overlap=10%")
            logger.notice("🎤 VAD will be used for voice activity detection during transcription")
        } else {
            logger.notice("🎤 VAD is DISABLED - VAD model path not found, proceeding without VAD")
            params.vad = false
            logger.notice("🎤 Transcription will process entire audio without voice activity detection")
        }
        
        samples.withUnsafeBufferPointer { samplesBuffer in
            if whisper_full(context, params, samplesBuffer.baseAddress, Int32(samplesBuffer.count)) != 0 {
                self.logger.error("Failed to run whisper_full")
            } else {
                if params.vad {
                    self.logger.notice("✅ Whisper transcription completed successfully with VAD processing")
                } else {
                    self.logger.notice("✅ Whisper transcription completed successfully without VAD")
                }
            }
        }
        
        languageCString = nil
        promptCString = nil
    }

    func getTranscription() -> String {
        guard let context = context else { return "" }
        var transcription = ""
        for i in 0..<whisper_full_n_segments(context) {
            transcription += String(cString: whisper_full_get_segment_text(context, i))
        }
        // Apply hallucination filtering
        let filteredTranscription = WhisperHallucinationFilter.filter(transcription)

        return filteredTranscription
    }

    static func createContext(path: String) async throws -> WhisperContext {
        // Create empty context first
        let whisperContext = WhisperContext()
        
        // Initialize the context within the actor's isolated context
        try await whisperContext.initializeModel(path: path)
        
        return whisperContext
    }
    
    private func initializeModel(path: String) throws {
        var params = whisper_context_default_params()
        #if targetEnvironment(simulator)
        params.use_gpu = false
        logger.notice("🖥️ Running on simulator, using CPU")
        #endif
        
        let context = whisper_init_from_file_with_params(path, params)
        if let context {
            self.context = context
        } else {
            logger.error("❌ Couldn't load model at \(path)")
            throw WhisperError.couldNotInitializeContext
        }
    }

    func releaseResources() {
        if let context = context {
            whisper_free(context)
            self.context = nil
        }
        languageCString = nil
    }

    func setPrompt(_ prompt: String?) {
        self.prompt = prompt
        logger.notice("💬 Prompt set: \(prompt ?? "none")")
    }
}

fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}
