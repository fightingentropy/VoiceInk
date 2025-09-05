import Foundation
import AVFoundation
import os.log

// MARK: - C API Bridge

// Opaque pointers for the C contexts
fileprivate typealias WhisperVADContext = OpaquePointer
fileprivate typealias WhisperVADSegments = OpaquePointer

// Define the C function signatures for Swift, scoped to this file

@_silgen_name("whisper_vad_default_params")
fileprivate func whisper_vad_default_params() -> whisper_vad_params

@_silgen_name("whisper_vad_default_context_params")
fileprivate func whisper_vad_default_context_params() -> whisper_vad_context_params

@_silgen_name("whisper_vad_init_from_file_with_params")
fileprivate func whisper_vad_init_from_file_with_params(_ path_model: UnsafePointer<CChar>, _ params: whisper_vad_context_params) -> WhisperVADContext?

@_silgen_name("whisper_vad_detect_speech")
fileprivate func whisper_vad_detect_speech(_ vctx: WhisperVADContext, _ samples: UnsafePointer<Float>, _ n_samples: Int32) -> Bool

@_silgen_name("whisper_vad_n_probs")
fileprivate func whisper_vad_n_probs(_ vctx: WhisperVADContext) -> Int32

@_silgen_name("whisper_vad_probs")
fileprivate func whisper_vad_probs(_ vctx: WhisperVADContext) -> UnsafeMutablePointer<Float>

@_silgen_name("whisper_vad_segments_from_probs")
fileprivate func whisper_vad_segments_from_probs(_ vctx: WhisperVADContext, _ params: whisper_vad_params) -> WhisperVADSegments?

@_silgen_name("whisper_vad_segments_n_segments")
fileprivate func whisper_vad_segments_n_segments(_ segments: WhisperVADSegments) -> Int32

@_silgen_name("whisper_vad_segments_get_segment_t0")
fileprivate func whisper_vad_segments_get_segment_t0(_ segments: WhisperVADSegments, _ i_segment: Int32) -> Float

@_silgen_name("whisper_vad_segments_get_segment_t1")
fileprivate func whisper_vad_segments_get_segment_t1(_ segments: WhisperVADSegments, _ i_segment: Int32) -> Float

@_silgen_name("whisper_vad_free_segments")
fileprivate func whisper_vad_free_segments(_ segments: WhisperVADSegments)

@_silgen_name("whisper_vad_free")
fileprivate func whisper_vad_free(_ ctx: WhisperVADContext)

// Structs matching whisper.h, scoped to this file
fileprivate struct whisper_vad_params {
    var threshold: Float
    var min_speech_duration_ms: Int32
    var min_silence_duration_ms: Int32
    var max_speech_duration_s: Float
    var speech_pad_ms: Int32
    var samples_overlap: Float
}

fileprivate struct whisper_vad_context_params {
    var n_threads: Int32
    var use_gpu: Bool
    var gpu_device: Int32
}


// MARK: - VoiceActivityDetector Class

class VoiceActivityDetector {
    private var vadContext: WhisperVADContext
    private let logger = Logger(subsystem: "com.voiceink.app", category: "VoiceActivityDetector")

    init?(modelPath: String) {
        var contextParams = whisper_vad_default_context_params()
        contextParams.n_threads = max(1, min(8, Int32(ProcessInfo.processInfo.processorCount) - 2))
        
        guard let context = whisper_vad_init_from_file_with_params(modelPath, contextParams) else {
            logger.error("Failed to initialize VAD context.")
            return nil
        }
        self.vadContext = context
        logger.notice("VAD context initialized successfully.")
    }

    deinit {
        whisper_vad_free(vadContext)
        logger.notice("VAD context freed.")
    }

    /// Processes audio samples to detect speech segments and returns the stitched audio containing only speech.
    func process(audioSamples: [Float]) -> [Float] {
        // 1. Detect speech and get probabilities internally in the context
        let success = audioSamples.withUnsafeBufferPointer { buffer in
            whisper_vad_detect_speech(vadContext, buffer.baseAddress!, Int32(audioSamples.count))
        }

        guard success else {
            logger.error("Failed to detect speech probabilities.")
            return []
        }

        // 2. Get segments from probabilities
        var vadParams = whisper_vad_default_params()
        vadParams.threshold = 0.5
        vadParams.min_speech_duration_ms = 250
        vadParams.min_silence_duration_ms = 100
        vadParams.speech_pad_ms = 30

        guard let segments = whisper_vad_segments_from_probs(vadContext, vadParams) else {
            logger.error("Failed to get VAD segments from probabilities.")
            return []
        }
        defer {
            // Ensure segments are freed
            whisper_vad_free_segments(segments)
        }
        
        let nSegments = whisper_vad_segments_n_segments(segments)
        logger.notice("Detected \(nSegments) speech segments.")

        // 3. Stitch audio segments together
        var stitchedAudio = [Float]()
        let sampleRate = 16000 // Assuming 16kHz sample rate

        for i in 0..<nSegments {
            // Timestamps from C are mysteriously multiplied by 100, so we correct them here.
            let startTimeSec = whisper_vad_segments_get_segment_t0(segments, i) / 100.0
            let endTimeSec = whisper_vad_segments_get_segment_t1(segments, i) / 100.0

            logger.debug("Segment \(i): start=\(startTimeSec, privacy: .public)s, end=\(endTimeSec, privacy: .public)s")

            let startSample = Int(startTimeSec * Float(sampleRate))
            var endSample = Int(endTimeSec * Float(sampleRate))

            logger.debug("Segment \(i): startSample=\(startSample, privacy: .public), endSample=\(endSample, privacy: .public)")

            // Cap endSample to the audio buffer size
            if endSample > audioSamples.count {
                logger.debug("Capping endSample from \(endSample, privacy: .public) to \(audioSamples.count, privacy: .public)")
                endSample = audioSamples.count
            }

            if startSample < endSample {
                stitchedAudio.append(contentsOf: audioSamples[startSample..<endSample])
            } else {
                logger.warning("Segment \(i): Invalid sample range, skipping.")
            }
        }

        logger.notice("Stitched audio contains \(stitchedAudio.count) samples.")
        return stitchedAudio
    }
}