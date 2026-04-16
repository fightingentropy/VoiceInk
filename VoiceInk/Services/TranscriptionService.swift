import Foundation

enum TranscriptionCapabilityError: LocalizedError {
    case audioFileInputUnsupported(modelName: String)
    case recorderUnsupported(modelName: String)

    var errorDescription: String? {
        switch self {
        case let .audioFileInputUnsupported(modelName):
            return "\(modelName) only supports live recorder transcription."
        case let .recorderUnsupported(modelName):
            return "\(modelName) does not support recorder-based transcription."
        }
    }
}

protocol RecorderTranscriptionService: AnyObject, Sendable {}

/// File-capable transcription services implement this protocol.
/// Recorder-only models such as Cohere and Voxtral intentionally do not conform.
protocol TranscriptionService: RecorderTranscriptionService {
    /// Transcribes the audio from a given file URL.
    ///
    /// - Parameters:
    ///   - audioURL: The URL of the audio file to transcribe.
    ///   - model: The `TranscriptionModel` to use for transcription. This provides context about the provider (local, OpenAI, etc.).
    /// - Returns: The transcribed text as a `String`.
    /// - Throws: An error if the transcription fails.
    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String
}

/// Optional fast path for services that can consume the recorder's native 16 kHz mono PCM chunks
/// directly instead of reopening the finalized WAV from disk.
protocol PCMBufferTranscriptionService: RecorderTranscriptionService {
    func transcribe(recordedPCMBuffer: Data, sampleRate: Int, model: any TranscriptionModel) async throws -> String
}
