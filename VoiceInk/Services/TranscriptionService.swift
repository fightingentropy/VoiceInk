import Foundation

enum TranscriptionCapabilityError: LocalizedError {
    case audioFileInputUnsupported(modelName: String)

    var errorDescription: String? {
        switch self {
        case let .audioFileInputUnsupported(modelName):
            return "\(modelName) only supports live recorder transcription."
        }
    }
}

protocol RecorderTranscriptionService: AnyObject, Sendable {}

/// A protocol defining the interface for a transcription service.
/// This allows for a unified way to handle both local and cloud-based transcription models.
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
