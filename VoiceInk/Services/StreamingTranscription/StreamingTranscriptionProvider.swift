import Foundation

/// Events emitted by a streaming transcription provider
enum StreamingTranscriptionEvent {
    case sessionStarted
    case partial(text: String)
    case committed(text: String)
    /// The provider has definitively finished emitting transcript segments for
    /// the most recent manual audio commit.
    case finalized
    case error(Error)
}

/// Describes how a provider proves that all transcript segments have arrived.
enum StreamingFinalizationMode: Sendable, Equatable {
    /// Preserve the bounded trailing quiet period for providers whose SDK does
    /// not expose an unambiguous end-of-transcript event.
    case trailingQuietPeriod
    /// Return as soon as the provider emits `.finalized` after a manual commit.
    case providerSignal
}

/// Errors specific to streaming transcription
enum StreamingTranscriptionError: LocalizedError {
    case missingAPIKey
    case connectionFailed(String)
    case timeout
    case serverError(String)
    case notConnected
    case audioBufferOverflow
    case unsupportedProvider(String)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "API key not configured for streaming transcription"
        case .connectionFailed(let message):
            return "Streaming connection failed: \(message)"
        case .timeout:
            return "Streaming transcription timed out waiting for final result"
        case .serverError(let message):
            return "Streaming server error: \(message)"
        case .notConnected:
            return "Not connected to streaming transcription service"
        case .audioBufferOverflow:
            return "Streaming audio could not be delivered without dropping buffered samples"
        case .unsupportedProvider(let provider):
            return "Streaming is not supported for provider: \(provider)"
        }
    }
}

/// Protocol for streaming transcription providers.
protocol StreamingTranscriptionProvider: AnyObject, Sendable {
    /// How this provider signals that a manual commit is fully transcribed.
    nonisolated var finalizationMode: StreamingFinalizationMode { get }

    /// Connect to the streaming transcription endpoint
    func connect(model: any TranscriptionModel, language: String?) async throws

    /// Send a chunk of raw PCM audio data (16-bit, 16kHz, mono, little-endian)
    func sendAudioChunk(_ data: Data) async throws

    /// Commit the current audio buffer to finalize transcription
    func commit() async throws

    /// Disconnect from the streaming endpoint
    func disconnect() async

    /// Stream of transcription events from the provider
    var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent> { get }
}

extension StreamingTranscriptionProvider {
    nonisolated var finalizationMode: StreamingFinalizationMode {
        .trailingQuietPeriod
    }
}
