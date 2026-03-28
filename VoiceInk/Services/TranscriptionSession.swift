import Foundation
import os

/// Encapsulates a single recording-to-transcription lifecycle (streaming or file-based).
@MainActor
protocol TranscriptionSession: AnyObject {
    /// Prepares the session. Returns an audio chunk callback for streaming, or nil for file-based.
    func prepare(model: any TranscriptionModel) async throws -> (@Sendable (Data) -> Void)?

    /// Called after recording stops. Returns the final transcribed text.
    func transcribe(audioURL: URL) async throws -> String

    /// Cancel the session and clean up resources.
    func cancel()
}

// MARK: - File-Based Session

/// File-based session: records to file, uploads after stop.
@MainActor
final class FileTranscriptionSession: TranscriptionSession {
    private static let recorderChunkSampleRate = 16_000

    private let service: any RecorderTranscriptionService
    private var model: (any TranscriptionModel)?
    private let pcmAccumulator = RecordedPCMAccumulator()

    init(service: any RecorderTranscriptionService) {
        self.service = service
    }

    func prepare(model: any TranscriptionModel) async throws -> (@Sendable (Data) -> Void)? {
        self.model = model
        pcmAccumulator.reset()

        guard service is PCMBufferTranscriptionService else {
            return nil
        }

        return { [pcmAccumulator] data in
            pcmAccumulator.append(data)
        }
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let model = model else {
            throw VoiceInkEngineError.transcriptionFailed
        }

        if let optimizedService = service as? PCMBufferTranscriptionService {
            let recordedPCMBuffer = pcmAccumulator.drain()
            guard !recordedPCMBuffer.isEmpty else {
                throw VoiceInkEngineError.transcriptionFailed
            }

            return try await optimizedService.transcribe(
                recordedPCMBuffer: recordedPCMBuffer,
                sampleRate: Self.recorderChunkSampleRate,
                model: model
            )
        }

        guard let fileService = service as? TranscriptionService else {
            throw VoiceInkEngineError.transcriptionFailed
        }

        return try await fileService.transcribe(audioURL: audioURL, model: model)
    }

    func cancel() {
        pcmAccumulator.reset()
    }
}

// MARK: - Streaming Session

/// Streaming session with automatic fallback to file-based transcription when available.
@MainActor
final class StreamingTranscriptionSession: TranscriptionSession {
    private let streamingService: StreamingTranscriptionService
    private let fallbackService: (any TranscriptionService)?
    private var model: (any TranscriptionModel)?
    private var streamingFailed = false
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "StreamingTranscriptionSession")

    init(
        streamingService: StreamingTranscriptionService,
        fallbackService: (any TranscriptionService)?
    ) {
        self.streamingService = streamingService
        self.fallbackService = fallbackService
    }

    func prepare(model: any TranscriptionModel) async throws -> (@Sendable (Data) -> Void)? {
        self.model = model
        do {
            try await streamingService.startStreaming(model: model)
            logger.notice("Streaming connected for \(model.displayName, privacy: .public)")
        } catch {
            logger.error("❌ Failed to start streaming, will fall back to batch: \(error.localizedDescription, privacy: .public)")
            streamingFailed = true
            return nil
        }

        let service = streamingService
        return { [weak service] data in
            service?.sendAudioChunk(data)
        }
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let model = model else {
            throw VoiceInkEngineError.transcriptionFailed
        }

        if !streamingFailed {
            do {
                let text = try await streamingService.stopAndGetFinalText()
                logger.notice("Streaming transcript received")
                if model.provider == .localVoxtral, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    NotificationCenter.default.post(name: .localModelDidUse, object: nil)
                }
                return text
            } catch {
                logger.error("❌ Streaming failed, falling back to batch: \(error.localizedDescription, privacy: .public)")
                streamingService.cancel()
            }
        } else {
            streamingService.cancel()
        }

        guard let fallbackService else {
            throw StreamingTranscriptionSessionError.batchFallbackUnavailable(modelName: model.displayName)
        }
        logger.notice("Using file-based fallback for \(model.displayName, privacy: .public)")
        return try await fallbackService.transcribe(audioURL: audioURL, model: model)
    }

    func cancel() {
        streamingService.cancel()
    }
}

enum StreamingTranscriptionSessionError: LocalizedError {
    case batchFallbackUnavailable(modelName: String)

    var errorDescription: String? {
        switch self {
        case let .batchFallbackUnavailable(modelName):
            return "\(modelName) is streaming-only. If live transcription startup fails, there is no batch fallback available for this model."
        }
    }
}

private final class RecordedPCMAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()

    func append(_ data: Data) {
        lock.lock()
        buffer.append(data)
        lock.unlock()
    }

    func drain() -> Data {
        lock.lock()
        defer { lock.unlock() }
        let drained = buffer
        buffer.removeAll(keepingCapacity: false)
        return drained
    }

    func reset() {
        lock.lock()
        buffer.removeAll(keepingCapacity: false)
        lock.unlock()
    }
}
