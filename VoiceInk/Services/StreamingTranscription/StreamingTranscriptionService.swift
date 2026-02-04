import Foundation
import os

/// Thread-safe lock-based buffer for audio chunks, accessible from any thread.
private final class ChunkBuffer: @unchecked Sendable {
    private let storage = OSAllocatedUnfairLock(initialState: [Data]())

    func append(_ data: Data) {
        storage.withLock { $0.append(data) }
    }

    func drainAll() -> [Data] {
        storage.withLock { chunks in
            let result = chunks
            chunks.removeAll()
            return result
        }
    }

    func clear() {
        storage.withLock { $0.removeAll() }
    }
}

/// Lifecycle states for a streaming transcription session.
enum StreamingState {
    case idle
    case connecting
    case streaming
    case committing
    case done
    case failed
    case cancelled
}

/// Manages a streaming transcription lifecycle: buffers audio chunks, sends them to the provider, and collects the final text on commit.
@MainActor
class StreamingTranscriptionService {

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "StreamingTranscriptionService")
    private var provider: StreamingTranscriptionProvider?
    private var sendTask: Task<Void, Never>?
    private var eventConsumerTask: Task<Void, Never>?
    private let chunkBuffer = ChunkBuffer()
    private var state: StreamingState = .idle
    private var committedSegments: [String] = []

    /// Whether the streaming connection is fully established and actively sending.
    var isActive: Bool { state == .streaming || state == .committing }

    /// Start a streaming transcription session for the given model.
    func startStreaming(model: any TranscriptionModel) async throws {
        state = .connecting
        committedSegments = []

        let provider = createProvider(for: model)
        self.provider = provider

        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"

        try await provider.connect(model: model, language: selectedLanguage)

        // If cancel() was called while we were awaiting the connection, tear down immediately.
        if state == .cancelled {
            await provider.disconnect()
            self.provider = nil
            return
        }

        state = .streaming
        startSendLoop()
        startEventConsumer()

        logger.notice("Streaming started for model: \(model.displayName)")
    }

    /// Buffers an audio chunk for sending. Safe to call from the audio callback thread.
    nonisolated func sendAudioChunk(_ data: Data) {
        chunkBuffer.append(data)
    }

    /// Stops streaming, commits remaining audio, and returns the final transcribed text.
    func stopAndGetFinalText() async throws -> String {
        guard let provider = provider, state == .streaming else {
            throw StreamingTranscriptionError.notConnected
        }

        state = .committing

        // Drain any remaining buffered chunks
        await drainRemainingChunks()

        let segmentCountBeforeCommit = committedSegments.count

        // Send commit to finalize any remaining audio
        do {
            try await provider.commit()
        } catch {
            logger.error("Failed to send commit: \(error.localizedDescription)")
            state = .failed
            await cleanupStreaming()
            throw error
        }

        // Wait for the committed segment from our explicit commit
        let finalText = await waitForFinalCommit(afterIndex: segmentCountBeforeCommit)

        state = .done
        await cleanupStreaming()

        return finalText
    }

    /// Cancels the streaming session without waiting for results.
    func cancel() {
        state = .cancelled
        eventConsumerTask?.cancel()
        eventConsumerTask = nil
        sendTask?.cancel()
        sendTask = nil

        let providerToDisconnect = provider
        provider = nil

        Task {
            chunkBuffer.clear()
            await providerToDisconnect?.disconnect()
        }

        committedSegments = []
        logger.notice("Streaming cancelled")
    }

    // MARK: - Private

    private func createProvider(for model: any TranscriptionModel) -> StreamingTranscriptionProvider {
        switch model.provider {
        case .elevenLabs:
            return ElevenLabsStreamingProvider()
        default:
            fatalError("Unsupported streaming provider: \(model.provider). Check supportsStreaming() before calling startStreaming().")
        }
    }

    private func startSendLoop() {
        let buffer = chunkBuffer
        let provider = provider

        sendTask = Task.detached { [weak self] in
            while !Task.isCancelled {
                let chunks = buffer.drainAll()

                if !chunks.isEmpty {
                    // Finish sending the entire batch before checking cancellation
                    for chunk in chunks {
                        do {
                            try await provider?.sendAudioChunk(chunk)
                        } catch {
                            let desc = error.localizedDescription
                            await MainActor.run {
                                self?.logger.error("Failed to send audio chunk: \(desc)")
                            }
                        }
                    }
                }

                // Small sleep to batch chunks and avoid excessive sends
                try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
            }
        }
    }

    private func drainRemainingChunks() async {
        sendTask?.cancel()
        sendTask = nil

        // Send any chunks that arrived after the send loop's last drain
        let remaining = chunkBuffer.drainAll()

        for chunk in remaining {
            do {
                try await provider?.sendAudioChunk(chunk)
            } catch {
                logger.error("Failed to send remaining chunk: \(error.localizedDescription)")
            }
        }
    }

    /// Consumes transcription events throughout the session, accumulating committed segments.
    private func startEventConsumer() {
        guard let provider = provider else { return }
        let events = provider.transcriptionEvents

        eventConsumerTask = Task { [weak self] in
            for await event in events {
                guard let self = self else { break }
                switch event {
                case .committed(let text):
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else {
                        self.logger.notice("Skipping empty committed segment")
                        break
                    }
                    self.committedSegments.append(trimmed)
                    self.logger.notice("Accumulated committed segment #\(self.committedSegments.count): \(trimmed.prefix(60))…")
                case .partial, .sessionStarted:
                    break
                case .error(let error):
                    self.logger.error("Streaming event error: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Polls until a new committed segment arrives beyond `afterIndex`, with a 10-second timeout.
    private func waitForFinalCommit(afterIndex expectedCount: Int) async -> String {
        let timeoutNs: UInt64 = 10_000_000_000 // 10 seconds
        let pollIntervalNs: UInt64 = 100_000_000 // 100ms
        var elapsed: UInt64 = 0

        while elapsed < timeoutNs {
            if committedSegments.count > expectedCount {
                logger.notice("Received final committed transcript (total segments: \(self.committedSegments.count))")
                return committedSegments.joined(separator: " ")
            }
            try? await Task.sleep(nanoseconds: pollIntervalNs)
            elapsed += pollIntervalNs
        }

        // Timeout — return whatever we accumulated
        if !committedSegments.isEmpty {
            logger.warning("Timeout waiting for final commit, returning \(self.committedSegments.count) accumulated segment(s)")
            return committedSegments.joined(separator: " ")
        }

        logger.warning("No transcript received from streaming")
        return ""
    }

    private func cleanupStreaming() async {
        eventConsumerTask?.cancel()
        eventConsumerTask = nil
        sendTask?.cancel()
        sendTask = nil
        await provider?.disconnect()
        provider = nil
        state = .idle
        chunkBuffer.clear()
        committedSegments = []
    }
}
