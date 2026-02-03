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
    private let chunkBuffer = ChunkBuffer()
    private var state: StreamingState = .idle

    /// Whether the streaming connection is fully established and actively sending.
    var isActive: Bool { state == .streaming || state == .committing }

    /// Start a streaming transcription session for the given model.
    func startStreaming(model: any TranscriptionModel) async throws {
        state = .connecting

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

        // Send commit to finalize transcription
        do {
            try await provider.commit()
        } catch {
            logger.error("Failed to send commit: \(error.localizedDescription)")
            state = .failed
            await cleanupStreaming()
            throw error
        }

        // Wait for the committed_transcript event with a timeout
        let finalText = await waitForCommittedTranscript(provider: provider)

        state = .done
        await cleanupStreaming()

        return finalText
    }

    /// Cancels the streaming session without waiting for results.
    func cancel() {
        state = .cancelled
        sendTask?.cancel()
        sendTask = nil

        let providerToDisconnect = provider
        provider = nil

        Task {
            chunkBuffer.clear()
            await providerToDisconnect?.disconnect()
        }

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

    private func waitForCommittedTranscript(provider: StreamingTranscriptionProvider) async -> String {
        let events = provider.transcriptionEvents

        return await withTaskGroup(of: String.self) { group in
            // Task 1: Listen for the committed transcript
            group.addTask { [logger] in
                var lastPartial = ""
                for await event in events {
                    switch event {
                    case .committed(let text):
                        logger.notice("Received final committed transcript")
                        return text
                    case .partial(let text):
                        lastPartial = text
                    case .error(let error):
                        logger.error("Streaming error while waiting for commit: \(error.localizedDescription)")
                        return lastPartial
                    case .sessionStarted:
                        break
                    }
                }
                return lastPartial
            }

            // Task 2: Timeout after 10 seconds
            group.addTask {
                try? await Task.sleep(nanoseconds: 10_000_000_000)
                return ""
            }

            // Whichever finishes first wins
            var result = ""
            if let first = await group.next() {
                result = first
            }
            group.cancelAll()

            // If the timeout won (empty string), give the event task a moment
            // to return any accumulated partial text after cancellation.
            if result.isEmpty, let second = await group.next() {
                result = second
            }

            if result.isEmpty {
                logger.warning("No transcript received from streaming")
            }
            return result
        }
    }

    private func cleanupStreaming() async {
        sendTask?.cancel()
        sendTask = nil
        await provider?.disconnect()
        provider = nil
        state = .idle
        chunkBuffer.clear()
    }
}
