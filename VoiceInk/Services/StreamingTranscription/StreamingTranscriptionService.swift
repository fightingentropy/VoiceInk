import Foundation
import os

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

/// Manages a streaming transcription lifecycle: buffers audio chunks, sends them to the provider, and collects the final text.
@MainActor
final class StreamingTranscriptionService {

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "StreamingTranscriptionService")
    private var provider: StreamingTranscriptionProvider?
    private var sendTask: Task<Void, Never>?
    private var eventConsumerTask: Task<Void, Never>?
    private let audioBuffer: BoundedPCMChunkBuffer
    private let bufferSignalStream: AsyncStream<Void>
    private let bufferSignalContinuation: AsyncStream<Void>.Continuation
    private var state: StreamingState = .idle
    private var committedSegments: [String] = []
    private var onPartialTranscript: (@Sendable (String) -> Void)?

    init(onPartialTranscript: (@Sendable (String) -> Void)? = nil) {
        let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "StreamingTranscriptionService")
        let (bufferSignalStream, bufferSignalContinuation) = AsyncStream.makeStream(
            of: Void.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        self.audioBuffer = BoundedPCMChunkBuffer(
            capacityBytes: BoundedPCMChunkBuffer.defaultCapacityBytes,
            logger: logger,
            label: "Streaming transcription"
        )
        self.bufferSignalStream = bufferSignalStream
        self.bufferSignalContinuation = bufferSignalContinuation
        self.onPartialTranscript = onPartialTranscript
    }

    deinit {
        onPartialTranscript = nil
        sendTask?.cancel()
        eventConsumerTask?.cancel()
        audioBuffer.close()
        bufferSignalContinuation.finish()
        commitSignal?.finish()
    }

    /// Signal used to notify `waitForFinalCommit` when a new committed segment arrives.
    private var commitSignal: AsyncStream<Void>.Continuation?

    /// Whether the streaming connection is fully established and actively sending.
    var isActive: Bool { state == .streaming || state == .committing }

    /// Start a streaming transcription session for the given model.
    func startStreaming(model: any TranscriptionModel) async throws {
        state = .connecting
        committedSegments = []

        let provider = try createProvider(for: model)
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

        logger.notice("Streaming started for model: \(model.displayName, privacy: .public)")
    }

    /// Buffers an audio chunk for sending. Safe to call from the audio callback thread.
    nonisolated func sendAudioChunk(_ data: Data) {
        audioBuffer.append(data)
        bufferSignalContinuation.yield()
    }

    /// Stops streaming, commits remaining audio, and returns the final transcribed text.
    func stopAndGetFinalText() async throws -> String {
        guard let provider = provider, state == .streaming else {
            throw StreamingTranscriptionError.notConnected
        }

        state = .committing

        // Finish the chunk source so the send loop drains remaining chunks and exits naturally.
        await drainRemainingChunks()

        // Set up the commit signal BEFORE sending commit to avoid a race with the response.
        let (signalStream, signalContinuation) = AsyncStream.makeStream(of: Void.self)
        self.commitSignal = signalContinuation

        // Send commit to finalize any remaining audio
        do {
            try await provider.commit()
        } catch {
            commitSignal?.finish()
            commitSignal = nil
            logger.error("Failed to send commit: \(error.localizedDescription, privacy: .public)")
            state = .failed
            await cleanupStreaming()
            throw error
        }

        // Wait for the server to acknowledge our commit (or timeout)
        let finalText = await waitForFinalCommit(signalStream: signalStream)

        state = .done
        await cleanupStreaming()

        return finalText
    }

    /// Cancels the streaming session without waiting for results.
    func cancel() {
        state = .cancelled
        onPartialTranscript = nil
        eventConsumerTask?.cancel()
        eventConsumerTask = nil
        sendTask?.cancel()
        sendTask = nil
        audioBuffer.close()
        audioBuffer.clear()
        bufferSignalContinuation.finish()

        // Clean up commit signal if waiting
        commitSignal?.finish()
        commitSignal = nil

        let providerToDisconnect = provider
        provider = nil

        Task { [providerToDisconnect] in
            await providerToDisconnect?.disconnect()
        }

        committedSegments = []
        logger.notice("Streaming cancelled")
    }

    // MARK: - Private

    private func createProvider(for model: any TranscriptionModel) throws -> StreamingTranscriptionProvider {
        switch model.provider {
        case .localVoxtral:
            return VoxtralNativeStreamingProvider()
        case .elevenLabs:
            return ElevenLabsStreamingProvider()
        case .xAI:
            return XAIStreamingProvider()
        default:
            throw StreamingTranscriptionError.unsupportedProvider(String(describing: model.provider))
        }
    }

    /// Consumes audio chunks from the AsyncStream and sends them to the provider.
    private func startSendLoop() {
        let provider = provider
        let signalStream = bufferSignalStream
        let audioBuffer = audioBuffer

        sendTask = Task { [weak self, provider, signalStream, audioBuffer] in
            let sendChunks: ([Data]) async -> Void = { chunks in
                for chunk in chunks {
                    do {
                        try await provider?.sendAudioChunk(chunk)
                    } catch {
                        self?.logger.error("Failed to send audio chunk: \(error.localizedDescription, privacy: .public)")
                    }
                }
            }

            for await _ in signalStream {
                await sendChunks(audioBuffer.drain())
            }

            await sendChunks(audioBuffer.drain())
        }

        bufferSignalContinuation.yield()
    }

    /// Finishes the chunk source and waits for the send loop to process all remaining buffered chunks.
    private func drainRemainingChunks() async {
        audioBuffer.close()
        bufferSignalContinuation.finish()
        await sendTask?.value
        sendTask = nil
    }

    /// Consumes transcription events throughout the session, accumulating committed segments.
    private func startEventConsumer() {
        guard let provider = provider else { return }
        let events = provider.transcriptionEvents

        eventConsumerTask = Task { [weak self, events] in
            for await event in events {
                guard let self = self else { break }
                switch event {
                case .committed(let text):
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        self.committedSegments.append(trimmed)
                    }
                    if self.state == .committing {
                        self.commitSignal?.yield()
                    }
                case .partial(let text):
                    if self.state == .streaming {
                        self.onPartialTranscript?(text)
                    }
                case .sessionStarted:
                    break
                case .error(let error):
                    self.logger.error("Streaming event error: \(error.localizedDescription, privacy: .public)")
                }
            }  
        }
    }

    /// Waits for the server to acknowledge our explicit commit, with a 10-second timeout.
    private func waitForFinalCommit(signalStream: AsyncStream<Void>) async -> String {
        let signalTask = Task {
            for await _ in signalStream {
                return true
            }
            return false
        }
        let timeoutTask = Task {
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            return false
        }

        let receivedInTime = await withTaskGroup(of: Bool.self) { group in
            group.addTask { await signalTask.value }
            group.addTask { await timeoutTask.value }

            let result = await group.next() ?? false
            signalTask.cancel()
            timeoutTask.cancel()
            group.cancelAll()
            return result
        }

        // Clean up the signal
        commitSignal?.finish()
        commitSignal = nil

        if !receivedInTime && committedSegments.isEmpty {
            logger.warning("No transcript received from streaming")
        }

        return committedSegments.isEmpty ? "" : committedSegments.joined(separator: " ")
    }

    private func cleanupStreaming() async {
        onPartialTranscript = nil
        eventConsumerTask?.cancel()
        eventConsumerTask = nil
        sendTask?.cancel()
        sendTask = nil
        audioBuffer.close()
        audioBuffer.clear()
        bufferSignalContinuation.finish()
        commitSignal?.finish()
        commitSignal = nil
        await provider?.disconnect()
        provider = nil
        state = .idle
        committedSegments = []
    }
}
