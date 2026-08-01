import AppKit
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

/// A one-shot audio pipe. Each streaming session gets a fresh instance so a closed
/// buffer or finished signal stream can never leak into the next dictation.
private final class StreamingAudioPipe: @unchecked Sendable {
    let buffer: BoundedPCMChunkBuffer
    let signals: AsyncStream<Void>

    private let continuation: AsyncStream<Void>.Continuation
    private let lock = NSLock()
    private var isClosed = false

    init(logger: Logger) {
        let (signals, continuation) = AsyncStream.makeStream(
            of: Void.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        self.signals = signals
        self.continuation = continuation
        self.buffer = BoundedPCMChunkBuffer(
            capacityBytes: BoundedPCMChunkBuffer.defaultCapacityBytes,
            logger: logger,
            label: "Streaming transcription"
        )
    }

    func send(_ data: Data) {
        lock.lock()
        guard !isClosed else {
            lock.unlock()
            return
        }
        buffer.append(data)
        continuation.yield()
        lock.unlock()
    }

    func close() {
        lock.lock()
        guard !isClosed else {
            lock.unlock()
            return
        }
        isClosed = true
        buffer.close()
        continuation.finish()
        lock.unlock()
    }

    func clear() {
        buffer.clear()
    }
}

/// Thread-safe bridge used by the real-time audio callback. The callback never
/// touches MainActor state and always targets the currently installed session.
private final class StreamingAudioRouter: @unchecked Sendable {
    private let lock = NSLock()
    private var pipe: StreamingAudioPipe?

    func install(_ pipe: StreamingAudioPipe) {
        lock.lock()
        self.pipe = pipe
        lock.unlock()
    }

    func remove(_ expectedPipe: StreamingAudioPipe) {
        lock.lock()
        if pipe === expectedPipe {
            pipe = nil
        }
        lock.unlock()
    }

    func send(_ data: Data) {
        lock.lock()
        let currentPipe = pipe
        lock.unlock()
        currentPipe?.send(data)
    }
}

/// Manages a streaming transcription lifecycle: buffers audio chunks, sends them to the provider, and collects the final text.
@MainActor
final class StreamingTranscriptionService: NSObject {

    typealias ProviderFactory = (any TranscriptionModel) throws -> any StreamingTranscriptionProvider

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "StreamingTranscriptionService")
    private let providerFactory: ProviderFactory
    private let audioRouter = StreamingAudioRouter()
    private let onPartialTranscript: (@Sendable (String) -> Void)?
    private let workspaceNotificationCenter: NotificationCenter

    private var activeSessionID: UUID?
    private var provider: (any StreamingTranscriptionProvider)?
    private var audioPipe: StreamingAudioPipe?
    private var sendTask: Task<Void, Error>?
    private var eventConsumerTask: Task<Void, Never>?
    private var state: StreamingState = .idle
    private var committedSegments: [String] = []
    private var streamingFailure: Error?

    init(
        onPartialTranscript: (@Sendable (String) -> Void)? = nil,
        providerFactory: ProviderFactory? = nil,
        workspaceNotificationCenter: NotificationCenter = NSWorkspace.shared.notificationCenter
    ) {
        self.onPartialTranscript = onPartialTranscript
        self.providerFactory = providerFactory ?? Self.createProvider
        self.workspaceNotificationCenter = workspaceNotificationCenter
        super.init()

        workspaceNotificationCenter.addObserver(
            self,
            selector: #selector(workspaceWillSleep(_:)),
            name: NSWorkspace.willSleepNotification,
            object: nil
        )
        workspaceNotificationCenter.addObserver(
            self,
            selector: #selector(workspaceDidWake(_:)),
            name: NSWorkspace.didWakeNotification,
            object: nil
        )
    }

    isolated deinit {
        workspaceNotificationCenter.removeObserver(self)
        sendTask?.cancel()
        eventConsumerTask?.cancel()
        if let audioPipe {
            audioRouter.remove(audioPipe)
            audioPipe.close()
        }
    }

    /// Whether the streaming connection is fully established and actively sending.
    var isActive: Bool { state == .streaming || state == .committing }

    /// Start a streaming transcription session for the given model.
    func startStreaming(model: any TranscriptionModel) async throws {
        guard activeSessionID == nil else {
            throw StreamingTranscriptionError.connectionFailed("A streaming session is already active")
        }

        let sessionID = UUID()
        let pipe = StreamingAudioPipe(logger: logger)
        let provider = try providerFactory(model)

        activeSessionID = sessionID
        audioPipe = pipe
        self.provider = provider
        streamingFailure = nil
        committedSegments = []
        state = .connecting
        audioRouter.install(pipe)

        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"

        do {
            try await provider.connect(model: model, language: selectedLanguage)
        } catch {
            if activeSessionID == sessionID {
                await cleanupStreaming(sessionID: sessionID, finalState: .failed)
            } else {
                await provider.disconnect()
            }
            throw error
        }

        // Cancellation or a replacement session may have happened while connect suspended.
        guard activeSessionID == sessionID else {
            await provider.disconnect()
            throw CancellationError()
        }

        state = .streaming
        startSendLoop(provider: provider, pipe: pipe, sessionID: sessionID)
        startEventConsumer(provider: provider, sessionID: sessionID)

        logger.notice("Streaming started for model: \(model.displayName, privacy: .public)")
    }

    /// Buffers an audio chunk for sending. Safe to call from the audio callback thread.
    nonisolated func sendAudioChunk(_ data: Data) {
        audioRouter.send(data)
    }

    /// Stops streaming, commits remaining audio, and returns the final transcribed text.
    func stopAndGetFinalText() async throws -> String {
        guard let sessionID = activeSessionID,
              let provider,
              let pipe = audioPipe else {
            throw StreamingTranscriptionError.notConnected
        }

        if let streamingFailure {
            await cleanupStreaming(sessionID: sessionID, finalState: .failed)
            throw streamingFailure
        }

        guard state == .streaming else {
            throw StreamingTranscriptionError.notConnected
        }
        state = .committing

        do {
            try await drainRemainingChunks(pipe: pipe)

            if pipe.buffer.hasTrimmedAudio {
                throw StreamingTranscriptionError.audioBufferOverflow
            }

            try await provider.commit()
            let finalText = try await waitForFinalCommit(sessionID: sessionID)
            state = .done
            await cleanupStreaming(sessionID: sessionID, finalState: .idle)
            return finalText
        } catch {
            logger.error("Streaming finalization failed: \(error.localizedDescription, privacy: .public)")
            await cleanupStreaming(sessionID: sessionID, finalState: .failed)
            throw error
        }
    }

    /// Cancels the streaming session without waiting for results.
    func cancel() {
        guard let sessionID = activeSessionID else {
            state = .idle
            return
        }

        let providerToDisconnect = provider
        let pipeToClose = audioPipe

        activeSessionID = nil
        provider = nil
        audioPipe = nil
        streamingFailure = nil
        committedSegments = []
        state = .idle

        sendTask?.cancel()
        sendTask = nil
        eventConsumerTask?.cancel()
        eventConsumerTask = nil

        if let pipeToClose {
            audioRouter.remove(pipeToClose)
            pipeToClose.close()
            pipeToClose.clear()
        }

        Task { [providerToDisconnect] in
            await providerToDisconnect?.disconnect()
        }

        logger.notice("Streaming session \(sessionID.uuidString, privacy: .private) cancelled")
    }

    // MARK: - Private

    @objc private func workspaceWillSleep(_ notification: Notification) {
        _ = notification
        guard activeSessionID != nil else { return }
        logger.notice("Cancelling streaming transcription before system sleep")
        cancel()
    }

    @objc private func workspaceDidWake(_ notification: Notification) {
        _ = notification
        logger.notice("System wake observed; the next dictation will create a fresh streaming session")
    }

    private static func createProvider(for model: any TranscriptionModel) throws -> any StreamingTranscriptionProvider {
        switch model.provider {
        case .localVoxtral:
            return VoxtralNativeStreamingProvider()
        case .elevenLabs:
            return ElevenLabsStreamingProvider()
        case .xAI:
            return XAIStreamingProvider()
        case .openAI:
            return OpenAIStreamingProvider()
        default:
            throw StreamingTranscriptionError.unsupportedProvider(String(describing: model.provider))
        }
    }

    private func startSendLoop(
        provider: any StreamingTranscriptionProvider,
        pipe: StreamingAudioPipe,
        sessionID: UUID
    ) {
        sendTask = Task { [weak self, provider, pipe] in
            do {
                for await _ in pipe.signals {
                    try Task.checkCancellation()
                    for chunk in pipe.buffer.drain() {
                        try await provider.sendAudioChunk(chunk)
                    }
                }

                for chunk in pipe.buffer.drain() {
                    try Task.checkCancellation()
                    try await provider.sendAudioChunk(chunk)
                }
            } catch {
                self?.recordStreamingFailure(error, sessionID: sessionID)
                throw error
            }
        }
    }

    /// Closes this session's source and waits for its sender to drain or fail.
    private func drainRemainingChunks(pipe: StreamingAudioPipe) async throws {
        audioRouter.remove(pipe)
        pipe.close()
        try await sendTask?.value
        sendTask = nil

        if let streamingFailure {
            throw streamingFailure
        }
    }

    /// Consumes transcription events throughout the session, accumulating committed segments.
    private func startEventConsumer(
        provider: any StreamingTranscriptionProvider,
        sessionID: UUID
    ) {
        let events = provider.transcriptionEvents

        eventConsumerTask = Task { [weak self, events] in
            for await event in events {
                guard let self, self.activeSessionID == sessionID else { break }

                switch event {
                case .committed(let text):
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        self.committedSegments.append(trimmed)
                    }
                case .partial(let text):
                    if self.state == .streaming {
                        self.onPartialTranscript?(text)
                    }
                case .sessionStarted:
                    break
                case .error(let error):
                    self.recordStreamingFailure(error, sessionID: sessionID)
                }
            }
        }
    }

    private func recordStreamingFailure(_ error: Error, sessionID: UUID) {
        guard activeSessionID == sessionID, streamingFailure == nil else { return }
        streamingFailure = error
        state = .failed
        logger.error("Streaming session failed: \(error.localizedDescription, privacy: .public)")
    }

    /// Wait for the first committed segment and then a bounded quiet period so
    /// trailing provider segments are not cut off. The overall deadline prevents
    /// a provider that never finishes from holding the recording indefinitely.
    private func waitForFinalCommit(sessionID: UUID) async throws -> String {
        let deadline = Date().addingTimeInterval(10)
        var observedSegmentCount = committedSegments.count
        var lastChange = Date()

        while Date() < deadline {
            try Task.checkCancellation()
            guard activeSessionID == sessionID else {
                throw CancellationError()
            }

            if let streamingFailure {
                throw streamingFailure
            }

            if committedSegments.count != observedSegmentCount {
                observedSegmentCount = committedSegments.count
                lastChange = Date()
            }

            if observedSegmentCount > 0, Date().timeIntervalSince(lastChange) >= 0.35 {
                break
            }

            try await Task.sleep(nanoseconds: 50_000_000)
        }

        if committedSegments.isEmpty {
            logger.warning("No transcript received from streaming")
        }

        return committedSegments.joined(separator: " ")
    }

    private func cleanupStreaming(sessionID: UUID, finalState: StreamingState) async {
        guard activeSessionID == sessionID else { return }

        let providerToDisconnect = provider
        let pipeToClose = audioPipe

        activeSessionID = nil
        provider = nil
        audioPipe = nil
        streamingFailure = nil
        committedSegments = []

        sendTask?.cancel()
        sendTask = nil
        eventConsumerTask?.cancel()
        eventConsumerTask = nil

        if let pipeToClose {
            audioRouter.remove(pipeToClose)
            pipeToClose.close()
            pipeToClose.clear()
        }

        await providerToDisconnect?.disconnect()
        state = finalState == .failed ? .idle : finalState
    }
}
