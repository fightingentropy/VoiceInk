import AppKit
import Foundation
import Testing
@testable import VoiceInk

@MainActor
@Suite(.serialized)
struct StreamingTranscriptionServiceTests {
    private let model = CloudModel(
        name: "streaming-test",
        displayName: "Streaming Test",
        description: "Test model",
        provider: .openAI,
        speed: 1,
        accuracy: 1,
        isMultilingual: true,
        supportedLanguages: ["en": "English"]
    )

    @Test
    func startStopCanBeRepeatedOnTheSameService() async throws {
        let first = FakeStreamingProvider(committedSegments: ["first"])
        let second = FakeStreamingProvider(committedSegments: ["second"])
        let providers = ProviderQueue([first, second])
        let service = StreamingTranscriptionService(providerFactory: { _ in
            try providers.next()
        })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))
        #expect(try await service.stopAndGetFinalText() == "first")

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x03, 0x04]))
        #expect(try await service.stopAndGetFinalText() == "second")

        let firstChunks = await first.sentChunks
        let secondChunks = await second.sentChunks
        let firstDisconnects = await first.disconnectCount
        let secondDisconnects = await second.disconnectCount
        #expect(firstChunks == [Data([0x01, 0x02])])
        #expect(secondChunks == [Data([0x03, 0x04])])
        #expect(firstDisconnects == 1)
        #expect(secondDisconnects == 1)
    }

    @Test
    func cancelThenStartUsesFreshSessionResources() async throws {
        let cancelled = FakeStreamingProvider(committedSegments: [])
        let restarted = FakeStreamingProvider(committedSegments: ["restarted"])
        let providers = ProviderQueue([cancelled, restarted])
        let service = StreamingTranscriptionService(providerFactory: { _ in
            try providers.next()
        })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))
        service.cancel()

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x05, 0x06]))
        #expect(try await service.stopAndGetFinalText() == "restarted")
        let restartedChunks = await restarted.sentChunks
        #expect(restartedChunks == [Data([0x05, 0x06])])
    }

    @Test
    func waitsForTrailingCommittedSegments() async throws {
        let provider = FakeStreamingProvider(
            committedSegments: ["hello", "world"],
            delayBetweenSegmentsNanoseconds: 100_000_000
        )
        let service = StreamingTranscriptionService(providerFactory: { _ in provider })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))

        #expect(try await service.stopAndGetFinalText() == "hello world")
    }

    @Test
    func providerFinalizationSignalSkipsTrailingQuietPeriod() async throws {
        let provider = FakeStreamingProvider(
            committedSegments: ["ready"],
            finalizationMode: .providerSignal
        )
        let service = StreamingTranscriptionService(providerFactory: { _ in provider })

        try await service.startStreaming(model: model)
        let start = ContinuousClock.now
        let text = try await service.stopAndGetFinalText()
        let elapsed = ContinuousClock.now - start

        #expect(text == "ready")
        #expect(elapsed < .milliseconds(150))
    }

    @Test
    func providerFinalizationSignalWaitsForDefinitiveCompletion() async throws {
        let provider = FakeStreamingProvider(
            committedSegments: ["ready"],
            finalizationMode: .providerSignal,
            finalizationSignalDelayNanoseconds: 150_000_000
        )
        let service = StreamingTranscriptionService(providerFactory: { _ in provider })

        try await service.startStreaming(model: model)
        let start = ContinuousClock.now
        let text = try await service.stopAndGetFinalText()
        let elapsed = ContinuousClock.now - start

        #expect(text == "ready")
        #expect(elapsed >= .milliseconds(100))
        #expect(elapsed < .milliseconds(300))
    }

    @Test
    func sendFailureIsObservableAndDoesNotPoisonTheNextSession() async throws {
        let failed = FakeStreamingProvider(
            committedSegments: [],
            sendError: StreamingTranscriptionError.serverError("send failed")
        )
        let recovered = FakeStreamingProvider(committedSegments: ["recovered"])
        let providers = ProviderQueue([failed, recovered])
        let service = StreamingTranscriptionService(providerFactory: { _ in
            try providers.next()
        })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))

        do {
            _ = try await service.stopAndGetFinalText()
            Issue.record("Expected the send failure to be returned")
        } catch {
            #expect(error.localizedDescription.contains("send failed"))
        }

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x03, 0x04]))
        #expect(try await service.stopAndGetFinalText() == "recovered")
    }

    @Test
    func audioOverflowFailsInsteadOfReturningAnIncompleteTranscript() async throws {
        let provider = FakeStreamingProvider(committedSegments: ["incomplete"])
        let service = StreamingTranscriptionService(providerFactory: { _ in provider })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data(repeating: 0x01, count: BoundedPCMChunkBuffer.defaultCapacityBytes + 2))

        do {
            _ = try await service.stopAndGetFinalText()
            Issue.record("Expected an audio-buffer overflow error")
        } catch {
            #expect(error is StreamingTranscriptionError)
            #expect(error.localizedDescription.contains("dropping buffered samples"))
        }
    }

    @Test
    func providerDisconnectFailsTheSessionAndAProviderRestartRecovers() async throws {
        let disconnected = FakeStreamingProvider(committedSegments: [])
        let recovered = FakeStreamingProvider(committedSegments: ["reconnected"])
        let providers = ProviderQueue([disconnected, recovered])
        let service = StreamingTranscriptionService(providerFactory: { _ in
            try providers.next()
        })

        try await service.startStreaming(model: model)
        await disconnected.failConnection("provider disconnected")
        try await waitUntil { !service.isActive }

        do {
            _ = try await service.stopAndGetFinalText()
            Issue.record("Expected the provider disconnect to fail the active session")
        } catch {
            #expect(error.localizedDescription.contains("provider disconnected"))
        }

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x07, 0x08]))
        #expect(try await service.stopAndGetFinalText() == "reconnected")
    }

    @Test
    func providerErrorWithQueuedChunksIsObservable() async throws {
        let provider = FakeStreamingProvider(
            committedSegments: [],
            sendDelayNanoseconds: 500_000_000
        )
        let service = StreamingTranscriptionService(providerFactory: { _ in provider })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))
        service.sendAudioChunk(Data([0x03, 0x04]))
        service.sendAudioChunk(Data([0x05, 0x06]))
        try await waitUntil { await provider.sendAttemptCount == 1 }

        await provider.failConnection("connection dropped with queued audio")
        try await waitUntil { !service.isActive }

        do {
            _ = try await service.stopAndGetFinalText()
            Issue.record("Expected the provider event error to be returned")
        } catch {
            #expect(error.localizedDescription.contains("queued audio"))
        }

        let sentChunks = await provider.sentChunks
        #expect(sentChunks.count < 3)
    }

    @Test
    func systemSleepCancelsRecordingAndWakeAllowsAFreshSession() async throws {
        let notificationCenter = NotificationCenter()
        let sleeping = FakeStreamingProvider(committedSegments: [])
        let awakened = FakeStreamingProvider(committedSegments: ["awake"])
        let providers = ProviderQueue([sleeping, awakened])
        let service = StreamingTranscriptionService(
            providerFactory: { _ in try providers.next() },
            workspaceNotificationCenter: notificationCenter
        )

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))
        notificationCenter.post(name: NSWorkspace.willSleepNotification, object: nil)
        #expect(!service.isActive)

        notificationCenter.post(name: NSWorkspace.didWakeNotification, object: nil)
        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x09, 0x0a]))
        #expect(try await service.stopAndGetFinalText() == "awake")
    }

    @Test
    func cancellationDuringFinalFlushStopsPromptlyAndDoesNotPoisonRestart() async throws {
        let flushing = FakeStreamingProvider(committedSegments: ["pending"])
        let restarted = FakeStreamingProvider(committedSegments: ["restarted"])
        let providers = ProviderQueue([flushing, restarted])
        let service = StreamingTranscriptionService(providerFactory: { _ in
            try providers.next()
        })

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x01, 0x02]))
        let finalization = Task { try await service.stopAndGetFinalText() }
        try await waitUntil { await flushing.commitCount == 1 }

        service.cancel()

        do {
            _ = try await finalization.value
            Issue.record("Expected cancellation during the final quiet period")
        } catch {
            #expect(error is CancellationError)
        }

        try await service.startStreaming(model: model)
        service.sendAudioChunk(Data([0x0b, 0x0c]))
        #expect(try await service.stopAndGetFinalText() == "restarted")
    }

    private func waitUntil(
        timeoutNanoseconds: UInt64 = 1_000_000_000,
        _ condition: @escaping @MainActor () async -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + .nanoseconds(Int64(timeoutNanoseconds))
        while ContinuousClock.now < deadline {
            if await condition() {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Timed out waiting for asynchronous lifecycle state")
    }
}

@MainActor
private final class ProviderQueue {
    private var providers: [FakeStreamingProvider]

    init(_ providers: [FakeStreamingProvider]) {
        self.providers = providers
    }

    func next() throws -> FakeStreamingProvider {
        guard !providers.isEmpty else {
            throw StreamingTranscriptionError.connectionFailed("No test provider available")
        }
        return providers.removeFirst()
    }
}

private actor FakeStreamingProvider: StreamingTranscriptionProvider {
    private nonisolated let continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation
    private let committedSegments: [String]
    private let delayBetweenSegmentsNanoseconds: UInt64
    private let sendError: StreamingTranscriptionError?
    private let sendDelayNanoseconds: UInt64
    private let finalizationSignalDelayNanoseconds: UInt64
    private var chunks: [Data] = []
    private var disconnects = 0
    private var sendAttempts = 0
    private var commits = 0

    nonisolated let finalizationMode: StreamingFinalizationMode
    nonisolated let transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    init(
        committedSegments: [String],
        delayBetweenSegmentsNanoseconds: UInt64 = 0,
        sendError: StreamingTranscriptionError? = nil,
        sendDelayNanoseconds: UInt64 = 0,
        finalizationMode: StreamingFinalizationMode = .trailingQuietPeriod,
        finalizationSignalDelayNanoseconds: UInt64 = 0
    ) {
        (transcriptionEvents, continuation) = AsyncStream.makeStream(
            of: StreamingTranscriptionEvent.self
        )
        self.committedSegments = committedSegments
        self.delayBetweenSegmentsNanoseconds = delayBetweenSegmentsNanoseconds
        self.sendError = sendError
        self.sendDelayNanoseconds = sendDelayNanoseconds
        self.finalizationMode = finalizationMode
        self.finalizationSignalDelayNanoseconds = finalizationSignalDelayNanoseconds
    }

    var sentChunks: [Data] {
        chunks
    }

    var disconnectCount: Int {
        disconnects
    }

    var sendAttemptCount: Int {
        sendAttempts
    }

    var commitCount: Int {
        commits
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        continuation.yield(.sessionStarted)
    }

    func sendAudioChunk(_ data: Data) async throws {
        sendAttempts += 1
        if sendDelayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: sendDelayNanoseconds)
        }
        if let sendError {
            throw sendError
        }
        chunks.append(data)
    }

    func commit() async throws {
        commits += 1
        for (index, segment) in committedSegments.enumerated() {
            if index > 0, delayBetweenSegmentsNanoseconds > 0 {
                try await Task.sleep(nanoseconds: delayBetweenSegmentsNanoseconds)
            }
            continuation.yield(.committed(text: segment))
        }

        if finalizationMode == .providerSignal {
            let continuation = continuation
            let delay = finalizationSignalDelayNanoseconds
            if delay == 0 {
                continuation.yield(.finalized)
            } else {
                Task {
                    try? await Task.sleep(nanoseconds: delay)
                    continuation.yield(.finalized)
                }
            }
        }
    }

    func disconnect() async {
        disconnects += 1
        continuation.finish()
    }

    func failConnection(_ message: String) {
        continuation.yield(.error(StreamingTranscriptionError.connectionFailed(message)))
        continuation.finish()
    }
}
