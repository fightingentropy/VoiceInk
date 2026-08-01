import Foundation

actor OpenAIStreamingProvider: StreamingTranscriptionProvider {
    static let modelName = "gpt-live-transcribe"
    static let inputSampleRate = 24_000
    static let transcriptionDelay = "minimal"
    static let webSocketURL = URL(string: "wss://api.openai.com/v1/realtime?intent=transcription")!

    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var isConnected = false
    private nonisolated let eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation
    private var partialTranscripts: [String: String] = [:]
    private var resampler = OpenAIPCM16Resampler()
    private var isAwaitingFinalization = false

    nonisolated let finalizationMode: StreamingFinalizationMode = .providerSignal
    nonisolated let transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    init() {
        (transcriptionEvents, eventsContinuation) = AsyncStream.makeStream(
            of: StreamingTranscriptionEvent.self
        )
    }

    deinit {
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        eventsContinuation.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        guard model.name == Self.modelName else {
            throw StreamingTranscriptionError.unsupportedProvider(model.name)
        }
        guard let apiKey = APIKeyManager.shared.getAPIKey(forProvider: model.provider.apiKeyProviderName),
              !apiKey.isEmpty else {
            throw StreamingTranscriptionError.missingAPIKey
        }

        receiveTask?.cancel()
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        partialTranscripts.removeAll(keepingCapacity: true)
        resampler.reset()
        isAwaitingFinalization = false

        var request = URLRequest(url: Self.webSocketURL)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let task = URLSession.shared.webSocketTask(with: request)
        webSocketTask = task
        task.resume()

        do {
            try await waitForSessionReady(
                on: task,
                language: Self.normalizeLanguage(language),
                prompt: Self.transcriptionPrompt()
            )
            isConnected = true
            startReceiveLoop(on: task)
        } catch {
            task.cancel(with: .goingAway, reason: nil)
            webSocketTask = nil
            isConnected = false
            throw mapConnectionError(error)
        }
    }

    func sendAudioChunk(_ data: Data) async throws {
        guard isConnected, let webSocketTask else {
            throw StreamingTranscriptionError.notConnected
        }

        let resampledData = resampler.append(data)
        guard !resampledData.isEmpty else { return }
        try await sendAudio(resampledData, on: webSocketTask)
    }

    func commit() async throws {
        guard isConnected, let webSocketTask else {
            throw StreamingTranscriptionError.notConnected
        }

        let remainingAudio = resampler.flush()
        if !remainingAudio.isEmpty {
            try await sendAudio(remainingAudio, on: webSocketTask)
        }

        isAwaitingFinalization = true
        do {
            try await webSocketTask.send(.string(#"{"type":"input_audio_buffer.commit"}"#))
        } catch {
            isAwaitingFinalization = false
            throw error
        }
    }

    func disconnect() async {
        isConnected = false
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        partialTranscripts.removeAll(keepingCapacity: false)
        resampler.reset()
        isAwaitingFinalization = false
        eventsContinuation.finish()
    }

    static func makeSessionUpdate(language: String?, prompt: String?) throws -> String {
        var transcription: [String: Any] = [
            "model": modelName,
            "delay": transcriptionDelay
        ]

        if let language = normalizeLanguage(language) {
            transcription["languages"] = [language]
        }
        if let prompt, !prompt.isEmpty {
            transcription["prompt"] = prompt
        }

        let event: [String: Any] = [
            "type": "session.update",
            "session": [
                "type": "transcription",
                "audio": [
                    "input": [
                        "format": [
                            "type": "audio/pcm",
                            "rate": inputSampleRate
                        ],
                        "transcription": transcription,
                        "turn_detection": NSNull()
                    ]
                ]
            ]
        ]

        let data = try JSONSerialization.data(withJSONObject: event)
        guard let string = String(data: data, encoding: .utf8) else {
            throw StreamingTranscriptionError.connectionFailed("Could not encode OpenAI session configuration")
        }
        return string
    }

    static func makeAudioAppendMessage(_ audioData: Data) throws -> String {
        let event: [String: Any] = [
            "type": "input_audio_buffer.append",
            "audio": audioData.base64EncodedString()
        ]
        let data = try JSONSerialization.data(withJSONObject: event)
        guard let string = String(data: data, encoding: .utf8) else {
            throw StreamingTranscriptionError.connectionFailed("Could not encode OpenAI audio")
        }
        return string
    }

    // MARK: - Private

    private func sendAudio(_ data: Data, on task: URLSessionWebSocketTask) async throws {
        try await task.send(.string(try Self.makeAudioAppendMessage(data)))
    }

    private func waitForSessionReady(
        on task: URLSessionWebSocketTask,
        language: String?,
        prompt: String?
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { [weak self] in
                guard let self else {
                    throw StreamingTranscriptionError.connectionFailed("Streaming provider was released")
                }
                try await self.configureSession(
                    on: task,
                    language: language,
                    prompt: prompt
                )
            }

            group.addTask {
                try await Task.sleep(nanoseconds: 10_000_000_000)
                throw StreamingTranscriptionError.timeout
            }

            do {
                try await group.next()
                group.cancelAll()
            } catch {
                group.cancelAll()
                throw error
            }
        }
    }

    private func configureSession(
        on task: URLSessionWebSocketTask,
        language: String?,
        prompt: String?
    ) async throws {
        var didSendConfiguration = false

        while !Task.isCancelled {
            let event = try decodeEvent(from: try await task.receive())
            switch event.type {
            case "session.created":
                guard !didSendConfiguration else { continue }
                try await task.send(.string(try Self.makeSessionUpdate(language: language, prompt: prompt)))
                didSendConfiguration = true
            case "session.updated":
                eventsContinuation.yield(.sessionStarted)
                return
            case "error":
                throw StreamingTranscriptionError.serverError(event.error?.message ?? "OpenAI Realtime error")
            default:
                handleEvent(event)
            }
        }
    }

    private func startReceiveLoop(on task: URLSessionWebSocketTask) {
        receiveTask = Task { [weak self, weak task] in
            guard let self, let task else { return }
            await self.receiveEvents(on: task)
        }
    }

    private func receiveEvents(on task: URLSessionWebSocketTask) async {
        while !Task.isCancelled {
            do {
                let event = try decodeEvent(from: try await task.receive())
                handleEvent(event)
            } catch {
                if !Task.isCancelled {
                    eventsContinuation.yield(.error(mapConnectionError(error)))
                }
                break
            }
        }
    }

    private func handleEvent(_ event: EventPayload) {
        switch event.type {
        case "conversation.item.input_audio_transcription.delta":
            let itemID = event.itemID ?? "active"
            let accumulated = partialTranscripts[itemID, default: ""] + (event.delta ?? "")
            partialTranscripts[itemID] = accumulated
            if !accumulated.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                eventsContinuation.yield(.partial(text: accumulated))
            }
        case "conversation.item.input_audio_transcription.completed":
            let itemID = event.itemID ?? "active"
            partialTranscripts.removeValue(forKey: itemID)
            eventsContinuation.yield(.committed(text: event.transcript ?? ""))
            if isAwaitingFinalization {
                isAwaitingFinalization = false
                eventsContinuation.yield(.finalized)
            }
        case "error":
            eventsContinuation.yield(
                .error(StreamingTranscriptionError.serverError(event.error?.message ?? "OpenAI Realtime error"))
            )
        default:
            break
        }
    }

    private func decodeEvent(from message: URLSessionWebSocketTask.Message) throws -> EventPayload {
        let data: Data
        switch message {
        case .data(let payload):
            data = payload
        case .string(let string):
            guard let encoded = string.data(using: .utf8) else {
                throw StreamingTranscriptionError.connectionFailed("Invalid OpenAI text frame")
            }
            data = encoded
        @unknown default:
            throw StreamingTranscriptionError.connectionFailed("Unsupported OpenAI WebSocket frame")
        }

        do {
            return try JSONDecoder().decode(EventPayload.self, from: data)
        } catch {
            throw StreamingTranscriptionError.connectionFailed("Invalid OpenAI Realtime response")
        }
    }

    private func mapConnectionError(_ error: Error) -> Error {
        if let streamingError = error as? StreamingTranscriptionError {
            return streamingError
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            return StreamingTranscriptionError.connectionFailed(nsError.localizedDescription)
        }

        return error
    }

    private static func normalizeLanguage(_ language: String?) -> String? {
        guard let language,
              !language.isEmpty,
              language != "auto" else {
            return nil
        }
        return language.lowercased()
    }

    private static func transcriptionPrompt() -> String? {
        let prompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt") ?? ""
        return prompt.isEmpty ? nil : prompt
    }

    private struct EventPayload: Decodable {
        let type: String
        let delta: String?
        let transcript: String?
        let itemID: String?
        let error: ErrorPayload?

        private enum CodingKeys: String, CodingKey {
            case type
            case delta
            case transcript
            case itemID = "item_id"
            case error
        }
    }

    private struct ErrorPayload: Decodable {
        let message: String?
    }
}

struct OpenAIPCM16Resampler {
    private static let sourceRate: Int64 = 16_000
    private static let targetRate: Int64 = 24_000

    private var samples: [Int16] = []
    /// Source position expressed in `targetRate`-sized fixed-point units.
    /// Integer math keeps output identical regardless of recorder chunk boundaries.
    private var nextOutputPosition: Int64 = 0
    private var trailingByte: UInt8?

    mutating func append(_ data: Data) -> Data {
        appendInputSamples(from: data)
        return render(flush: false)
    }

    mutating func flush() -> Data {
        let output = render(flush: true)
        reset()
        return output
    }

    mutating func reset() {
        samples.removeAll(keepingCapacity: false)
        nextOutputPosition = 0
        trailingByte = nil
    }

    private mutating func appendInputSamples(from data: Data) {
        var bytes = [UInt8]()
        bytes.reserveCapacity(data.count + (trailingByte == nil ? 0 : 1))

        if let trailingByte {
            bytes.append(trailingByte)
            self.trailingByte = nil
        }
        bytes.append(contentsOf: data)

        if bytes.count % 2 != 0 {
            trailingByte = bytes.removeLast()
        }

        samples.reserveCapacity(samples.count + bytes.count / 2)
        for offset in stride(from: 0, to: bytes.count, by: 2) {
            let value = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
            samples.append(Int16(bitPattern: value))
        }
    }

    private mutating func render(flush: Bool) -> Data {
        guard !samples.isEmpty else { return Data() }

        var outputSamples: [Int16] = []
        let sampleLimit = Int64(samples.count) * Self.targetRate

        while flush
            ? nextOutputPosition < sampleLimit
            : nextOutputPosition + Self.targetRate < sampleLimit {
            let lowerIndex = min(Int(nextOutputPosition / Self.targetRate), samples.count - 1)
            let upperIndex = min(lowerIndex + 1, samples.count - 1)
            let fraction = Double(nextOutputPosition % Self.targetRate) / Double(Self.targetRate)
            let lower = Double(samples[lowerIndex])
            let upper = Double(samples[upperIndex])
            let interpolated = lower + ((upper - lower) * fraction)
            let clamped = max(Double(Int16.min), min(Double(Int16.max), interpolated.rounded()))
            outputSamples.append(Int16(clamped))
            nextOutputPosition += Self.sourceRate
        }

        if !flush {
            let discardCount = min(
                Int(nextOutputPosition / Self.targetRate),
                max(samples.count - 1, 0)
            )
            if discardCount > 0 {
                samples.removeFirst(discardCount)
                nextOutputPosition -= Int64(discardCount) * Self.targetRate
            }
        }

        var output = Data(capacity: outputSamples.count * MemoryLayout<Int16>.size)
        for sample in outputSamples {
            let value = UInt16(bitPattern: sample)
            output.append(UInt8(truncatingIfNeeded: value))
            output.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        return output
    }
}
