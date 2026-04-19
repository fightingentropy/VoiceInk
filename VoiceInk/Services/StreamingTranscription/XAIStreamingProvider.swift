import Foundation

final class XAIStreamingProvider: StreamingTranscriptionProvider, @unchecked Sendable {
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var isConnected = false
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?

    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        eventsContinuation?.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        guard let apiKey = APIKeyManager.shared.getAPIKey(forProvider: model.provider.apiKeyProviderName), !apiKey.isEmpty else {
            throw StreamingTranscriptionError.missingAPIKey
        }

        receiveTask?.cancel()
        webSocketTask?.cancel(with: .goingAway, reason: nil)

        var request = URLRequest(url: try makeStreamingURL(language: language))
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let task = URLSession.shared.webSocketTask(with: request)
        webSocketTask = task
        task.resume()

        do {
            try await waitForSessionStarted(on: task)
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

        try await webSocketTask.send(.data(data))
    }

    func commit() async throws {
        guard isConnected, let webSocketTask else {
            throw StreamingTranscriptionError.notConnected
        }

        try await webSocketTask.send(.string(#"{"type":"audio.done"}"#))
    }

    func disconnect() async {
        isConnected = false
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        eventsContinuation?.finish()
    }

    // MARK: - Private

    private func makeStreamingURL(language: String?) throws -> URL {
        var components = URLComponents()
        components.scheme = "wss"
        components.host = "api.x.ai"
        components.path = "/v1/stt"

        var queryItems = [
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "encoding", value: "pcm"),
            URLQueryItem(name: "interim_results", value: "true")
        ]

        if let language = normalizeLanguage(language) {
            queryItems.append(URLQueryItem(name: "language", value: language))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw StreamingTranscriptionError.connectionFailed("Invalid xAI streaming URL")
        }

        return url
    }

    private func normalizeLanguage(_ language: String?) -> String? {
        guard let language,
              !language.isEmpty,
              language != "auto" else {
            return nil
        }

        return language.split(separator: "-").first.map(String.init)
    }

    private func waitForSessionStarted(on task: URLSessionWebSocketTask) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { [weak self] in
                guard let self else {
                    throw StreamingTranscriptionError.connectionFailed("Streaming provider was released")
                }
                try await self.receiveUntilSessionStarted(from: task)
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

    private func receiveUntilSessionStarted(from task: URLSessionWebSocketTask) async throws {
        while !Task.isCancelled {
            let event = try decodeEvent(from: try await task.receive())
            switch event.type {
            case "transcript.created":
                eventsContinuation?.yield(.sessionStarted)
                return
            case "error":
                throw StreamingTranscriptionError.serverError(event.message ?? "xAI streaming error")
            default:
                handleEvent(event)
            }
        }
    }

    private func startReceiveLoop(on task: URLSessionWebSocketTask) {
        receiveTask = Task { [weak self, weak task] in
            guard let self, let task else { return }

            while !Task.isCancelled {
                do {
                    let event = try self.decodeEvent(from: try await task.receive())
                    self.handleEvent(event)
                } catch {
                    if !Task.isCancelled {
                        self.eventsContinuation?.yield(.error(self.mapConnectionError(error)))
                    }
                    break
                }
            }
        }
    }

    private func handleEvent(_ event: EventPayload) {
        switch event.type {
        case "transcript.created":
            eventsContinuation?.yield(.sessionStarted)
        case "transcript.partial":
            let text = event.text ?? ""
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

            if event.isFinal == true && event.speechFinal == true {
                eventsContinuation?.yield(.committed(text: text))
            } else {
                eventsContinuation?.yield(.partial(text: text))
            }
        case "transcript.done":
            eventsContinuation?.yield(.committed(text: event.text ?? ""))
        case "error":
            eventsContinuation?.yield(.error(StreamingTranscriptionError.serverError(event.message ?? "xAI streaming error")))
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
                throw StreamingTranscriptionError.connectionFailed("Invalid xAI text frame")
            }
            data = encoded
        @unknown default:
            throw StreamingTranscriptionError.connectionFailed("Unsupported xAI WebSocket frame")
        }

        do {
            return try JSONDecoder().decode(EventPayload.self, from: data)
        } catch {
            throw StreamingTranscriptionError.connectionFailed("Invalid xAI streaming response")
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

    private struct EventPayload: Decodable {
        let type: String
        let text: String?
        let message: String?
        let isFinal: Bool?
        let speechFinal: Bool?

        private enum CodingKeys: String, CodingKey {
            case type
            case text
            case message
            case isFinal = "is_final"
            case speechFinal = "speech_final"
        }
    }
}
