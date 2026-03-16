import Foundation
import os

/// Streaming provider for a local voxmlx server speaking an OpenAI-Realtime-style protocol.
final class LocalVoxtralStreamingProvider: NSObject, StreamingTranscriptionProvider, @unchecked Sendable {

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "LocalVoxtralStreaming")
    private let stateQueue = DispatchQueue(label: "com.fightingentropy.voiceink.localVoxtral.state")

    private var urlSession: URLSession?
    private var webSocketTask: URLSessionWebSocketTask?
    private var connectContinuation: CheckedContinuation<Void, Error>?
    private var didReceiveSessionCreated = false
    private var didEmitSessionStarted = false
    private var pendingModelName = ""
    private var isDisconnecting = false

    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    override init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
        super.init()
    }

    deinit {
        stateQueue.sync {
            connectContinuation?.resume(throwing: StreamingTranscriptionError.notConnected)
            connectContinuation = nil
        }
        urlSession?.invalidateAndCancel()
        eventsContinuation?.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        _ = language

        try await LocalVoxtralServerManager.shared.prepareServerIfNeeded()

        guard let endpoint = LocalVoxtralConfiguration.endpointURL else {
            throw StreamingTranscriptionError.connectionFailed("Invalid Local Voxtral endpoint.")
        }

        guard let scheme = endpoint.scheme?.lowercased(), scheme == "ws" || scheme == "wss" else {
            throw StreamingTranscriptionError.connectionFailed("Local Voxtral endpoint must use ws:// or wss://.")
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            stateQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: StreamingTranscriptionError.notConnected)
                    return
                }

                self.cleanupLocked(cancelTask: true)

                let configuration = URLSessionConfiguration.default
                configuration.waitsForConnectivity = true
                configuration.timeoutIntervalForRequest = 15
                configuration.timeoutIntervalForResource = 15

                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
                let task = session.webSocketTask(with: endpoint)

                self.urlSession = session
                self.webSocketTask = task
                self.connectContinuation = continuation
                self.didReceiveSessionCreated = false
                self.didEmitSessionStarted = false
                self.pendingModelName = model.name == "voxtral-mini-realtime-local" ? LocalVoxtralConfiguration.modelName : model.name
                self.isDisconnecting = false

                task.resume()
                self.listenForMessages(on: task)

                self.scheduleConnectTimeout(for: task)
            }
        }
    }

    func sendAudioChunk(_ data: Data) async throws {
        let payload: [String: Any] = [
            "type": "input_audio_buffer.append",
            "audio": data.base64EncodedString()
        ]
        try await send(payload)
    }

    func commit() async throws {
        try await send([
            "type": "input_audio_buffer.commit",
            "final": true
        ])
    }

    func disconnect() async {
        let taskToCancel: URLSessionWebSocketTask? = stateQueue.sync {
            isDisconnecting = true
            let task = webSocketTask
            cleanupLocked(cancelTask: false)
            return task
        }

        taskToCancel?.cancel(with: .normalClosure, reason: nil)
        eventsContinuation?.finish()
    }

    // MARK: - Private

    private func scheduleConnectTimeout(for task: URLSessionWebSocketTask) {
        stateQueue.asyncAfter(deadline: .now() + 10) { [weak self] in
            guard let self else { return }
            guard self.webSocketTask === task, let continuation = self.connectContinuation else { return }

            self.connectContinuation = nil
            self.cleanupLocked(cancelTask: true)
            continuation.resume(
                throwing: StreamingTranscriptionError.connectionFailed("Timed out waiting for Local Voxtral session.")
            )
        }
    }

    private func send(_ payload: [String: Any]) async throws {
        let text = try serialize(payload)
        let task = try stateQueue.sync {
            guard let task = webSocketTask else {
                throw StreamingTranscriptionError.notConnected
            }
            return task
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            task.send(.string(text)) { error in
                if let error {
                    continuation.resume(throwing: StreamingTranscriptionError.connectionFailed(error.localizedDescription))
                } else {
                    continuation.resume()
                }
            }
        }
    }

    private func serialize(_ payload: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: payload)
        guard let text = String(data: data, encoding: .utf8) else {
            throw StreamingTranscriptionError.serverError("Failed to encode Local Voxtral payload.")
        }
        return text
    }

    private func listenForMessages(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self else { return }

            let shouldContinue = self.stateQueue.sync { self.webSocketTask === task }
            guard shouldContinue else { return }

            switch result {
            case .success(let message):
                self.handle(message)
                self.listenForMessages(on: task)
            case .failure(let error):
                self.handleSocketFailure(error.localizedDescription, task: task)
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            handleText(text)
        case .data(let data):
            guard let text = String(data: data, encoding: .utf8) else { return }
            handleText(text)
        @unknown default:
            break
        }
    }

    private func handleText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            logger.error("Failed to parse Local Voxtral message: \(text, privacy: .public)")
            return
        }

        let type = json["type"] as? String ?? ""

        switch type {
        case "session.created":
            handleSessionCreated()
        case "session.updated":
            completeConnectIfNeeded()
        case "response.audio_transcript.delta",
             "transcription.delta",
             "conversation.item.input_audio_transcription.delta":
            if let delta = findFirstString(in: json, keys: ["delta", "text", "transcript"]) {
                eventsContinuation?.yield(.partial(text: delta))
            }
        case "response.audio_transcript.done",
             "transcription.done",
             "conversation.item.input_audio_transcription.completed":
            if let text = findFirstString(in: json, keys: ["text", "transcript", "delta"]) {
                eventsContinuation?.yield(.committed(text: text))
            } else {
                eventsContinuation?.yield(.committed(text: ""))
            }
        case "error":
            let message = findFirstString(in: json, keys: ["message", "error", "detail"]) ?? "Unknown Local Voxtral error."
            failConnectIfNeeded(message)
            eventsContinuation?.yield(.error(StreamingTranscriptionError.serverError(message)))
        default:
            break
        }
    }

    private func handleSessionCreated() {
        let modelName = stateQueue.sync { () -> String in
            didReceiveSessionCreated = true
            return pendingModelName
        }

        if !modelName.isEmpty {
            Task {
                do {
                    try await send([
                        "type": "session.update",
                        "model": modelName
                    ])
                } catch {
                    failConnectIfNeeded(error.localizedDescription)
                    eventsContinuation?.yield(.error(error))
                }
            }
        } else {
            completeConnectIfNeeded()
        }
    }

    private func completeConnectIfNeeded() {
        let continuation: CheckedContinuation<Void, Error>? = stateQueue.sync {
            let continuation = connectContinuation
            connectContinuation = nil
            return continuation
        }

        guard let continuation else { return }
        continuation.resume()

        let shouldEmitSessionStarted = stateQueue.sync { () -> Bool in
            guard !didEmitSessionStarted else { return false }
            didEmitSessionStarted = true
            return true
        }
        if shouldEmitSessionStarted {
            eventsContinuation?.yield(.sessionStarted)
        }
    }

    private func failConnectIfNeeded(_ message: String) {
        let continuation: CheckedContinuation<Void, Error>? = stateQueue.sync {
            let continuation = connectContinuation
            connectContinuation = nil
            return continuation
        }

        continuation?.resume(throwing: StreamingTranscriptionError.connectionFailed(message))
    }

    private func handleSocketFailure(_ message: String, task: URLSessionWebSocketTask) {
        let shouldHandle = stateQueue.sync { webSocketTask === task }
        guard shouldHandle else { return }

        failConnectIfNeeded(message)
        eventsContinuation?.yield(.error(StreamingTranscriptionError.connectionFailed(message)))

        let shouldEmitDisconnected = stateQueue.sync { !isDisconnecting }
        stateQueue.async { [weak self] in
            self?.cleanupLocked(cancelTask: false)
        }

        if shouldEmitDisconnected {
            logger.error("Local Voxtral socket failed: \(message, privacy: .public)")
        }
    }

    private func findFirstString(in json: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = json[key] as? String, !value.isEmpty {
                return value
            }
        }
        return nil
    }

    private func cleanupLocked(cancelTask: Bool) {
        if cancelTask {
            webSocketTask?.cancel(with: .normalClosure, reason: nil)
        }
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        didReceiveSessionCreated = false
        didEmitSessionStarted = false
        pendingModelName = ""
    }
}

extension LocalVoxtralStreamingProvider: URLSessionWebSocketDelegate, URLSessionTaskDelegate {
    func urlSession(
        _: URLSession,
        webSocketTask _: URLSessionWebSocketTask,
        didOpenWithProtocol _: String?
    ) {
        logger.notice("Connected to Local Voxtral realtime endpoint")
    }

    func urlSession(
        _: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith _: URLSessionWebSocketTask.CloseCode,
        reason _: Data?
    ) {
        handleSocketFailure("Local Voxtral connection closed.", task: webSocketTask)
    }

    func urlSession(_: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let webSocketTask = task as? URLSessionWebSocketTask, let error else { return }
        handleSocketFailure(error.localizedDescription, task: webSocketTask)
    }
}
