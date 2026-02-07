import Foundation
import os

/// Deepgram Nova-3 Real-Time streaming provider using WebSocket.
final class DeepgramStreamingProvider: StreamingTranscriptionProvider {

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "DeepgramStreaming")
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var keepaliveTask: Task<Void, Never>?

    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        keepaliveTask?.cancel()
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
        eventsContinuation?.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        guard let apiKey = APIKeyManager.shared.getAPIKey(forProvider: "Deepgram"), !apiKey.isEmpty else {
            throw StreamingTranscriptionError.missingAPIKey
        }

        // Build the WebSocket URL with query parameters
        var components = URLComponents(string: "wss://api.deepgram.com/v1/listen")!
        var queryItems: [URLQueryItem] = [
            URLQueryItem(name: "model", value: model.name),
            URLQueryItem(name: "encoding", value: "linear16"),
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "channels", value: "1"),
            URLQueryItem(name: "smart_format", value: "true"),
            URLQueryItem(name: "punctuate", value: "true"),
            URLQueryItem(name: "interim_results", value: "true")
        ]

        if let language = language, language != "auto", !language.isEmpty {
            queryItems.append(URLQueryItem(name: "language", value: language))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw StreamingTranscriptionError.connectionFailed("Invalid WebSocket URL")
        }

        var request = URLRequest(url: url)
        request.setValue("Token \(apiKey)", forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)

        self.urlSession = session
        self.webSocketTask = task

        task.resume()

        logger.notice("WebSocket connecting to \(url.absoluteString)")

        // Deepgram doesn't send a session_started message, connection is established when resume() succeeds
        eventsContinuation?.yield(.sessionStarted)
        logger.notice("Streaming session started")

        // Start the background receive loop
        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }

        // Start keepalive timer (send keepalive every 5 seconds)
        keepaliveTask = Task { [weak self] in
            await self?.keepaliveLoop()
        }
    }

    func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw StreamingTranscriptionError.notConnected
        }

        // Deepgram expects raw binary PCM data (NOT base64 encoded)
        try await task.send(.data(data))
    }

    func commit() async throws {
        guard let task = webSocketTask else {
            throw StreamingTranscriptionError.notConnected
        }

        // Send Finalize message to commit remaining audio
        let finalizeMessage: [String: Any] = ["type": "Finalize"]
        let jsonData = try JSONSerialization.data(withJSONObject: finalizeMessage)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        logger.notice("Sending finalize message")
        try await task.send(.string(jsonString))
    }

    func disconnect() async {
        // Stop keepalive first
        keepaliveTask?.cancel()
        keepaliveTask = nil

        // Send CloseStream message if still connected
        if let task = webSocketTask {
            do {
                let closeMessage: [String: Any] = ["type": "CloseStream"]
                let jsonData = try JSONSerialization.data(withJSONObject: closeMessage)
                let jsonString = String(data: jsonData, encoding: .utf8)!
                try await task.send(.string(jsonString))
            } catch {
                logger.warning("Failed to send CloseStream message: \(error.localizedDescription)")
            }
        }

        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
        logger.notice("WebSocket disconnected")
    }

    // MARK: - Private

    private func keepaliveLoop() async {
        while !Task.isCancelled {
            do {
                // Wait 5 seconds between keepalives
                try await Task.sleep(nanoseconds: 5_000_000_000)

                guard let task = webSocketTask, !Task.isCancelled else { break }

                let keepaliveMessage: [String: Any] = ["type": "KeepAlive"]
                let jsonData = try JSONSerialization.data(withJSONObject: keepaliveMessage)
                let jsonString = String(data: jsonData, encoding: .utf8)!

                try await task.send(.string(jsonString))
                logger.debug("Sent keepalive")
            } catch {
                if !Task.isCancelled {
                    logger.warning("Keepalive error: \(error.localizedDescription)")
                }
                break
            }
        }
    }

    private func receiveLoop() async {
        guard let task = webSocketTask else { return }

        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    handleMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleMessage(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                if !Task.isCancelled {
                    logger.error("WebSocket receive error: \(error.localizedDescription)")
                    eventsContinuation?.yield(.error(error))
                }
                break
            }
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            logger.warning("Received unparseable message")
            return
        }

        // Check for error messages
        if let errorMessage = json["error"] as? String {
            logger.error("Server error: \(errorMessage)")
            eventsContinuation?.yield(.error(StreamingTranscriptionError.serverError(errorMessage)))
            return
        }

        // Check for type field (control messages)
        if let type = json["type"] as? String {
            logger.debug("Received control message: \(type)")
            return
        }

        // Parse transcription results
        guard let channel = json["channel"] as? [String: Any],
              let alternatives = channel["alternatives"] as? [[String: Any]],
              let firstAlternative = alternatives.first,
              let transcript = firstAlternative["transcript"] as? String,
              !transcript.isEmpty else {
            return
        }

        // Check if this is a final result
        let isFinal = json["is_final"] as? Bool ?? false
        let speechFinal = json["speech_final"] as? Bool ?? false

        if isFinal || speechFinal {
            // Final transcript
            logger.notice("Final: \(transcript)")
            eventsContinuation?.yield(.committed(text: transcript))
        } else {
            // Partial transcript
            eventsContinuation?.yield(.partial(text: transcript))
        }
    }
}
