import Foundation
import SwiftData
import LLMkit

/// Soniox streaming provider wrapping `LLMkit.SonioxStreamingClient`.
final class SonioxStreamingProvider: StreamingTranscriptionProvider, @unchecked Sendable {

    private let client = LLMkit.SonioxStreamingClient()
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var forwardingTask: Task<Void, Never>?
    private let modelContext: ModelContext

    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    init(modelContext: ModelContext) {
        self.modelContext = modelContext
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        forwardingTask?.cancel()
        eventsContinuation?.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        guard let apiKey = APIKeyManager.shared.getAPIKey(forProvider: "Soniox"), !apiKey.isEmpty else {
            throw StreamingTranscriptionError.missingAPIKey
        }

        let vocabulary = getCustomDictionaryTerms()

        // Cancel any existing forwarding task before starting a new one
        forwardingTask?.cancel()
        startEventForwarding()

        do {
            try await client.connect(apiKey: apiKey, model: model.name, language: language, customVocabulary: vocabulary)
        } catch {
            // Clean up forwarding task on connection failure
            forwardingTask?.cancel()
            forwardingTask = nil
            throw mapError(error)
        }
    }

    func sendAudioChunk(_ data: Data) async throws {
        do {
            try await client.sendAudioChunk(data)
        } catch {
            throw mapError(error)
        }
    }

    func commit() async throws {
        do {
            try await client.commit()
        } catch {
            throw mapError(error)
        }
    }

    func disconnect() async {
        forwardingTask?.cancel()
        forwardingTask = nil
        await client.disconnect()
        eventsContinuation?.finish()
    }

    // MARK: - Private

    private func startEventForwarding() {
        let client = client
        let eventsContinuation = eventsContinuation

        forwardingTask = Task {
            for await event in client.transcriptionEvents {
                switch event {
                case .sessionStarted:
                    eventsContinuation?.yield(.sessionStarted)
                case .partial(let text):
                    eventsContinuation?.yield(.partial(text: text))
                case .committed(let text):
                    eventsContinuation?.yield(.committed(text: text))
                case .error(let message):
                    eventsContinuation?.yield(.error(StreamingTranscriptionError.serverError(message)))
                }
            }
        }
    }

    private func getCustomDictionaryTerms() -> [String] {
        let descriptor = FetchDescriptor<VocabularyWord>(sortBy: [SortDescriptor(\.word)])
        guard let vocabularyWords = try? modelContext.fetch(descriptor) else {
            return []
        }
        var seen = Set<String>()
        var unique: [String] = []
        for word in vocabularyWords {
            let trimmed = word.word.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let key = trimmed.lowercased()
            if !seen.contains(key) {
                seen.insert(key)
                unique.append(trimmed)
            }
        }
        return unique
    }

    private func mapError(_ error: Error) -> Error {
        guard let llmError = error as? LLMKitError else { return error }
        switch llmError {
        case .missingAPIKey:
            return StreamingTranscriptionError.missingAPIKey
        case .httpError(_, let message):
            return StreamingTranscriptionError.serverError(message)
        case .networkError(let detail):
            return StreamingTranscriptionError.connectionFailed(detail)
        default:
            return StreamingTranscriptionError.serverError(llmError.localizedDescription ?? "Unknown error")
        }
    }
}
