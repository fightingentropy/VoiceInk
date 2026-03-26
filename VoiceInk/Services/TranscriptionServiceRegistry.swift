import Foundation
import SwiftUI
import os

@MainActor
class TranscriptionServiceRegistry {
    private weak var modelProvider: (any LocalModelProvider)?
    private let modelsDirectory: URL
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionServiceRegistry")

    private(set) lazy var localTranscriptionService = LocalTranscriptionService(
        modelsDirectory: modelsDirectory,
        modelProvider: modelProvider
    )
    private(set) lazy var cloudTranscriptionService = CloudTranscriptionService()
    private(set) lazy var nativeAppleTranscriptionService = NativeAppleTranscriptionService()
    private(set) lazy var parakeetTranscriptionService = ParakeetTranscriptionService()
    private(set) lazy var localVoxtralTranscriptionService = LocalVoxtralTranscriptionService()
    private(set) lazy var cohereTranscribeTranscriptionService = CohereTranscribeTranscriptionService()

    init(modelProvider: any LocalModelProvider, modelsDirectory: URL) {
        self.modelProvider = modelProvider
        self.modelsDirectory = modelsDirectory
    }

    func service(for provider: ModelProvider) -> TranscriptionService {
        switch provider {
        case .local:
            return localTranscriptionService
        case .localVoxtral:
            return localVoxtralTranscriptionService
        case .cohereTranscribe:
            return cohereTranscribeTranscriptionService
        case .parakeet:
            return parakeetTranscriptionService
        case .nativeApple:
            return nativeAppleTranscriptionService
        default:
            return cloudTranscriptionService
        }
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        let effectiveModel = batchFallbackModel(for: model) ?? model
        let service = service(for: effectiveModel.provider)
        logger.debug("Transcribing with \(effectiveModel.displayName, privacy: .public) using \(String(describing: type(of: service)), privacy: .public)")
        let text = try await service.transcribe(audioURL: audioURL, model: effectiveModel)

        if effectiveModel.provider == .local
            || effectiveModel.provider == .parakeet
            || effectiveModel.provider == .cohereTranscribe {
            NotificationCenter.default.post(name: .localModelDidUse, object: nil)
        }

        return text
    }

    /// Creates a streaming or file-based session depending on the model's capabilities.
    func createSession(for model: any TranscriptionModel, onPartialTranscript: (@Sendable (String) -> Void)? = nil) -> TranscriptionSession {
        if supportsStreaming(model: model) {
            let streamingService = StreamingTranscriptionService(
                onPartialTranscript: onPartialTranscript
            )
            let fallback = service(for: model.provider)
            let fallbackModel = batchFallbackModel(for: model)
            return StreamingTranscriptionSession(streamingService: streamingService, fallbackService: fallback, fallbackModel: fallbackModel)
        } else {
            return FileTranscriptionSession(service: service(for: model.provider))
        }
    }

    // Maps streaming-only models to a batch-compatible equivalent for fallback.
    private func batchFallbackModel(for model: any TranscriptionModel) -> (any TranscriptionModel)? {
        switch (model.provider, model.name) {
        case (.localVoxtral, "voxtral-mini-realtime-local"):
            return nil
        default:
            return nil
        }
    }

    /// Whether the given model supports streaming transcription
    private func supportsStreaming(model: any TranscriptionModel) -> Bool {
        switch model.provider {
        case .localVoxtral:
            return model.name == "voxtral-mini-realtime-local"
        case .elevenLabs:
            return model.name == "scribe_v2"
        default:
            return false
        }
    }

    func cleanup() {
        parakeetTranscriptionService.cleanup()
    }
}
