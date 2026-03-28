import Foundation
import SwiftUI
import os

@MainActor
class TranscriptionServiceRegistry {
    private weak var modelProvider: (any LocalModelProvider)?
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionServiceRegistry")

    private(set) lazy var localTranscriptionService = LocalTranscriptionService(
        modelProvider: modelProvider
    )
    private(set) lazy var cloudTranscriptionService = CloudTranscriptionService()
    private(set) lazy var nativeAppleTranscriptionService = NativeAppleTranscriptionService()
    private(set) lazy var parakeetTranscriptionService = ParakeetTranscriptionService()
    private(set) lazy var cohereTranscribeTranscriptionService = CohereTranscribeTranscriptionService()

    init(modelProvider: any LocalModelProvider) {
        self.modelProvider = modelProvider
    }

    private func fileService(for provider: ModelProvider) -> any TranscriptionService {
        switch provider {
        case .local:
            return localTranscriptionService
        case .parakeet:
            return parakeetTranscriptionService
        case .nativeApple:
            return nativeAppleTranscriptionService
        case .localVoxtral, .cohereTranscribe:
            preconditionFailure("Streaming-only or recorder-only models do not have file transcription services")
        default:
            return cloudTranscriptionService
        }
    }

    private func recorderService(for provider: ModelProvider) -> any RecorderTranscriptionService {
        switch provider {
        case .cohereTranscribe:
            return cohereTranscribeTranscriptionService
        case .local:
            return localTranscriptionService
        case .parakeet:
            return parakeetTranscriptionService
        case .nativeApple:
            return nativeAppleTranscriptionService
        case .localVoxtral:
            preconditionFailure("Streaming-only models do not create recorder sessions")
        default:
            return cloudTranscriptionService
        }
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard model.supportsAudioFileTranscription else {
            throw TranscriptionCapabilityError.audioFileInputUnsupported(modelName: model.displayName)
        }
        let service = fileService(for: model.provider)
        logger.debug("Transcribing with \(model.displayName, privacy: .public) using \(String(describing: type(of: service)), privacy: .public)")
        let text = try await service.transcribe(audioURL: audioURL, model: model)

        if model.provider == .local || model.provider == .parakeet {
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
            let fallbackService = model.supportsAudioFileTranscription ? fileService(for: model.provider) : nil
            return StreamingTranscriptionSession(
                streamingService: streamingService,
                fallbackService: fallbackService
            )
        } else {
            return FileTranscriptionSession(service: recorderService(for: model.provider))
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
        Task {
            await cohereTranscribeTranscriptionService.cleanup()
        }
    }
}
