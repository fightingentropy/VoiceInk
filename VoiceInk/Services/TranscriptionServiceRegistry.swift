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

    private func fileService(for model: any TranscriptionModel) throws -> any TranscriptionService {
        switch model.provider {
        case .local:
            return localTranscriptionService
        case .parakeet:
            return parakeetTranscriptionService
        case .nativeApple:
            return nativeAppleTranscriptionService
        case .localVoxtral, .cohereTranscribe:
            throw TranscriptionCapabilityError.audioFileInputUnsupported(modelName: model.displayName)
        default:
            return cloudTranscriptionService
        }
    }

    private func recorderService(for model: any TranscriptionModel) throws -> any RecorderTranscriptionService {
        switch model.provider {
        case .cohereTranscribe:
            return cohereTranscribeTranscriptionService
        case .local:
            return localTranscriptionService
        case .parakeet:
            return parakeetTranscriptionService
        case .nativeApple:
            return nativeAppleTranscriptionService
        case .localVoxtral:
            throw TranscriptionCapabilityError.recorderUnsupported(modelName: model.displayName)
        default:
            return cloudTranscriptionService
        }
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard model.supportsAudioFileTranscription else {
            throw TranscriptionCapabilityError.audioFileInputUnsupported(modelName: model.displayName)
        }
        let service = try fileService(for: model)
        logger.debug("Transcribing with \(model.displayName, privacy: .public) using \(String(describing: type(of: service)), privacy: .public)")
        let text = try await service.transcribe(audioURL: audioURL, model: model)

        if model.provider == .local || model.provider == .parakeet {
            NotificationCenter.default.post(name: .localModelDidUse, object: nil)
        }

        return text
    }

    /// Creates a streaming or file-based session depending on the model's capabilities.
    func createSession(for model: any TranscriptionModel, onPartialTranscript: (@Sendable (String) -> Void)? = nil) throws -> TranscriptionSession {
        if supportsStreaming(model: model) {
            let streamingService = StreamingTranscriptionService(
                onPartialTranscript: onPartialTranscript
            )
            let fallbackService = model.supportsAudioFileTranscription ? try fileService(for: model) : nil
            return StreamingTranscriptionSession(
                streamingService: streamingService,
                fallbackService: fallbackService
            )
        } else {
            return FileTranscriptionSession(service: try recorderService(for: model))
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
        // Both services are now actor-isolated, so fire their cleanups as
        // detached tasks. The registry's own callers don't await the result.
        let parakeet = parakeetTranscriptionService
        let cohere = cohereTranscribeTranscriptionService
        Task {
            await parakeet.cleanup()
            await cohere.cleanup()
        }
    }
}
