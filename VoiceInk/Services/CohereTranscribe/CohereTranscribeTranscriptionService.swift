import Foundation
import os

enum CohereTranscribeServiceError: LocalizedError {
    case unsupportedModel

    var errorDescription: String? {
        switch self {
        case .unsupportedModel:
            return "Unsupported Cohere Transcribe model."
        }
    }
}

final class CohereTranscribeTranscriptionService: PCMBufferTranscriptionService, @unchecked Sendable {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "CohereNativeMLX")

    func prepareModel(for model: LocalCohereTranscribeModel) async throws {
        let language = selectedLanguage(for: model)
        _ = try await CohereNativeRuntime.shared.warmupModel(language: language)
    }

    func warmup(for model: LocalCohereTranscribeModel) async throws {
        let language = selectedLanguage(for: model)
        _ = try await CohereNativeRuntime.shared.warmupModel(language: language)
    }

    func cleanup() async {
        await CohereNativeRuntime.shared.clearPreparedBootstraps()
    }

    func transcribe(recordedPCMBuffer: Data, sampleRate: Int, model: any TranscriptionModel) async throws -> String {
        guard let cohereModel = model as? LocalCohereTranscribeModel else {
            throw CohereTranscribeServiceError.unsupportedModel
        }

        let audioSamples = try CohereNativeFeatureExtractor.decodePCM16Mono(recordedPCMBuffer)
        let language = selectedLanguage(for: cohereModel)
        let result = try await CohereNativeRuntime.shared.transcribe(
            audioSamples: audioSamples,
            sampleRate: sampleRate,
            language: language
        )
        return logAndReturn(result)
    }

    private func selectedLanguage(for model: LocalCohereTranscribeModel) -> String {
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage")
        guard let selectedLanguage,
              model.supportedLanguages[selectedLanguage] != nil else {
            return LocalCohereTranscribeConfiguration.fallbackLanguage
        }

        return selectedLanguage
    }

    private func logAndReturn(_ result: CohereNativeGenerationResult) -> String {
        logger.notice(
            "Cohere MLX transcription completed with stop reason \(String(describing: result.stopReason), privacy: .public) and \(result.generatedTokenIDs.count, privacy: .public) generated tokens"
        )
        return result.text
    }
}
