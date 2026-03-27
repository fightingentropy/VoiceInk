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

final class CohereTranscribeTranscriptionService: TranscriptionService, @unchecked Sendable {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "CohereNativeMLX")

    func prepareModel(for model: LocalCohereTranscribeModel) async throws {
        let language = selectedLanguage(for: model)
        _ = try await CohereNativeRuntime.shared.warmupModel(language: language)
    }

    func warmup(for model: LocalCohereTranscribeModel, using audioURL: URL) async throws {
        _ = audioURL
        let language = selectedLanguage(for: model)
        _ = try await CohereNativeRuntime.shared.warmupModel(language: language)
    }

    func cleanup() async {
        await CohereNativeRuntime.shared.clearPreparedBootstraps()
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let cohereModel = model as? LocalCohereTranscribeModel else {
            throw CohereTranscribeServiceError.unsupportedModel
        }

        let language = selectedLanguage(for: cohereModel)
        let result = try await CohereNativeRuntime.shared.transcribe(
            audioURL: audioURL,
            language: language
        )
        logger.notice(
            "Cohere MLX transcription completed with stop reason \(String(describing: result.stopReason), privacy: .public) and \(result.generatedTokenIDs.count, privacy: .public) generated tokens"
        )
        return result.text
    }

    private func selectedLanguage(for model: LocalCohereTranscribeModel) -> String {
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage")
        guard let selectedLanguage,
              model.supportedLanguages[selectedLanguage] != nil else {
            return LocalCohereTranscribeConfiguration.fallbackLanguage
        }

        return selectedLanguage
    }
}
