import Foundation

final class CohereTranscribeTranscriptionService: TranscriptionService, @unchecked Sendable {
    func prepareModel(for model: LocalCohereTranscribeModel) async throws {
        _ = model
        try await CohereTranscribePythonRuntime.shared.prepareModel()
    }

    func warmup(for model: LocalCohereTranscribeModel, using audioURL: URL) async throws {
        let language = selectedLanguage(for: model)
        try await CohereTranscribePythonRuntime.shared.warmup(audioURL: audioURL, language: language)
    }

    func cleanup() async {
        await CohereTranscribePythonRuntime.shared.shutdown()
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let cohereModel = model as? LocalCohereTranscribeModel else {
            throw CohereTranscribeSetupError.commandFailed("Unsupported Cohere Transcribe model.")
        }

        let language = selectedLanguage(for: cohereModel)
        return try await CohereTranscribePythonRuntime.shared.transcribe(audioURL: audioURL, language: language)
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
