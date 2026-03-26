import Foundation

enum LocalCohereTranscribeConfiguration {
    static let modelNameKey = "LocalCohereTranscribeModelName"
    static let defaultModelName = "CohereLabs/cohere-transcribe-03-2026"
    static let runtimeVersion = 1
    static let huggingFaceProviderName = "HuggingFace"
    static let workerScriptName = "cohere_transcribe_worker"
    static let fallbackLanguage = "en"

    static var modelName: String {
        let stored = UserDefaults.standard.string(forKey: modelNameKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return stored?.isEmpty == false ? stored! : defaultModelName
    }
}
