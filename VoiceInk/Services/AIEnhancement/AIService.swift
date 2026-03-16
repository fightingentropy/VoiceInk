import Foundation
import LLMkit

enum AIProvider: String, CaseIterable {
    case openAI = "OpenAI"
}

@MainActor
class AIService: ObservableObject {
    @Published var apiKey: String = ""
    @Published var isAPIKeyValid: Bool = false
    @Published var selectedProvider: AIProvider = .openAI {
        didSet {
            userDefaults.set(selectedProvider.rawValue, forKey: "selectedAIProvider")
            reloadAPIKeyState()
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
        }
    }

    @Published private var selectedModels: [AIProvider: String] = [:]

    private let userDefaults = UserDefaults.standard

    var connectedProviders: [AIProvider] {
        APIKeyManager.shared.hasAPIKey(forProvider: selectedProvider.rawValue) ? [.openAI] : []
    }

    var currentModel: String {
        if let selectedModel = selectedModels[selectedProvider],
           availableModels.contains(selectedModel) {
            return selectedModel
        }
        return selectedProvider.defaultModel
    }

    var availableModels: [String] {
        selectedProvider.availableModels
    }

    init() {
        userDefaults.set(AIProvider.openAI.rawValue, forKey: "selectedAIProvider")
        loadSavedModelSelections()
        reloadAPIKeyState()
    }

    func selectModel(_ model: String) {
        guard !model.isEmpty else { return }

        selectedModels[selectedProvider] = model
        userDefaults.set(model, forKey: "\(selectedProvider.rawValue)SelectedModel")
        objectWillChange.send()
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }

    func saveAPIKey(_ key: String, completion: @MainActor @escaping (Bool, String?) -> Void) {
        verifyAPIKey(key) { [weak self] isValid, errorMessage in
            guard let self else { return }
            Task { @MainActor in
                if isValid {
                    apiKey = key
                    isAPIKeyValid = true
                    APIKeyManager.shared.saveAPIKey(key, forProvider: selectedProvider.rawValue)
                    NotificationCenter.default.post(name: .aiProviderKeyChanged, object: nil)
                } else {
                    isAPIKeyValid = false
                }
                completion(isValid, errorMessage)
            }
        }
    }

    func verifyAPIKey(_ key: String, completion: @MainActor @escaping (Bool, String?) -> Void) {
        let currentModel = self.currentModel
        Task {
            let result = await OpenAILLMClient.verifyAPIKey(
                baseURL: selectedProvider.baseURL,
                apiKey: key,
                model: currentModel
            )
            completion(result.isValid, result.errorMessage)
        }
    }

    func clearAPIKey() {
        apiKey = ""
        isAPIKeyValid = false
        APIKeyManager.shared.deleteAPIKey(forProvider: selectedProvider.rawValue)
        NotificationCenter.default.post(name: .aiProviderKeyChanged, object: nil)
    }

    private func loadSavedModelSelections() {
        for provider in AIProvider.allCases {
            let key = "\(provider.rawValue)SelectedModel"
            if let savedModel = userDefaults.string(forKey: key), !savedModel.isEmpty {
                selectedModels[provider] = savedModel
            }
        }
    }

    private func reloadAPIKeyState() {
        if let savedKey = APIKeyManager.shared.getAPIKey(forProvider: selectedProvider.rawValue) {
            apiKey = savedKey
            isAPIKeyValid = true
        } else {
            apiKey = ""
            isAPIKeyValid = false
        }
    }
}

private extension AIProvider {
    var baseURL: URL {
        URL(string: "https://api.openai.com/v1/chat/completions")!
    }

    var defaultModel: String {
        "gpt-5.4"
    }

    var availableModels: [String] {
        ["gpt-5.4"]
    }
}
