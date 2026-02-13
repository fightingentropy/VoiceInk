import Foundation
import SwiftUI
import LLMkit

class OllamaService: ObservableObject {
    static let defaultBaseURL = "http://localhost:11434"

    // MARK: - Published Properties
    @Published var baseURL: String {
        didSet {
            UserDefaults.standard.set(baseURL, forKey: "ollamaBaseURL")
        }
    }

    @Published var selectedModel: String {
        didSet {
            UserDefaults.standard.set(selectedModel, forKey: "ollamaSelectedModel")
        }
    }

    @Published var availableModels: [OllamaModel] = []
    @Published var isConnected: Bool = false
    @Published var isLoadingModels: Bool = false

    private let defaultTemperature: Double = 0.3

    init() {
        self.baseURL = UserDefaults.standard.string(forKey: "ollamaBaseURL") ?? Self.defaultBaseURL
        self.selectedModel = UserDefaults.standard.string(forKey: "ollamaSelectedModel") ?? "llama2"
    }

    private var baseURLValue: URL {
        URL(string: baseURL) ?? OllamaClient.defaultBaseURL
    }

    @MainActor
    func checkConnection() async {
        isConnected = await OllamaClient.checkConnection(baseURL: baseURLValue)
    }

    @MainActor
    func refreshModels() async {
        isLoadingModels = true
        defer { isLoadingModels = false }

        do {
            let models = try await OllamaClient.fetchModels(baseURL: baseURLValue)
            availableModels = models

            if !models.contains(where: { $0.name == selectedModel }) && !models.isEmpty {
                selectedModel = models[0].name
            }
        } catch {
            print("Error fetching models: \(error)")
            availableModels = []
        }
    }

    func enhance(_ text: String, withSystemPrompt systemPrompt: String? = nil) async throws -> String {
        guard let systemPrompt = systemPrompt else {
            throw LocalAIError.invalidRequest
        }

        do {
            return try await OllamaClient.generate(
                baseURL: baseURLValue,
                model: selectedModel,
                prompt: text,
                systemPrompt: systemPrompt,
                temperature: defaultTemperature
            )
        } catch let error as LLMKitError {
            throw mapLLMKitError(error)
        }
    }

    private func mapLLMKitError(_ error: LLMKitError) -> LocalAIError {
        switch error {
        case .invalidURL:
            return .invalidURL
        case .httpError(let statusCode, _):
            if statusCode == 404 { return .modelNotFound }
            if statusCode == 500 { return .serverError }
            return .invalidResponse
        case .networkError:
            return .serviceUnavailable
        case .noResultReturned, .decodingError:
            return .invalidResponse
        case .encodingError:
            return .invalidRequest
        case .missingAPIKey, .timeout:
            return .invalidResponse
        }
    }
}

// MARK: - Error Types
enum LocalAIError: Error, LocalizedError {
    case invalidURL
    case serviceUnavailable
    case invalidResponse
    case modelNotFound
    case serverError
    case invalidRequest

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid Ollama server URL"
        case .serviceUnavailable:
            return "Ollama service is not available"
        case .invalidResponse:
            return "Invalid response from Ollama server"
        case .modelNotFound:
            return "Selected model not found"
        case .serverError:
            return "Ollama server error"
        case .invalidRequest:
            return "System prompt is required"
        }
    }
}
