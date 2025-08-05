import Foundation
import AppKit

// Represents the state of the application that can be modified by a Power Mode.
// This struct captures the settings that will be temporarily overridden.
struct ApplicationState: Codable {
    var isEnhancementEnabled: Bool
    var useScreenCaptureContext: Bool
    var selectedPromptId: String? // Storing as String for Codable simplicity
    var selectedAIProvider: String?
    var selectedAIModel: String?
    var selectedLanguage: String?
    var transcriptionModelName: String?
}

// Represents an active Power Mode session.
struct PowerModeSession: Codable {
    let id: UUID
    let startTime: Date
    var originalState: ApplicationState
}

@MainActor
class PowerModeSessionManager {
    static let shared = PowerModeSessionManager()
    private let sessionKey = "powerModeActiveSession.v1"

    private var whisperState: WhisperState?
    private var enhancementService: AIEnhancementService?

    private init() {
        // Attempt to recover a session on startup in case of a crash.
        recoverSession()
    }

    func configure(whisperState: WhisperState, enhancementService: AIEnhancementService) {
        self.whisperState = whisperState
        self.enhancementService = enhancementService
    }

    // Begins a new Power Mode session. It captures the current state,
    // applies the new configuration, and saves the session.
    func beginSession(with config: PowerModeConfig) async {
        guard let whisperState = whisperState, let enhancementService = enhancementService else {
            print("SessionManager not configured.")
            return
        }

        // 1. Capture the current application state.
        let originalState = ApplicationState(
            isEnhancementEnabled: enhancementService.isEnhancementEnabled,
            useScreenCaptureContext: enhancementService.useScreenCaptureContext,
            selectedPromptId: enhancementService.selectedPromptId?.uuidString,
            selectedAIProvider: enhancementService.getAIService()?.selectedProvider.rawValue,
            selectedAIModel: enhancementService.getAIService()?.currentModel,
            selectedLanguage: UserDefaults.standard.string(forKey: "SelectedLanguage"),
            transcriptionModelName: whisperState.currentTranscriptionModel?.name
        )

        // 2. Create and save the session.
        let newSession = PowerModeSession(
            id: UUID(),
            startTime: Date(),
            originalState: originalState
        )
        saveSession(newSession)

        // 3. Apply the new configuration's settings.
        await applyConfiguration(config)
    }

    // Ends the current Power Mode session and restores the original state.
    func endSession() async {
        guard let session = loadSession() else { return }

        // Restore the original state from the session.
        await restoreState(session.originalState)

        // Clear the session from UserDefaults.
        clearSession()
    }

    // Applies the settings from a PowerModeConfig.
    private func applyConfiguration(_ config: PowerModeConfig) async {
        guard let enhancementService = enhancementService else { return }

        await MainActor.run {
            enhancementService.isEnhancementEnabled = config.isAIEnhancementEnabled
            enhancementService.useScreenCaptureContext = config.useScreenCapture

            if config.isAIEnhancementEnabled {
                if let promptId = config.selectedPrompt, let uuid = UUID(uuidString: promptId) {
                    enhancementService.selectedPromptId = uuid
                }

                if let aiService = enhancementService.getAIService() {
                    if let providerName = config.selectedAIProvider, let provider = AIProvider(rawValue: providerName) {
                        aiService.selectedProvider = provider
                    }
                    if let model = config.selectedAIModel {
                        aiService.selectModel(model)
                    }
                }
            }

            if let language = config.selectedLanguage {
                UserDefaults.standard.set(language, forKey: "SelectedLanguage")
                NotificationCenter.default.post(name: .languageDidChange, object: nil)
            }
        }

        if let whisperState = whisperState,
           let modelName = config.selectedTranscriptionModelName,
           let selectedModel = await whisperState.allAvailableModels.first(where: { $0.name == modelName }),
           whisperState.currentTranscriptionModel?.name != modelName {
            await handleModelChange(to: selectedModel)
        }
    }

    // Restores the application state from a saved state object.
    private func restoreState(_ state: ApplicationState) async {
        guard let enhancementService = enhancementService else { return }

        await MainActor.run {
            enhancementService.isEnhancementEnabled = state.isEnhancementEnabled
            enhancementService.useScreenCaptureContext = state.useScreenCaptureContext
            enhancementService.selectedPromptId = state.selectedPromptId.flatMap(UUID.init)

            if let aiService = enhancementService.getAIService() {
                if let providerName = state.selectedAIProvider, let provider = AIProvider(rawValue: providerName) {
                    aiService.selectedProvider = provider
                }
                if let model = state.selectedAIModel {
                    aiService.selectModel(model)
                }
            }

            if let language = state.selectedLanguage {
                UserDefaults.standard.set(language, forKey: "SelectedLanguage")
                NotificationCenter.default.post(name: .languageDidChange, object: nil)
            }
        }

        if let whisperState = whisperState,
           let modelName = state.transcriptionModelName,
           let selectedModel = await whisperState.allAvailableModels.first(where: { $0.name == modelName }),
           whisperState.currentTranscriptionModel?.name != modelName {
            await handleModelChange(to: selectedModel)
        }
    }
    
    // Handles the logic for switching transcription models.
    private func handleModelChange(to newModel: any TranscriptionModel) async {
        guard let whisperState = whisperState else { return }

        await whisperState.setDefaultTranscriptionModel(newModel)

        switch newModel.provider {
        case .local:
            await whisperState.cleanupModelResources()
            if let localModel = await whisperState.availableModels.first(where: { $0.name == newModel.name }) {
                do {
                    try await whisperState.loadModel(localModel)
                } catch {
                    // Log error appropriately
                    print("Power Mode: Failed to load local model '\(localModel.name)': \(error)")
                }
            }
        case .parakeet:
            await whisperState.cleanupModelResources()
            // Parakeet models are loaded on demand, so we only need to clean up.

        default:
            await whisperState.cleanupModelResources()
        }
    }
    
    private func recoverSession() {
        guard let session = loadSession() else { return }
        print("Recovering abandoned Power Mode session.")
        Task {
            await endSession()
        }
    }

    // MARK: - UserDefaults Persistence

    private func saveSession(_ session: PowerModeSession) {
        do {
            let data = try JSONEncoder().encode(session)
            UserDefaults.standard.set(data, forKey: sessionKey)
        } catch {
            print("Error saving Power Mode session: \(error)")
        }
    }
    
    private func loadSession() -> PowerModeSession? {
        guard let data = UserDefaults.standard.data(forKey: sessionKey) else { return nil }
        do {
            return try JSONDecoder().decode(PowerModeSession.self, from: data)
        } catch {
            print("Error loading Power Mode session: \(error)")
            return nil
        }
    }

    private func clearSession() {
        UserDefaults.standard.removeObject(forKey: sessionKey)
    }
}
