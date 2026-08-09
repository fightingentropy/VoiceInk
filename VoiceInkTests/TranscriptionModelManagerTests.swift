import Foundation
import Testing
@testable import VoiceInk

@MainActor
struct TranscriptionModelManagerTests {
    @Test
    func selectsOpenAIAsDefaultWhenConfigured() {
        let (defaults, suiteName) = makeIsolatedDefaults()
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let manager = TranscriptionModelManager(
            whisperModelManager: WhisperModelManager(),
            userDefaults: defaults,
            hasAPIKey: { $0.lowercased() == "openai" }
        )

        manager.refreshAllAvailableModels()
        manager.loadCurrentTranscriptionModel()

        #expect(manager.currentTranscriptionModel?.name == "gpt-live-transcribe")
    }

    @Test
    func doesNotSelectOpenAIWithoutConfiguredKey() {
        let (defaults, suiteName) = makeIsolatedDefaults()
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let manager = TranscriptionModelManager(
            whisperModelManager: WhisperModelManager(),
            userDefaults: defaults,
            hasAPIKey: { _ in false }
        )

        manager.refreshAllAvailableModels()
        manager.loadCurrentTranscriptionModel()

        #expect(manager.currentTranscriptionModel == nil)
    }

    @Test
    func fallsBackToXAIWhenOpenAIIsNotConfigured() {
        let (defaults, suiteName) = makeIsolatedDefaults()
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let manager = TranscriptionModelManager(
            whisperModelManager: WhisperModelManager(),
            userDefaults: defaults,
            hasAPIKey: { $0.lowercased() == "xai" }
        )

        manager.refreshAllAvailableModels()
        manager.loadCurrentTranscriptionModel()

        #expect(manager.currentTranscriptionModel?.name == "xai-stt")
    }

    @Test
    func refreshReplacesADeletedCurrentModelWithAUsableFallback() {
        let (defaults, suiteName) = makeIsolatedDefaults()
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let fallback = CloudModel(
            name: "xai-stt",
            displayName: "xAI Speech-to-Text",
            description: "Fallback",
            provider: .xAI,
            speed: 1,
            accuracy: 1,
            isMultilingual: true,
            supportedLanguages: ["en": "English"]
        )
        let deletedModel = CloudModel(
            name: "deleted-custom-model",
            displayName: "Deleted Custom Model",
            description: "Removed",
            provider: .custom,
            speed: 1,
            accuracy: 1,
            isMultilingual: true,
            supportedLanguages: ["en": "English"]
        )
        let manager = TranscriptionModelManager(
            whisperModelManager: WhisperModelManager(),
            userDefaults: defaults,
            hasAPIKey: { $0.lowercased() == "xai" },
            availableModels: { [fallback] }
        )

        manager.setDefaultTranscriptionModel(deletedModel)
        manager.refreshAllAvailableModels()

        #expect(manager.currentTranscriptionModel?.name == "xai-stt")
        #expect(defaults.string(forKey: "CurrentTranscriptionModel") == "xai-stt")
    }

    private func makeIsolatedDefaults() -> (UserDefaults, String) {
        let suiteName = "VoiceInk.TranscriptionModelManagerTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return (defaults, suiteName)
    }
}
