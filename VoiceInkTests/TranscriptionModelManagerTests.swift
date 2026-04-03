import Foundation
import Testing
@testable import VoiceInk

@MainActor
struct TranscriptionModelManagerTests {
    @Test
    func selectsElevenLabsAsDefaultWhenConfigured() {
        let defaults = UserDefaults.standard
        let currentModelKey = "CurrentTranscriptionModel"
        let localKeychainKey = "LocalKeychain_elevenLabsAPIKey"

        let previousModelValue = defaults.object(forKey: currentModelKey)
        let previousAPIKeyValue = defaults.object(forKey: localKeychainKey)

        defer {
            restore(defaults: defaults, value: previousModelValue, forKey: currentModelKey)
            restore(defaults: defaults, value: previousAPIKeyValue, forKey: localKeychainKey)
        }

        defaults.removeObject(forKey: currentModelKey)
        defaults.set(Data("test-elevenlabs-key".utf8), forKey: localKeychainKey)

        let manager = TranscriptionModelManager(
            whisperModelManager: WhisperModelManager(),
            parakeetModelManager: ParakeetModelManager()
        )

        manager.refreshAllAvailableModels()
        manager.loadCurrentTranscriptionModel()

        #expect(manager.currentTranscriptionModel?.name == "scribe_v2")
    }

    @Test
    func doesNotSelectElevenLabsWithoutConfiguredKey() {
        let defaults = UserDefaults.standard
        let currentModelKey = "CurrentTranscriptionModel"
        let localKeychainKey = "LocalKeychain_elevenLabsAPIKey"

        let previousModelValue = defaults.object(forKey: currentModelKey)
        let previousAPIKeyValue = defaults.object(forKey: localKeychainKey)

        defer {
            restore(defaults: defaults, value: previousModelValue, forKey: currentModelKey)
            restore(defaults: defaults, value: previousAPIKeyValue, forKey: localKeychainKey)
        }

        defaults.removeObject(forKey: currentModelKey)
        defaults.removeObject(forKey: localKeychainKey)

        let manager = TranscriptionModelManager(
            whisperModelManager: WhisperModelManager(),
            parakeetModelManager: ParakeetModelManager()
        )

        manager.refreshAllAvailableModels()
        manager.loadCurrentTranscriptionModel()

        #expect(manager.currentTranscriptionModel == nil)
    }

    private func restore(defaults: UserDefaults, value: Any?, forKey key: String) {
        if let value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }
}
