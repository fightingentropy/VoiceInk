import Foundation

enum AppDefaults {
    static func registerDefaults() {
        clearRemovedFeatureValues()

        UserDefaults.standard.register(defaults: [
            // Onboarding & General
            "hasCompletedOnboarding": false,
            "enableAnnouncements": true,
            "autoUpdateCheck": true,

            // Clipboard
            "restoreClipboardAfterPaste": true,
            "clipboardRestoreDelay": 2.0,
            "useAppleScriptPaste": false,

            // Audio & Media
            "isSystemMuteEnabled": true,
            "audioResumptionDelay": 0.0,
            "isPauseMediaEnabled": false,
            "isSoundFeedbackEnabled": true,

            // Recording & Transcription
            "IsTextFormattingEnabled": true,
            "RemoveFillerWords": true,
            "SelectedLanguage": "en",
            "AppendTrailingSpace": true,
            "RecorderType": "mini",

            // Cleanup
            "IsTranscriptionCleanupEnabled": false,
            "TranscriptionRetentionMinutes": 1440,
            "IsAudioCleanupEnabled": false,
            "AudioRetentionPeriod": 7,

            // UI & Behavior
            "IsMenuBarOnly": false,
            // Hotkey
            "isMiddleClickToggleEnabled": false,
            "middleClickActivationDelay": 200,

            // Model
            "PrewarmModelOnWake": true,
            "LocalModelWarmRetentionSeconds": LocalModelWarmRetention.fifteenMinutes.rawValue,
            "LocalVoxtralModelName": LocalVoxtralConfiguration.defaultModelName,
        ])
    }

    private static func clearRemovedFeatureValues() {
        let keys = [
            "isAIEnhancementEnabled",
            "useClipboardContext",
            "useScreenCaptureContext",
            "customPrompts",
            "selectedPromptId",
            "selectedAIProvider",
            "OpenAISelectedModel",
            "isToggleEnhancementShortcutEnabled",
            "powerModeAutoRestoreEnabled",
            "powerModeUIFlag",
            "powerModeConfigurationsV2",
            "powerModeActiveSession.v1",
            "customPowerModeEmojis",
            "CustomVocabularyItems",
            "RecorderType"
        ]

        for key in keys {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }
}
