import Foundation

enum AppDefaults {
    static let pasteLiveTranscriptImmediatelyKey = "PasteLiveTranscriptImmediately"
    static let pasteLiveTranscriptImmediatelyDefault = true

    static func registerDefaults() {
        clearRemovedFeatureValues(in: .standard)

        UserDefaults.standard.register(defaults: [
            // Onboarding & General
            "hasCompletedOnboarding": false,

            // Clipboard
            "restoreClipboardAfterPaste": true,
            "clipboardRestoreDelay": 2.0,

            // Audio & Media
            "isSystemMuteEnabled": true,
            "audioResumptionDelay": 0.0,
            "isPauseMediaEnabled": false,
            "isSoundFeedbackEnabled": true,

            // Recording & Transcription
            "IsTextFormattingEnabled": true,
            "RemoveFillerWords": true,
            "ConvertSpokenPunctuation": true,
            "ConvertLiteralDictationTokens": true,
            "SelectedLanguage": "en",
            "AppendTrailingSpace": true,
            pasteLiveTranscriptImmediatelyKey: pasteLiveTranscriptImmediatelyDefault,
            "RecorderType": "mini",

            // Cleanup
            "IsTranscriptionCleanupEnabled": false,
            "TranscriptionRetentionMinutes": 1440,
            "IsAudioCleanupEnabled": false,
            "AudioRetentionPeriod": 7,

            // Model
            "PrewarmModelOnWake": false,
            "LocalModelWarmRetentionSeconds": LocalModelWarmRetention.fiveMinutes.rawValue,
            "LocalVoxtralModelName": LocalVoxtralConfiguration.defaultModelName,
        ])
    }

    static func clearRemovedFeatureValues(in defaults: UserDefaults) {
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
            "useAppleScriptPaste",
            "CustomVocabularyItems",
            "isMiddleClickToggleEnabled",
            "middleClickActivationDelay",
            "RecorderType"
        ]

        for key in keys {
            defaults.removeObject(forKey: key)
        }
    }
}
