import Foundation

extension Notification.Name {
    static let AppSettingsDidChange = Notification.Name("appSettingsDidChange")
    static let languageDidChange = Notification.Name("languageDidChange")
    static let toggleMiniRecorder = Notification.Name("toggleMiniRecorder")
    static let dismissMiniRecorder = Notification.Name("dismissMiniRecorder")
    static let didChangeModel = Notification.Name("didChangeModel")
    static let navigateToDestination = Notification.Name("navigateToDestination")
    static let transcriptionCreated = Notification.Name("transcriptionCreated")
    static let transcriptionCompleted = Notification.Name("transcriptionCompleted")
    static let transcriptionDeleted = Notification.Name("transcriptionDeleted")
    static let openFileForTranscription = Notification.Name("openFileForTranscription")
    static let audioDeviceSwitchRequired = Notification.Name("audioDeviceSwitchRequired")
    static let localModelDidUse = Notification.Name("localModelDidUse")
    static let cohereTranscribeAvailabilityDidChange = Notification.Name("cohereTranscribeAvailabilityDidChange")
}
