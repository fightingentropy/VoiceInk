import Foundation

enum MinimalModePolicy {
    static let enabledKey = "MinimalModeEnabled"

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: enabledKey)
    }

    static func setEnabled(_ enabled: Bool, in defaults: UserDefaults = .standard) {
        defaults.set(enabled, forKey: enabledKey)
    }

    static func shouldPersistTranscriptions(in defaults: UserDefaults = .standard) -> Bool {
        !isEnabled(in: defaults)
    }

    static func allowsBackgroundNetworkActivity(
        requested: Bool,
        in defaults: UserDefaults = .standard
    ) -> Bool {
        requested && !isEnabled(in: defaults)
    }

    @discardableResult
    static func discardRecordingIfNeeded(
        at url: URL,
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default
    ) -> Bool {
        discardRecording(at: url, when: isEnabled(in: defaults), fileManager: fileManager)
    }

    @discardableResult
    static func discardRecording(
        at url: URL,
        when enabled: Bool,
        fileManager: FileManager = .default
    ) -> Bool {
        guard enabled, fileManager.fileExists(atPath: url.path) else {
            return false
        }

        do {
            try fileManager.removeItem(at: url)
            return true
        } catch {
            return false
        }
    }
}
