import Foundation

enum EphemeralTranscriptionPolicy {
    @discardableResult
    static func discardRecording(
        at url: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        guard fileManager.fileExists(atPath: url.path) else {
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
