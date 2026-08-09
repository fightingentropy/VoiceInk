import Foundation

enum EphemeralTranscriptionPolicy {
    @discardableResult
    static func discardRecordings(
        referencedBy urls: inout Set<URL>,
        fileManager: FileManager = .default
    ) -> Bool {
        guard !urls.isEmpty else {
            return false
        }

        var discardedEveryRecording = true
        for recordingURL in Array(urls) {
            let discarded = discardRecording(at: recordingURL, fileManager: fileManager)
            if discarded || !fileManager.fileExists(atPath: recordingURL.path) {
                urls.remove(recordingURL)
            } else {
                discardedEveryRecording = false
            }
        }
        return discardedEveryRecording
    }

    @discardableResult
    static func discardRecording(
        referencedBy url: inout URL?,
        fileManager: FileManager = .default
    ) -> Bool {
        guard let recordingURL = url else {
            return false
        }

        let discarded = discardRecording(at: recordingURL, fileManager: fileManager)
        if discarded || !fileManager.fileExists(atPath: recordingURL.path) {
            url = nil
        }
        return discarded
    }

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
