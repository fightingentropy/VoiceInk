import Foundation
import Testing
@testable import VoiceInk

struct EphemeralTranscriptionPolicyTests {
    @Test
    func policyDoesNotTouchSelectedModel() {
        let defaults = makeDefaults()
        defaults.set("whisper-large-v3-turbo", forKey: "CurrentTranscriptionModel")

        let missingRecording = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        _ = EphemeralTranscriptionPolicy.discardRecording(at: missingRecording)

        #expect(defaults.string(forKey: "CurrentTranscriptionModel") == "whisper-large-v3-turbo")
    }

    @Test
    func temporaryRecordingIsAlwaysDiscarded() throws {
        let fileManager = FileManager.default
        let directory = fileManager.temporaryDirectory
            .appendingPathComponent("VoiceInkEphemeralTests-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: directory) }

        let recordingURL = directory.appendingPathComponent("recording.wav")
        try Data("audio".utf8).write(to: recordingURL)

        #expect(
            EphemeralTranscriptionPolicy.discardRecording(
                at: recordingURL,
                fileManager: fileManager
            )
        )
        #expect(!fileManager.fileExists(atPath: recordingURL.path))
    }

    private func makeDefaults() -> UserDefaults {
        let suiteName = "VoiceInk.EphemeralTranscriptionPolicyTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}
