import Foundation
import Testing
@testable import VoiceInk

struct MinimalModePolicyTests {
    @Test
    func enablingMinimalModePreservesSelectedModel() {
        let defaults = makeDefaults()
        defaults.set("whisper-large-v3-turbo", forKey: "CurrentTranscriptionModel")

        MinimalModePolicy.setEnabled(true, in: defaults)

        #expect(MinimalModePolicy.isEnabled(in: defaults))
        #expect(defaults.string(forKey: "CurrentTranscriptionModel") == "whisper-large-v3-turbo")
        #expect(!MinimalModePolicy.shouldPersistTranscriptions(in: defaults))
    }

    @Test
    func backgroundActivityRequiresPreferenceAndMinimalModeToBeOff() {
        let defaults = makeDefaults()

        MinimalModePolicy.setEnabled(false, in: defaults)
        #expect(MinimalModePolicy.allowsBackgroundNetworkActivity(requested: true, in: defaults))
        #expect(!MinimalModePolicy.allowsBackgroundNetworkActivity(requested: false, in: defaults))

        MinimalModePolicy.setEnabled(true, in: defaults)
        #expect(!MinimalModePolicy.allowsBackgroundNetworkActivity(requested: true, in: defaults))
    }

    @Test
    func temporaryRecordingIsDiscardedOnlyWhenMinimalModeIsEnabled() throws {
        let defaults = makeDefaults()
        let fileManager = FileManager.default
        let directory = fileManager.temporaryDirectory
            .appendingPathComponent("VoiceInkMinimalModeTests-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: directory) }

        let recordingURL = directory.appendingPathComponent("recording.wav")
        try Data("audio".utf8).write(to: recordingURL)

        MinimalModePolicy.setEnabled(false, in: defaults)
        #expect(
            !MinimalModePolicy.discardRecordingIfNeeded(
                at: recordingURL,
                defaults: defaults,
                fileManager: fileManager
            )
        )
        #expect(fileManager.fileExists(atPath: recordingURL.path))

        MinimalModePolicy.setEnabled(true, in: defaults)
        #expect(
            MinimalModePolicy.discardRecordingIfNeeded(
                at: recordingURL,
                defaults: defaults,
                fileManager: fileManager
            )
        )
        #expect(!fileManager.fileExists(atPath: recordingURL.path))
    }

    private func makeDefaults() -> UserDefaults {
        let suiteName = "VoiceInk.MinimalModePolicyTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}
