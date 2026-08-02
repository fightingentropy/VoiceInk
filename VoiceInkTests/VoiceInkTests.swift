//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Foundation
import Testing
@testable import VoiceInk

struct VoiceInkTests {

    @Test func immediateLiveTranscriptPasteIsEnabledByDefault() {
        #expect(AppDefaults.pasteLiveTranscriptImmediatelyDefault)
    }

    @Test func removedMiddleClickPreferencesAreCleared() {
        let suiteName = "VoiceInkTests.middleClickMigration.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set(true, forKey: "isMiddleClickToggleEnabled")
        defaults.set(200, forKey: "middleClickActivationDelay")

        AppDefaults.clearRemovedFeatureValues(in: defaults)

        #expect(defaults.object(forKey: "isMiddleClickToggleEnabled") == nil)
        #expect(defaults.object(forKey: "middleClickActivationDelay") == nil)
    }

    @Test func appStoragePathsStayInsideVoiceInkAppSupportFolder() async throws {
        let appSupportPath = AppStoragePaths.applicationSupportDirectory.path

        #expect(appSupportPath.contains("/Library/Application Support/com.fightingentropy.VoiceInk"))
        #expect(!AppStoragePaths.customSoundsDirectory.path.contains("/Library/Application Support/VoiceInk/"))
        #expect(AppStoragePaths.recordingsDirectory.deletingLastPathComponent() == AppStoragePaths.applicationSupportDirectory)
        #expect(AppStoragePaths.modelsDirectory.deletingLastPathComponent() == AppStoragePaths.applicationSupportDirectory)
        #expect(AppStoragePaths.whisperKitModelsDirectory.deletingLastPathComponent() == AppStoragePaths.modelsDirectory)
    }

    @Test func miniRecorderWaveformUsesAFixedCompactPill() {
        #expect(MiniRecorderWaveformLayout.pillSize == CGSize(width: 104, height: 34))
        #expect(MiniRecorderWaveformLayout.panelSize == CGSize(width: 120, height: 50))
    }

    @Test func immediateLiveTranscriptRequiresTheOptionAStreamingSessionAndVisibleText() {
        #expect(
            LiveTranscriptReleasePolicy.immediateText(
                from: "  exact live words  ",
                isEnabled: true,
                hasStreamingSession: true
            ) == "exact live words"
        )
        #expect(
            LiveTranscriptReleasePolicy.immediateText(
                from: "exact live words",
                isEnabled: false,
                hasStreamingSession: true
            ) == nil
        )
        #expect(
            LiveTranscriptReleasePolicy.immediateText(
                from: "exact live words",
                isEnabled: true,
                hasStreamingSession: false
            ) == nil
        )
        #expect(
            LiveTranscriptReleasePolicy.immediateText(
                from: "  \n ",
                isEnabled: true,
                hasStreamingSession: true
            ) == nil
        )
    }

}
