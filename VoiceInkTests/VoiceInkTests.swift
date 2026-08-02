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

    @Test func miniRecorderPillExpandsWithTranscriptAndWrapsInsteadOfTruncating() {
        let collapsed = MiniRecorderPillLayout.size(for: "", isTranscriptVisible: true)
        let short = MiniRecorderPillLayout.size(for: "hello", isTranscriptVisible: true)
        let sentence = MiniRecorderPillLayout.size(
            for: "hello this sentence should remain visible as the live transcript grows",
            isTranscriptVisible: true
        )
        let reportedLayoutIssue = MiniRecorderPillLayout.size(
            for: "It's fine, no need for the custom",
            isTranscriptVisible: true
        )
        let providerLineBreak = MiniRecorderPillLayout.size(
            for: "This is\na test",
            isTranscriptVisible: true
        )
        let normalizedShortPhrase = MiniRecorderPillLayout.size(
            for: "This is a test",
            isTranscriptVisible: true
        )
        let multilineProviderText = """
        This is a test
        But the blueberries have been shown to improve cognition.
        Everyone has every reason to try at least one cup a day.
        """
        let normalizedProviderText = MiniRecorderPillLayout.displayText(for: multilineProviderText)
        let multilineProviderSize = MiniRecorderPillLayout.size(
            for: multilineProviderText,
            isTranscriptVisible: true
        )
        let normalizedProviderSize = MiniRecorderPillLayout.size(
            for: normalizedProviderText,
            isTranscriptVisible: true
        )
        let long = MiniRecorderPillLayout.size(
            for: String(repeating: "the complete live transcript remains visible ", count: 12),
            isTranscriptVisible: true
        )
        let extremelyLong = MiniRecorderPillLayout.size(
            for: String(repeating: "the live transcript stays inside its scrollable panel ", count: 100),
            isTranscriptVisible: true
        )
        let hidden = MiniRecorderPillLayout.size(for: "hello", isTranscriptVisible: false)

        #expect(collapsed == MiniRecorderPillLayout.collapsedSize)
        #expect(hidden == MiniRecorderPillLayout.collapsedSize)
        #expect(short.width >= collapsed.width)
        #expect(sentence.width > short.width)
        #expect(reportedLayoutIssue.height <= MiniRecorderPillLayout.expandedHeight + 1)
        #expect(reportedLayoutIssue.width < 200)
        #expect(MiniRecorderPillLayout.displayText(for: "This is\na test") == "This is a test")
        #expect(providerLineBreak == normalizedShortPhrase)
        #expect(providerLineBreak.height <= MiniRecorderPillLayout.expandedHeight + 1)
        #expect(multilineProviderSize == normalizedProviderSize)
        #expect(long.width == MiniRecorderPillLayout.maximumWidth)
        #expect(MiniRecorderPillLayout.maximumWidth == 540)
        #expect(long.height > MiniRecorderPillLayout.expandedHeight)
        #expect(long.height <= MiniRecorderPillLayout.maximumHeight)
        #expect(extremelyLong.height == MiniRecorderPillLayout.maximumHeight)
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
