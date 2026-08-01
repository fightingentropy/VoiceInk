//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Testing
@testable import VoiceInk

struct VoiceInkTests {

    @Test func immediateLiveTranscriptPasteIsEnabledByDefault() {
        #expect(AppDefaults.pasteLiveTranscriptImmediatelyDefault)
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
        let long = MiniRecorderPillLayout.size(
            for: String(repeating: "the complete live transcript remains visible ", count: 12),
            isTranscriptVisible: true
        )
        let hidden = MiniRecorderPillLayout.size(for: "hello", isTranscriptVisible: false)

        #expect(collapsed == MiniRecorderPillLayout.collapsedSize)
        #expect(hidden == MiniRecorderPillLayout.collapsedSize)
        #expect(short.width > collapsed.width)
        #expect(sentence.width > short.width)
        #expect(long.width == MiniRecorderPillLayout.maximumWidth)
        #expect(long.height > MiniRecorderPillLayout.expandedHeight)
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
