//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Testing
@testable import VoiceInk

struct VoiceInkTests {

    @Test func appStoragePathsStayInsideVoiceInkAppSupportFolder() async throws {
        let appSupportPath = AppStoragePaths.applicationSupportDirectory.path

        #expect(appSupportPath.contains("/Library/Application Support/com.fightingentropy.VoiceInk"))
        #expect(!AppStoragePaths.customSoundsDirectory.path.contains("/Library/Application Support/VoiceInk/"))
        #expect(AppStoragePaths.recordingsDirectory.deletingLastPathComponent() == AppStoragePaths.applicationSupportDirectory)
        #expect(AppStoragePaths.whisperModelsDirectory.deletingLastPathComponent() == AppStoragePaths.applicationSupportDirectory)
    }

}
