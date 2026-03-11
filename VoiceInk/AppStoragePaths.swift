import Foundation

enum AppStoragePaths {
    private static let appSupportFolderName = "com.fightingentropy.VoiceInk"

    static var applicationSupportDirectory: URL {
        let baseDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return baseDirectory.appendingPathComponent(appSupportFolderName, isDirectory: true)
    }

    static var recordingsDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("Recordings", isDirectory: true)
    }

    static var whisperModelsDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("WhisperModels", isDirectory: true)
    }

    static var customSoundsDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("CustomSounds", isDirectory: true)
    }

    static var defaultStoreURL: URL {
        applicationSupportDirectory.appendingPathComponent("default.store")
    }

    static var dictionaryStoreURL: URL {
        applicationSupportDirectory.appendingPathComponent("dictionary.store")
    }

    static func createDirectoryIfNeeded(at directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
}
