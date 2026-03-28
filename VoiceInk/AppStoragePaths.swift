import Foundation

enum AppStoragePaths {
    private static let appSupportFolderName = "com.fightingentropy.VoiceInk"
    private static let legacyAppSupportFolderName = "VoiceInk"

    static var applicationSupportDirectory: URL {
        let baseDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return baseDirectory.appendingPathComponent(appSupportFolderName, isDirectory: true)
    }

    static var cachesDirectory: URL {
        let baseDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return baseDirectory.appendingPathComponent(appSupportFolderName, isDirectory: true)
    }

    static var recordingsDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("Recordings", isDirectory: true)
    }

    static var benchmarkDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("Benchmarks", isDirectory: true)
    }

    static var benchmarkReportsDirectory: URL {
        benchmarkDirectory.appendingPathComponent("Reports", isDirectory: true)
    }

    static var standardBenchmarkCorpusDirectory: URL {
        benchmarkDirectory.appendingPathComponent("StandardCorpus", isDirectory: true)
    }

    static var standardBenchmarkManifestURL: URL {
        standardBenchmarkCorpusDirectory.appendingPathComponent("standard-benchmark-corpus.json")
    }

    static var recentBenchmarkCorpusDirectory: URL {
        benchmarkDirectory.appendingPathComponent("RecentCorpus", isDirectory: true)
    }

    static var recentBenchmarkManifestURL: URL {
        recentBenchmarkCorpusDirectory.appendingPathComponent("recent-benchmark-corpus.json")
    }

    static var modelsDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("Models", isDirectory: true)
    }

    static var whisperKitModelsDirectory: URL {
        modelsDirectory.appendingPathComponent("WhisperKit", isDirectory: true)
    }

    static var voxtralModelsDirectory: URL {
        modelsDirectory.appendingPathComponent("Voxtral", isDirectory: true)
    }

    static var cohereTranscribeDirectory: URL {
        modelsDirectory.appendingPathComponent("CohereTranscribe", isDirectory: true)
    }

    static var cohereTranscribeNativeDirectory: URL {
        cohereTranscribeDirectory.appendingPathComponent("NativeMLX", isDirectory: true)
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

    static func migrateLegacyApplicationSupportIfNeeded() throws {
        let fileManager = FileManager.default
        let legacyDirectory = legacyApplicationSupportDirectory
        guard fileManager.fileExists(atPath: legacyDirectory.path) else { return }

        try createDirectoryIfNeeded(at: applicationSupportDirectory)
        try mergeDirectoryContents(from: legacyDirectory, into: applicationSupportDirectory, fileManager: fileManager)
        try removeDirectoryIfEmpty(legacyDirectory, fileManager: fileManager)
    }

    private static var legacyApplicationSupportDirectory: URL {
        applicationSupportDirectory
            .deletingLastPathComponent()
            .appendingPathComponent(legacyAppSupportFolderName, isDirectory: true)
    }

    private static func mergeDirectoryContents(from sourceDirectory: URL, into destinationDirectory: URL, fileManager: FileManager) throws {
        let sourceItems = try fileManager.contentsOfDirectory(
            at: sourceDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for sourceItem in sourceItems {
            let sourceValues = try sourceItem.resourceValues(forKeys: [.isDirectoryKey])
            let isDirectory = sourceValues.isDirectory == true
            let destinationItem = destinationDirectory.appendingPathComponent(sourceItem.lastPathComponent, isDirectory: isDirectory)

            if !fileManager.fileExists(atPath: destinationItem.path) {
                try fileManager.moveItem(at: sourceItem, to: destinationItem)
                continue
            }

            let destinationValues = try destinationItem.resourceValues(forKeys: [.isDirectoryKey])
            let destinationIsDirectory = destinationValues.isDirectory == true

            if isDirectory && destinationIsDirectory {
                try mergeDirectoryContents(from: sourceItem, into: destinationItem, fileManager: fileManager)
                try removeDirectoryIfEmpty(sourceItem, fileManager: fileManager)
                continue
            }

            if !isDirectory && !destinationIsDirectory, fileContentsMatch(sourceItem, destinationItem, fileManager: fileManager) {
                try fileManager.removeItem(at: sourceItem)
                continue
            }

            let uniqueDestination = uniqueDestinationURL(for: destinationItem, fileManager: fileManager)
            try fileManager.moveItem(at: sourceItem, to: uniqueDestination)
        }
    }

    private static func fileContentsMatch(_ source: URL, _ destination: URL, fileManager: FileManager) -> Bool {
        fileManager.contentsEqual(atPath: source.path, andPath: destination.path)
    }

    private static func uniqueDestinationURL(for destination: URL, fileManager: FileManager) -> URL {
        let directory = destination.deletingLastPathComponent()
        let stem = destination.deletingPathExtension().lastPathComponent
        let ext = destination.pathExtension
        var candidate = destination
        var counter = 2

        while fileManager.fileExists(atPath: candidate.path) {
            let filename = ext.isEmpty ? "\(stem)-legacy-\(counter)" : "\(stem)-legacy-\(counter).\(ext)"
            candidate = directory.appendingPathComponent(filename, isDirectory: false)
            counter += 1
        }

        return candidate
    }

    private static func removeDirectoryIfEmpty(_ directory: URL, fileManager: FileManager) throws {
        guard fileManager.fileExists(atPath: directory.path) else { return }
        let remainingItems = try fileManager.contentsOfDirectory(atPath: directory.path)
        guard remainingItems.isEmpty else { return }
        try fileManager.removeItem(at: directory)
    }
}
