import Foundation
import os

@MainActor
final class VoxtralNativeModelManager: ObservableObject {
    static let shared = VoxtralNativeModelManager()

    enum DownloadState: Equatable {
        case idle
        case downloading
        case completed
        case failed(String)

        var errorMessage: String? {
            if case .failed(let message) = self {
                return message
            }
            return nil
        }
    }

    @Published private(set) var downloadStates: [String: DownloadState] = [:]
    /// Fractional progress (0.0-1.0) of the single file currently being
    /// fetched. Reset to 0 at the start of each file so large shards don't
    /// appear to jump backwards when smaller siblings finish.
    @Published private(set) var downloadProgress: [String: Double] = [:]

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "VoxtralNativeModelManager")
    private let urlSession = URLSession(configuration: .default)
    private var hasMigratedLegacyStorage = false

    private init() {}

    func availability(for modelReference: String) -> VoxtralNativeModelLocator.Availability {
        migrateLegacyStorageIfNeeded()
        return VoxtralNativeModelLocator.availability(for: modelReference)
    }

    func downloadState(for modelReference: String) -> DownloadState {
        downloadStates[modelReference] ?? .idle
    }

    func isDownloading(_ modelReference: String) -> Bool {
        downloadState(for: modelReference) == .downloading
    }

    func downloadModelIfNeeded(_ modelReference: String) async {
        guard !isDownloading(modelReference) else { return }
        switch availability(for: modelReference) {
        case .appManaged, .externalLocalPath:
            downloadStates[modelReference] = .completed
            return
        case .missing:
            break
        }

        guard let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
            downloadStates[modelReference] = .failed("Only Hugging Face repository IDs can be downloaded automatically.")
            return
        }

        downloadStates[modelReference] = .downloading
        downloadProgress[modelReference] = 0

        do {
            _ = try await materializeRepository(repoID, progressKey: modelReference)
            downloadStates[modelReference] = .completed
            downloadProgress[modelReference] = 1
        } catch {
            let message = error.localizedDescription
            logger.error("Native Voxtral asset download failed: \(message, privacy: .public)")
            downloadStates[modelReference] = .failed(message)
            downloadProgress[modelReference] = nil
        }
    }

    func preparedModelDirectory(for modelReference: String, autoDownload: Bool = false) async throws -> URL {
        switch availability(for: modelReference) {
        case .appManaged(let url), .externalLocalPath(let url):
            return url
        case .missing:
            guard autoDownload, let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
                throw VoxtralNativeModelError.missingModelAssets
            }
            return try await materializeRepository(repoID, progressKey: modelReference)
        }
    }

    func deleteModelAssets(for modelReference: String) async {
        guard case .appManaged(let directoryURL) = availability(for: modelReference) else { return }

        _ = await VoxtralNativeRuntime.shared.unloadPreparedState(modelReference)

        do {
            try FileManager.default.removeItem(at: directoryURL)
            downloadStates[modelReference] = .idle
        } catch {
            logger.error("Failed to delete native Voxtral assets: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func materializeRepository(_ repoID: String, progressKey: String? = nil) async throws -> URL {
        migrateLegacyStorageIfNeeded()

        let destinationDirectory = VoxtralNativeModelLocator.appManagedModelDirectory(for: repoID)
        if VoxtralNativeModelLocator.isModelDirectoryComplete(destinationDirectory) {
            return destinationDirectory
        }

        return try await downloadRepository(repoID, progressKey: progressKey)
    }

    private func downloadRepository(_ repoID: String, progressKey: String? = nil) async throws -> URL {
        let fileManager = FileManager.default
        let destinationDirectory = VoxtralNativeModelLocator.appManagedModelDirectory(for: repoID)
        let temporaryDirectory = destinationDirectory.appendingPathExtension("download")

        try fileManager.createDirectory(
            at: temporaryDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        if fileManager.fileExists(atPath: temporaryDirectory.path) {
            try fileManager.removeItem(at: temporaryDirectory)
        }
        try fileManager.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true,
            attributes: nil
        )

        do {
            try await downloadRequiredFiles(for: repoID, into: temporaryDirectory, progressKey: progressKey)

            if fileManager.fileExists(atPath: destinationDirectory.path) {
                try fileManager.removeItem(at: destinationDirectory)
            }
            try fileManager.moveItem(at: temporaryDirectory, to: destinationDirectory)
            return destinationDirectory
        } catch {
            try? fileManager.removeItem(at: temporaryDirectory)
            throw error
        }
    }

    private func migrateLegacyStorageIfNeeded() {
        guard !hasMigratedLegacyStorage else { return }

        do {
            try migrateLegacyManagedModelsDirectoryIfNeeded()
            hasMigratedLegacyStorage = true
        } catch {
            logger.error("Native Voxtral storage migration failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func migrateLegacyManagedModelsDirectoryIfNeeded() throws {
        let fileManager = FileManager.default
        let legacyRoot = legacyVoxtralModelsDirectory
        guard fileManager.fileExists(atPath: legacyRoot.path) else { return }

        try fileManager.createDirectory(
            at: AppStoragePaths.voxtralModelsDirectory,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let legacyDirectories = try fileManager.contentsOfDirectory(
            at: legacyRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for legacyDirectory in legacyDirectories {
            let resourceValues = try legacyDirectory.resourceValues(forKeys: [.isDirectoryKey])
            guard resourceValues.isDirectory == true else { continue }
            guard VoxtralNativeModelLocator.isModelDirectoryComplete(legacyDirectory) else { continue }

            let destinationDirectory = AppStoragePaths.voxtralModelsDirectory
                .appendingPathComponent(legacyDirectory.lastPathComponent, isDirectory: true)

            if VoxtralNativeModelLocator.isModelDirectoryComplete(destinationDirectory) {
                try? fileManager.removeItem(at: legacyDirectory)
                continue
            }

            if fileManager.fileExists(atPath: destinationDirectory.path) {
                try fileManager.removeItem(at: destinationDirectory)
            }

            try fileManager.moveItem(at: legacyDirectory, to: destinationDirectory)
        }

        try removeDirectoryIfEmpty(legacyRoot, fileManager: fileManager)
        try removeDirectoryIfEmpty(legacyRoot.deletingLastPathComponent(), fileManager: fileManager)
        try removeDirectoryIfEmpty(legacyRoot.deletingLastPathComponent().deletingLastPathComponent(), fileManager: fileManager)
    }

    private var legacyVoxtralModelsDirectory: URL {
        return AppStoragePaths.applicationSupportDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("VoiceInk", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("Voxtral", isDirectory: true)
    }

    private func removeDirectoryIfEmpty(_ directory: URL, fileManager: FileManager) throws {
        guard fileManager.fileExists(atPath: directory.path) else { return }
        let remainingItems = try fileManager.contentsOfDirectory(atPath: directory.path)
        guard remainingItems.isEmpty else { return }
        try fileManager.removeItem(at: directory)
    }

    private func downloadRequiredFiles(for repoID: String, into directory: URL, progressKey: String? = nil) async throws {
        let baseFiles = ["config.json", "tekken.json"]
        for filename in baseFiles {
            try await downloadFile(named: filename, from: repoID, into: directory, progressKey: progressKey)
        }

        let indexWasDownloaded = try await downloadOptionalFile(named: "model.safetensors.index.json", from: repoID, into: directory, progressKey: progressKey)
        let shardFiles: [String]
        if indexWasDownloaded,
           let indexData = try? Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json")),
           let index = try? JSONDecoder().decode(VoxtralModelDownloadIndex.self, from: indexData) {
            shardFiles = Array(Set(index.weightMap.values)).sorted()
        } else {
            shardFiles = ["model.safetensors"]
        }

        for filename in shardFiles {
            try await downloadFile(named: filename, from: repoID, into: directory, progressKey: progressKey)
        }
    }

    private func downloadOptionalFile(named filename: String, from repoID: String, into directory: URL, progressKey: String? = nil) async throws -> Bool {
        do {
            try await downloadFile(named: filename, from: repoID, into: directory, progressKey: progressKey)
            return true
        } catch let error as VoxtralNativeModelError where error == .fileNotFound(filename) {
            return false
        }
    }

    private func downloadFile(named filename: String, from repoID: String, into directory: URL, progressKey: String? = nil) async throws {
        let destinationURL = directory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return
        }

        guard let remoteURL = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(filename)") else {
            throw VoxtralNativeModelError.invalidRepository
        }

        var request = URLRequest(url: remoteURL)
        request.timeoutInterval = 600
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        if let progressKey {
            await MainActor.run { self.downloadProgress[progressKey] = 0 }
        }

        let delegate = progressKey.map { key in
            ModelDownloadProgressDelegate { fraction in
                Task { @MainActor in
                    VoxtralNativeModelManager.shared.downloadProgress[key] = fraction
                }
            }
        }

        let (temporaryURL, response) = try await urlSession.download(for: request, delegate: delegate)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw VoxtralNativeModelError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if httpResponse.statusCode == 404 {
                throw VoxtralNativeModelError.fileNotFound(filename)
            }
            throw VoxtralNativeModelError.httpError(httpResponse.statusCode)
        }

        try FileManager.default.moveItem(at: temporaryURL, to: destinationURL)
    }
}

enum VoxtralNativeModelError: LocalizedError, Equatable {
    case invalidRepository
    case invalidResponse
    case httpError(Int)
    case fileNotFound(String)
    case missingModelAssets

    var errorDescription: String? {
        switch self {
        case .invalidRepository:
            return "Invalid Voxtral repository ID."
        case .invalidResponse:
            return "The download server returned an invalid response."
        case .httpError(let statusCode):
            return "Model download failed with HTTP \(statusCode)."
        case .fileNotFound(let filename):
            return "The model file \(filename) was not found."
        case .missingModelAssets:
            return "Local Voxtral model assets are not available yet."
        }
    }
}

private struct VoxtralModelDownloadIndex: Decodable {
    let weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
