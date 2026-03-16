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

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "VoxtralNativeModelManager")
    private let urlSession = URLSession(configuration: .default)

    private init() {}

    func availability(for modelReference: String) -> VoxtralNativeModelLocator.Availability {
        VoxtralNativeModelLocator.availability(for: modelReference)
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
        case .sharedCache, .missing:
            break
        }

        guard let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
            downloadStates[modelReference] = .failed("Only Hugging Face repository IDs can be downloaded automatically.")
            return
        }

        downloadStates[modelReference] = .downloading

        do {
            _ = try await materializeRepository(repoID)
            downloadStates[modelReference] = .completed
        } catch {
            let message = error.localizedDescription
            logger.error("Native Voxtral asset download failed: \(message, privacy: .public)")
            downloadStates[modelReference] = .failed(message)
        }
    }

    func preparedModelDirectory(for modelReference: String, autoDownload: Bool = false) async throws -> URL {
        switch availability(for: modelReference) {
        case .appManaged(let url), .externalLocalPath(let url):
            return url
        case .sharedCache:
            guard let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
                throw VoxtralNativeModelError.missingModelAssets
            }
            return try await materializeRepository(repoID)
        case .missing:
            guard autoDownload, let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
                throw VoxtralNativeModelError.missingModelAssets
            }
            return try await materializeRepository(repoID)
        }
    }

    func migrateSharedCacheToManagedCopyIfNeeded(_ modelReference: String) async {
        guard !isDownloading(modelReference) else { return }
        guard case .sharedCache = availability(for: modelReference) else { return }
        guard let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else { return }

        downloadStates[modelReference] = .downloading

        do {
            _ = try await materializeRepository(repoID)
            downloadStates[modelReference] = .completed
        } catch {
            let message = error.localizedDescription
            logger.error("Native Voxtral cache migration failed: \(message, privacy: .public)")
            downloadStates[modelReference] = .failed(message)
        }
    }

    private func materializeRepository(_ repoID: String) async throws -> URL {
        let destinationDirectory = VoxtralNativeModelLocator.appManagedModelDirectory(for: repoID)
        if VoxtralNativeModelLocator.isModelDirectoryComplete(destinationDirectory) {
            return destinationDirectory
        }

        if let sharedCache = VoxtralNativeModelLocator.sharedCacheSnapshot(for: repoID),
           VoxtralNativeModelLocator.isModelDirectoryComplete(sharedCache) {
            return try adoptSharedCacheSnapshot(sharedCache, for: repoID)
        }

        return try await downloadRepository(repoID)
    }

    private func downloadRepository(_ repoID: String) async throws -> URL {
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
            try await downloadRequiredFiles(for: repoID, into: temporaryDirectory)

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

    private func adoptSharedCacheSnapshot(_ sourceDirectory: URL, for repoID: String) throws -> URL {
        let fileManager = FileManager.default
        let destinationDirectory = VoxtralNativeModelLocator.appManagedModelDirectory(for: repoID)
        let temporaryDirectory = destinationDirectory.appendingPathExtension("import")

        try fileManager.createDirectory(
            at: destinationDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )

        if fileManager.fileExists(atPath: temporaryDirectory.path) {
            try fileManager.removeItem(at: temporaryDirectory)
        }
        if fileManager.fileExists(atPath: destinationDirectory.path) {
            try fileManager.removeItem(at: destinationDirectory)
        }

        try fileManager.copyItem(at: sourceDirectory, to: temporaryDirectory)
        try fileManager.moveItem(at: temporaryDirectory, to: destinationDirectory)

        let sharedCacheDirectory = VoxtralNativeModelLocator.huggingFaceCacheDirectory(for: repoID)
        if fileManager.fileExists(atPath: sharedCacheDirectory.path) {
            try? fileManager.removeItem(at: sharedCacheDirectory)
        }

        return destinationDirectory
    }

    private func downloadRequiredFiles(for repoID: String, into directory: URL) async throws {
        let baseFiles = ["config.json", "tekken.json"]
        for filename in baseFiles {
            try await downloadFile(named: filename, from: repoID, into: directory)
        }

        let indexWasDownloaded = try await downloadOptionalFile(named: "model.safetensors.index.json", from: repoID, into: directory)
        let shardFiles: [String]
        if indexWasDownloaded,
           let indexData = try? Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json")),
           let index = try? JSONDecoder().decode(VoxtralModelDownloadIndex.self, from: indexData) {
            shardFiles = Array(Set(index.weightMap.values)).sorted()
        } else {
            shardFiles = ["model.safetensors"]
        }

        for filename in shardFiles {
            try await downloadFile(named: filename, from: repoID, into: directory)
        }
    }

    private func downloadOptionalFile(named filename: String, from repoID: String, into directory: URL) async throws -> Bool {
        do {
            try await downloadFile(named: filename, from: repoID, into: directory)
            return true
        } catch let error as VoxtralNativeModelError where error == .fileNotFound(filename) {
            return false
        }
    }

    private func downloadFile(named filename: String, from repoID: String, into directory: URL) async throws {
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

        let (temporaryURL, response) = try await urlSession.download(for: request)
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
