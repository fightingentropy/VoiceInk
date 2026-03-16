import AppKit
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
        guard case .missing = availability(for: modelReference) else {
            downloadStates[modelReference] = .completed
            return
        }

        guard let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
            downloadStates[modelReference] = .failed("Only Hugging Face repository IDs can be downloaded automatically.")
            return
        }

        downloadStates[modelReference] = .downloading

        do {
            _ = try await downloadRepository(repoID)
            downloadStates[modelReference] = .completed
        } catch {
            let message = error.localizedDescription
            logger.error("Native Voxtral asset download failed: \(message, privacy: .public)")
            downloadStates[modelReference] = .failed(message)
        }
    }

    func deleteAppManagedModel(_ modelReference: String) {
        guard let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else { return }

        let directory = VoxtralNativeModelLocator.appManagedModelDirectory(for: repoID)
        do {
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
            downloadStates[modelReference] = .idle
        } catch {
            downloadStates[modelReference] = .failed(error.localizedDescription)
        }
    }

    func showModelInFinder(_ modelReference: String) {
        guard let directory = availability(for: modelReference).directoryURL else { return }
        NSWorkspace.shared.selectFile(directory.path, inFileViewerRootedAtPath: "")
    }

    func preparedModelDirectory(for modelReference: String, autoDownload: Bool = false) async throws -> URL {
        switch availability(for: modelReference) {
        case .appManaged(let url), .sharedCache(let url), .externalLocalPath(let url):
            return url
        case .missing:
            guard autoDownload, let repoID = VoxtralNativeModelLocator.repositoryID(from: modelReference) else {
                throw VoxtralNativeModelError.missingModelAssets
            }
            return try await downloadRepository(repoID)
        }
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
