import Foundation
import os

@MainActor
final class CohereNativeModelManager: ObservableObject {
    static let shared = CohereNativeModelManager()
    private static let obsoleteHuggingFaceProviderName = "HuggingFace"

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

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "CohereNativeModelManager")
    private let urlSession = URLSession(configuration: .default)

    private init() {
        purgeObsoleteManagedArtifactsIfNeeded()
        syncDownloadState()
    }

    func availability(for modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository) -> CohereNativeModelLocator.Availability {
        CohereNativeModelLocator.availability(for: modelReference)
    }

    func isModelDownloaded(for modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository) -> Bool {
        switch availability(for: modelReference) {
        case .appManaged, .externalLocalPath:
            return true
        case .missing:
            return false
        }
    }

    func downloadState(for modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository) -> DownloadState {
        if isModelDownloaded(for: modelReference) {
            return .completed
        }

        return downloadStates[modelReference] ?? .idle
    }

    func downloadModelIfNeeded(_ modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository) async {
        guard downloadState(for: modelReference) != .downloading else { return }

        switch availability(for: modelReference) {
        case .appManaged, .externalLocalPath:
            downloadStates[modelReference] = .completed
            postAvailabilityDidChange()
            return
        case .missing:
            break
        }

        guard let repositoryID = CohereNativeModelLocator.repositoryID(from: modelReference) else {
            downloadStates[modelReference] = .failed("Only Hugging Face repository IDs can be downloaded automatically.")
            return
        }

        downloadStates[modelReference] = .downloading

        do {
            _ = try await materializeRepository(repositoryID)
            downloadStates[modelReference] = .completed
            postAvailabilityDidChange()
        } catch {
            let message = error.localizedDescription
            logger.error("Native Cohere MLX asset download failed: \(message, privacy: .public)")
            downloadStates[modelReference] = .failed(message)
        }
    }

    func preparedModelDirectory(
        for modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository,
        autoDownload: Bool = false
    ) async throws -> URL {
        switch availability(for: modelReference) {
        case .appManaged(let url), .externalLocalPath(let url):
            return url
        case .missing:
            guard autoDownload,
                  let repositoryID = CohereNativeModelLocator.repositoryID(from: modelReference) else {
                throw CohereNativeModelError.missingModelAssets
            }
            return try await materializeRepository(repositoryID)
        }
    }

    func deleteManagedAssets(for modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository) async {
        await CohereNativeRuntime.shared.clearPreparedBootstraps()

        do {
            if case .appManaged(let directoryURL) = availability(for: modelReference),
               FileManager.default.fileExists(atPath: directoryURL.path) {
                try FileManager.default.removeItem(at: directoryURL)
            }

            purgeObsoleteManagedArtifactsIfNeeded()
            downloadStates[modelReference] = .idle
            postAvailabilityDidChange()
        } catch {
            logger.error("Failed to delete native Cohere MLX assets: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func materializeRepository(_ repositoryID: String) async throws -> URL {
        let destinationDirectory = CohereNativeModelLocator.appManagedModelDirectory(for: repositoryID)
        if CohereNativeModelLocator.isModelDirectoryComplete(destinationDirectory) {
            return destinationDirectory
        }

        return try await downloadRepository(repositoryID)
    }

    private func downloadRepository(_ repositoryID: String) async throws -> URL {
        let fileManager = FileManager.default
        let destinationDirectory = CohereNativeModelLocator.appManagedModelDirectory(for: repositoryID)
        let temporaryDirectory = destinationDirectory.appendingPathExtension("download")

        try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.cohereTranscribeNativeDirectory)
        if fileManager.fileExists(atPath: temporaryDirectory.path) {
            try fileManager.removeItem(at: temporaryDirectory)
        }
        try fileManager.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)

        do {
            try await downloadRequiredFiles(for: repositoryID, into: temporaryDirectory)
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

    private func downloadRequiredFiles(for repositoryID: String, into directory: URL) async throws {
        let requiredFiles = [
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "model.safetensors"
        ]

        for filename in requiredFiles {
            try await downloadFile(named: filename, from: repositoryID, into: directory)
        }

        let optionalFiles = [
            "conversion_summary.json",
            "key_map.json"
        ]

        for filename in optionalFiles {
            _ = try? await downloadFile(named: filename, from: repositoryID, into: directory)
        }
    }

    @discardableResult
    private func downloadFile(named filename: String, from repositoryID: String, into directory: URL) async throws -> URL {
        let destinationURL = directory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return destinationURL
        }

        let revision = LocalCohereTranscribeConfiguration.nativeModelRevision
        guard let remoteURL = URL(string: "https://huggingface.co/\(repositoryID)/resolve/\(revision)/\(filename)") else {
            throw CohereNativeModelError.invalidRepository
        }

        var request = URLRequest(url: remoteURL)
        request.timeoutInterval = 600
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        let (temporaryURL, response) = try await urlSession.download(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CohereNativeModelError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if httpResponse.statusCode == 404 {
                throw CohereNativeModelError.fileNotFound(filename)
            }
            throw CohereNativeModelError.httpError(httpResponse.statusCode)
        }

        try FileManager.default.moveItem(at: temporaryURL, to: destinationURL)
        return destinationURL
    }

    private func syncDownloadState(for modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository) {
        if isModelDownloaded(for: modelReference) {
            downloadStates[modelReference] = .completed
        } else if downloadStates[modelReference] != .downloading {
            downloadStates[modelReference] = .idle
        }
    }

    private func purgeObsoleteManagedArtifactsIfNeeded() {
        let fileManager = FileManager.default
        var obsoleteDirectories = [
            AppStoragePaths.cachesDirectory.appendingPathComponent("CohereTranscribe", isDirectory: true),
            AppStoragePaths.cohereTranscribeDirectory.appendingPathComponent("HuggingFace", isDirectory: true),
            AppStoragePaths.cohereTranscribeDirectory.appendingPathComponent("Runtime", isDirectory: true)
        ]

        // If the user previously downloaded the 4.1 GB fp16 build but is now
        // on the 4-bit default, reclaim the disk by removing the old artifacts.
        // Users who pinned the fp16 repo via UserDefaults will skip this branch.
        let activeRepository = LocalCohereTranscribeConfiguration.nativeModelRepository
        let legacyFP16Repository = "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
        if activeRepository != legacyFP16Repository {
            obsoleteDirectories.append(
                CohereNativeModelLocator.appManagedModelDirectory(for: legacyFP16Repository)
            )
        }

        for directory in obsoleteDirectories where fileManager.fileExists(atPath: directory.path) {
            do {
                try fileManager.removeItem(at: directory)
            } catch {
                logger.error("Failed to remove obsolete Cohere asset at \(directory.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }

        _ = APIKeyManager.shared.deleteAPIKey(forProvider: Self.obsoleteHuggingFaceProviderName)
    }

    private func postAvailabilityDidChange() {
        NotificationCenter.default.post(name: .cohereTranscribeAvailabilityDidChange, object: nil)
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }
}

enum CohereNativeModelError: LocalizedError, Equatable {
    case invalidRepository
    case invalidResponse
    case httpError(Int)
    case fileNotFound(String)
    case missingModelAssets

    var errorDescription: String? {
        switch self {
        case .invalidRepository:
            return "Invalid Cohere MLX repository ID."
        case .invalidResponse:
            return "The native Cohere MLX download server returned an invalid response."
        case .httpError(let statusCode):
            return "Native Cohere MLX download failed with HTTP \(statusCode)."
        case .fileNotFound(let filename):
            return "The native Cohere MLX file \(filename) was not found."
        case .missingModelAssets:
            return "Cohere Transcribe MLX has not been downloaded yet."
        }
    }
}
