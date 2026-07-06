import Foundation
import FluidAudio
import AppKit
import os

@MainActor
class ParakeetModelManager: ObservableObject {
    @Published var parakeetDownloadStates: [String: Bool] = [:]
    @Published var downloadProgress: [String: Double] = [:]

    /// Called when a model is deleted, passing the model name.
    /// TranscriptionModelManager listens to clear currentTranscriptionModel if needed.
    var onModelDeleted: ((String) -> Void)?

    /// Called after a model is successfully downloaded so TranscriptionModelManager
    /// can rebuild allAvailableModels.
    var onModelsChanged: (() -> Void)?

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "ParakeetModelManager")

    init() {
        Self.migrateLegacyV2CacheIfNeeded()
    }

    // MARK: - Legacy cache migration

    /// FluidAudio renamed the v2 cache folder from "parakeet-tdt-0.6b-v2-coreml"
    /// to "parakeet-tdt-0.6b-v2" (the "-coreml" suffix is now stripped from
    /// Repo.folderName). Move an existing download across so it is not silently
    /// re-downloaded and the old ~474MB directory orphaned. Runs once per process.
    nonisolated static func migrateLegacyV2CacheIfNeeded() {
        _ = parakeetLegacyV2CacheMigration
    }

    // MARK: - Query helpers

    func isParakeetModelDownloaded(named modelName: String) -> Bool {
        UserDefaults.standard.bool(forKey: parakeetDefaultsKey(for: modelName))
    }

    func isParakeetModelDownloaded(_ model: ParakeetModel) -> Bool {
        isParakeetModelDownloaded(named: model.name)
    }

    func isParakeetModelDownloading(_ model: ParakeetModel) -> Bool {
        parakeetDownloadStates[model.name] ?? false
    }

    // MARK: - Download

    func downloadParakeetModel(_ model: ParakeetModel) async {
        if isParakeetModelDownloaded(model) {
            return
        }

        let modelName = model.name
        parakeetDownloadStates[modelName] = true
        downloadProgress[modelName] = 0.0

        let timer = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: true) { timer in
            Task { @MainActor in
                if let currentProgress = self.downloadProgress[modelName], currentProgress < 0.9 {
                    self.downloadProgress[modelName] = currentProgress + 0.005
                }
            }
        }

        let version = parakeetVersion(for: modelName)

        do {
            _ = try await AsrModels.downloadAndLoad(version: version)
            UserDefaults.standard.set(true, forKey: parakeetDefaultsKey(for: modelName))
            downloadProgress[modelName] = 1.0
        } catch {
            UserDefaults.standard.set(false, forKey: parakeetDefaultsKey(for: modelName))
            logger.error("❌ Parakeet download failed for \(modelName, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }

        timer.invalidate()
        parakeetDownloadStates[modelName] = false
        downloadProgress[modelName] = nil

        onModelsChanged?()
    }

    // MARK: - Delete

    func deleteParakeetModel(_ model: ParakeetModel) {
        let version = parakeetVersion(for: model.name)
        let cacheDirectory = parakeetCacheDirectory(for: version)

        do {
            if FileManager.default.fileExists(atPath: cacheDirectory.path) {
                try FileManager.default.removeItem(at: cacheDirectory)
            }
            UserDefaults.standard.set(false, forKey: parakeetDefaultsKey(for: model.name))
        } catch {
            // Silently ignore removal errors
        }

        // Notify TranscriptionModelManager to clear currentTranscriptionModel if it matches
        onModelDeleted?(model.name)
    }

    // MARK: - Finder

    func showParakeetModelInFinder(_ model: ParakeetModel) {
        let cacheDirectory = parakeetCacheDirectory(for: parakeetVersion(for: model.name))

        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            NSWorkspace.shared.selectFile(cacheDirectory.path, inFileViewerRootedAtPath: "")
        }
    }

    // MARK: - Private helpers

    private func parakeetDefaultsKey(for modelName: String) -> String {
        "ParakeetModelDownloaded_\(modelName)"
    }

    private func parakeetVersion(for modelName: String) -> AsrModelVersion {
        .v2
    }

    private func parakeetCacheDirectory(for version: AsrModelVersion) -> URL {
        AsrModels.defaultCacheDirectory(for: version)
    }
}

private let parakeetLegacyV2CacheMigration: Void = {
    let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "ParakeetModelManager")
    let fileManager = FileManager.default
    let currentDirectory = AsrModels.defaultCacheDirectory(for: .v2)
    let legacyDirectory = currentDirectory.deletingLastPathComponent()
        .appendingPathComponent(currentDirectory.lastPathComponent + "-coreml")

    guard fileManager.fileExists(atPath: legacyDirectory.path) else { return }

    do {
        if fileManager.fileExists(atPath: currentDirectory.path) {
            // A (possibly partial) re-download already exists; keep its files
            // and fill in the rest from the legacy download.
            for item in try fileManager.contentsOfDirectory(atPath: legacyDirectory.path) {
                let destination = currentDirectory.appendingPathComponent(item)
                if !fileManager.fileExists(atPath: destination.path) {
                    try fileManager.moveItem(
                        at: legacyDirectory.appendingPathComponent(item),
                        to: destination
                    )
                }
            }
            try fileManager.removeItem(at: legacyDirectory)
        } else {
            try fileManager.moveItem(at: legacyDirectory, to: currentDirectory)
        }
        logger.notice("Migrated legacy Parakeet v2 model cache to \(currentDirectory.path, privacy: .public)")
    } catch {
        logger.error("Failed to migrate legacy Parakeet v2 model cache: \(error.localizedDescription, privacy: .public)")
    }
}()
