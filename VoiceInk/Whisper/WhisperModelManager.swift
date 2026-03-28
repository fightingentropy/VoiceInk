import Foundation
import os
import SwiftUI
@preconcurrency import WhisperKit

struct WhisperModel: Identifiable {
    let id = UUID()
    let name: String
    let url: URL
}

private final class WhisperKitDownloadProgressRelay: @unchecked Sendable {
    private weak var manager: WhisperModelManager?
    private let progressKey: String

    init(manager: WhisperModelManager, progressKey: String) {
        self.manager = manager
        self.progressKey = progressKey
    }

    func update(_ progress: Progress) {
        let fractionCompleted = progress.fractionCompleted * 0.85
        Task { @MainActor [weak manager, progressKey] in
            manager?.downloadProgress[progressKey] = fractionCompleted
        }
    }
}

private enum WhisperKitDownloadClient {
    nonisolated static func download(
        variant: String,
        downloadBase: URL,
        relay: WhisperKitDownloadProgressRelay?
    ) async throws -> URL {
        try await WhisperKit.download(
            variant: variant,
            downloadBase: downloadBase,
            useBackgroundSession: false,
            progressCallback: relay.map { relay in
                { progress in relay.update(progress) }
            }
        )
    }
}

@MainActor
class WhisperModelManager: ObservableObject {
    @Published var availableModels: [WhisperModel] = []
    @Published var downloadProgress: [String: Double] = [:]
    @Published var whisperKitRuntime: WhisperKitRuntime?
    @Published var isModelLoaded = false
    @Published var loadedLocalModel: WhisperModel?
    @Published var isModelLoading = false

    let whisperPrompt = WhisperPrompt()

    var onModelDeleted: ((String) -> Void)?
    var onModelsChanged: (() -> Void)?

    let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "WhisperModelManager")

    func createModelsDirectoryIfNeeded() {
        do {
            try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.whisperKitModelsDirectory)
        } catch {
            logError("Error creating WhisperKit models directory", error)
        }
    }

    func loadAvailableModels() {
        availableModels = discoverManagedWhisperKitModels()
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    private func discoverManagedWhisperKitModels() -> [WhisperModel] {
        PredefinedModels.models
            .compactMap { $0 as? LocalModel }
            .compactMap { localModel in
                guard let modelURL = locateWhisperKitModelDirectory(matching: localModel.whisperKitVariant) else {
                    return nil
                }
                return WhisperModel(name: localModel.name, url: modelURL)
            }
    }

    private func locateWhisperKitModelDirectory(matching variant: String) -> URL? {
        let baseDirectory = AppStoragePaths.whisperKitModelsDirectory
        guard FileManager.default.fileExists(atPath: baseDirectory.path) else {
            return nil
        }

        let enumerator = FileManager.default.enumerator(
            at: baseDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        while let next = enumerator?.nextObject() as? URL {
            guard next.lastPathComponent == variant else { continue }
            let values = try? next.resourceValues(forKeys: [.isDirectoryKey])
            if values?.isDirectory == true {
                return next
            }
        }

        return nil
    }

    func loadModel(_ model: WhisperModel) async throws {
        if loadedLocalModel?.name == model.name, isModelLoaded {
            return
        }

        isModelLoading = true
        defer { isModelLoading = false }

        do {
            await cleanupResources()
            whisperKitRuntime = try await WhisperKitRuntime(modelFolder: model.url.path)
            isModelLoaded = true
            loadedLocalModel = model
        } catch {
            whisperKitRuntime = nil
            loadedLocalModel = nil
            isModelLoaded = false
            throw VoiceInkEngineError.modelLoadFailed
        }
    }

    func downloadModel(_ model: LocalModel) async {
        do {
            let progressKey = model.name + "_main"
            try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.whisperKitModelsDirectory)

            let progressRelay = WhisperKitDownloadProgressRelay(manager: self, progressKey: progressKey)
            let modelDirectory = try await WhisperKitDownloadClient.download(
                variant: model.whisperKitVariant,
                downloadBase: AppStoragePaths.whisperKitModelsDirectory,
                relay: progressRelay
            )

            downloadProgress[progressKey] = 0.9
            try await WhisperKitRuntime.prewarmModel(at: modelDirectory.path)
            downloadProgress[progressKey] = 1.0

            let downloadedModel = WhisperModel(name: model.name, url: modelDirectory)
            availableModels.removeAll { $0.name == downloadedModel.name }
            availableModels.append(downloadedModel)
            downloadProgress.removeValue(forKey: progressKey)
            onModelsChanged?()
        } catch {
            downloadProgress.removeValue(forKey: model.name + "_main")
            logError("Failed to download WhisperKit model \(model.name)", error)
        }
    }

    func deleteModel(_ model: WhisperModel) async {
        do {
            try FileManager.default.removeItem(at: model.url)
            pruneEmptyDirectoryChain(
                startingAt: model.url.deletingLastPathComponent(),
                stopAt: AppStoragePaths.whisperKitModelsDirectory
            )

            availableModels.removeAll { $0.id == model.id }
            if loadedLocalModel?.name == model.name {
                await cleanupResources()
            }
            onModelDeleted?(model.name)
        } catch {
            logError("Error deleting model: \(model.name)", error)
        }
    }

    func clearDownloadedModels() async {
        await cleanupResources()
        try? FileManager.default.removeItem(at: AppStoragePaths.whisperKitModelsDirectory)
        availableModels.removeAll()
    }

    func cleanupResources() async {
        logger.notice("WhisperModelManager.cleanupResources: releasing WhisperKit runtime")
        await whisperKitRuntime?.unload()
        whisperKitRuntime = nil
        loadedLocalModel = nil
        isModelLoaded = false
        logger.notice("WhisperModelManager.cleanupResources: completed")
    }

    private func logError(_ message: String, _ error: Error) {
        logger.error("❌ \(message, privacy: .public): \(error.localizedDescription, privacy: .public)")
    }

    private func pruneEmptyDirectoryChain(startingAt url: URL, stopAt stopURL: URL) {
        var currentURL = url

        while currentURL.path.hasPrefix(stopURL.path), currentURL != stopURL {
            let isEmpty = (try? FileManager.default.contentsOfDirectory(
                at: currentURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ).isEmpty) ?? false

            guard isEmpty else { return }
            try? FileManager.default.removeItem(at: currentURL)
            currentURL.deleteLastPathComponent()
        }
    }

}

extension WhisperModelManager: LocalModelProvider {}

struct DownloadProgressView: View {
    let modelName: String
    let downloadProgress: [String: Double]

    private var totalProgress: Double {
        downloadProgress[modelName + "_main"] ?? 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Downloading \(modelName)")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(Color(.secondaryLabelColor))

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.separatorColor).opacity(0.3))
                        .frame(height: 6)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.controlAccentColor))
                        .frame(width: max(0, min(geometry.size.width * totalProgress, geometry.size.width)), height: 6)
                }
            }
            .frame(height: 6)

            HStack {
                Spacer()
                Text("\(Int(totalProgress * 100))%")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(Color(.secondaryLabelColor))
            }
        }
        .padding(.vertical, 4)
        .animation(.smooth, value: totalProgress)
    }
}
