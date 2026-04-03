import Foundation
import SwiftUI
import os

@MainActor
class TranscriptionModelManager: ObservableObject {
    private static let preferredDefaultModelName = "scribe_v2"

    @Published var currentTranscriptionModel: (any TranscriptionModel)?
    @Published var allAvailableModels: [any TranscriptionModel] = PredefinedModels.models

    private weak var whisperModelManager: WhisperModelManager?
    private weak var parakeetModelManager: ParakeetModelManager?
    private var cohereAvailabilityObserverTask: Task<Void, Never>?

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionModelManager")

    init(whisperModelManager: WhisperModelManager, parakeetModelManager: ParakeetModelManager) {
        self.whisperModelManager = whisperModelManager
        self.parakeetModelManager = parakeetModelManager

        // Wire up deletion callbacks so each manager notifies this manager.
        whisperModelManager.onModelDeleted = { [weak self] modelName in
            Task { @MainActor [weak self] in
                self?.handleModelDeleted(modelName)
            }
        }
        parakeetModelManager.onModelDeleted = { [weak self] modelName in
            Task { @MainActor [weak self] in
                self?.handleModelDeleted(modelName)
            }
        }

        // Wire up "models changed" callbacks so this manager rebuilds allAvailableModels.
        whisperModelManager.onModelsChanged = { [weak self] in
            Task { @MainActor [weak self] in
                self?.refreshAllAvailableModels()
            }
        }
        parakeetModelManager.onModelsChanged = { [weak self] in
            Task { @MainActor [weak self] in
                self?.refreshAllAvailableModels()
            }
        }

        cohereAvailabilityObserverTask = Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(named: .cohereTranscribeAvailabilityDidChange) {
                guard !Task.isCancelled else { break }
                await MainActor.run {
                    self?.refreshAllAvailableModels()
                }
            }
        }
    }

    // MARK: - Computed: usable models

    var usableModels: [any TranscriptionModel] {
        allAvailableModels.filter { model in
            switch model.provider {
            case .local:
                return whisperModelManager?.availableModels.contains { $0.name == model.name } ?? false
            case .parakeet:
                return parakeetModelManager?.isParakeetModelDownloaded(named: model.name) ?? false
            case .nativeApple:
                if #available(macOS 26, *) {
                    return true
                } else {
                    return false
                }
            case .localVoxtral:
                return true
            case .cohereTranscribe:
                return CohereNativeModelManager.shared.isModelDownloaded()
            case .elevenLabs:
                return APIKeyManager.shared.hasAPIKey(forProvider: "ElevenLabs")
            case .custom:
                return true
            }
        }
    }

    // MARK: - Model loading from UserDefaults

    func loadCurrentTranscriptionModel() {
        guard let savedModelName = UserDefaults.standard.string(forKey: "CurrentTranscriptionModel") else {
            selectPreferredDefaultModelIfNeeded()
            return
        }

        guard let savedModel = allAvailableModels.first(where: { $0.name == savedModelName }) else {
            UserDefaults.standard.removeObject(forKey: "CurrentTranscriptionModel")
            selectPreferredDefaultModelIfNeeded()
            return
        }

        currentTranscriptionModel = savedModel
    }

    // MARK: - Set default model

    func setDefaultTranscriptionModel(_ model: any TranscriptionModel) {
        let previousModelName = currentTranscriptionModel?.name
        self.currentTranscriptionModel = model
        UserDefaults.standard.set(model.name, forKey: "CurrentTranscriptionModel")

        if model.provider != .local {
            whisperModelManager?.loadedLocalModel = nil
            whisperModelManager?.isModelLoaded = true
        }

        postModelChange(previousModelName: previousModelName, newModelName: model.name)
    }

    // MARK: - Refresh all available models

    func refreshAllAvailableModels() {
        let currentModelName = currentTranscriptionModel?.name
        allAvailableModels = PredefinedModels.models

        if let currentName = currentModelName,
           let updatedModel = allAvailableModels.first(where: { $0.name == currentName }) {
            setDefaultTranscriptionModel(updatedModel)
        }
    }

    // MARK: - Clear current model

    func clearCurrentTranscriptionModel() {
        let previousModelName = currentTranscriptionModel?.name
        currentTranscriptionModel = nil
        UserDefaults.standard.removeObject(forKey: "CurrentTranscriptionModel")
        postModelChange(previousModelName: previousModelName, newModelName: nil)
    }

    // MARK: - Handle model deletion callback

    /// Called by WhisperModelManager.onModelDeleted or ParakeetModelManager.onModelDeleted.
    func handleModelDeleted(_ modelName: String) {
        let previousModelName = currentTranscriptionModel?.name
        if currentTranscriptionModel?.name == modelName {
            currentTranscriptionModel = nil
            UserDefaults.standard.removeObject(forKey: "CurrentTranscriptionModel")
            whisperModelManager?.loadedLocalModel = nil
            whisperModelManager?.isModelLoaded = false
            UserDefaults.standard.removeObject(forKey: "CurrentModel")
            postModelChange(previousModelName: previousModelName, newModelName: nil)
        }
        refreshAllAvailableModels()
    }

    private func postModelChange(previousModelName: String?, newModelName: String?) {
        var userInfo: [String: Any] = [:]
        if let previousModelName {
            userInfo["previousModelName"] = previousModelName
        }
        if let newModelName {
            userInfo["modelName"] = newModelName
        }

        NotificationCenter.default.post(
            name: .didChangeModel,
            object: nil,
            userInfo: userInfo.isEmpty ? nil : userInfo
        )
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }

    deinit {
        cohereAvailabilityObserverTask?.cancel()
    }

    private func selectPreferredDefaultModelIfNeeded() {
        guard currentTranscriptionModel == nil else { return }
        guard let preferredModel = usableModels.first(where: { $0.name == Self.preferredDefaultModelName }) else {
            return
        }

        setDefaultTranscriptionModel(preferredModel)
    }
}
