import Foundation
import SwiftUI
import os

@MainActor
class TranscriptionModelManager: ObservableObject {
    private static let preferredDefaultModelNames = ["gpt-live-transcribe", "xai-stt"]

    @Published var currentTranscriptionModel: (any TranscriptionModel)?
    @Published var allAvailableModels: [any TranscriptionModel]

    private weak var whisperModelManager: WhisperModelManager?
    private var cohereAvailabilityObserverTask: Task<Void, Never>?
    private let userDefaults: UserDefaults
    private let hasAPIKey: (String) -> Bool
    private let availableModels: () -> [any TranscriptionModel]

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "TranscriptionModelManager")

    init(
        whisperModelManager: WhisperModelManager,
        userDefaults: UserDefaults = .standard,
        hasAPIKey: @escaping (String) -> Bool = { provider in
            APIKeyManager.shared.hasAPIKey(forProvider: provider)
        },
        availableModels: @escaping () -> [any TranscriptionModel] = { PredefinedModels.models }
    ) {
        self.whisperModelManager = whisperModelManager
        self.userDefaults = userDefaults
        self.hasAPIKey = hasAPIKey
        self.availableModels = availableModels
        self.allAvailableModels = availableModels()

        // Wire up deletion callbacks so each manager notifies this manager.
        whisperModelManager.onModelDeleted = { [weak self] modelName in
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
            case .openAI, .xAI:
                return hasAPIKey(model.provider.apiKeyProviderName)
            case .custom:
                return true
            }
        }
    }

    // MARK: - Model loading from UserDefaults

    func loadCurrentTranscriptionModel() {
        guard let savedModelName = userDefaults.string(forKey: "CurrentTranscriptionModel") else {
            selectPreferredDefaultModelIfNeeded()
            return
        }

        guard let savedModel = allAvailableModels.first(where: { $0.name == savedModelName }) else {
            currentTranscriptionModel = nil
            userDefaults.removeObject(forKey: "CurrentTranscriptionModel")
            selectPreferredDefaultModelIfNeeded()
            return
        }

        currentTranscriptionModel = savedModel
    }

    // MARK: - Set default model

    func setDefaultTranscriptionModel(_ model: any TranscriptionModel) {
        setDefaultTranscriptionModel(
            model,
            previousModelName: currentTranscriptionModel?.name
        )
    }

    private func setDefaultTranscriptionModel(
        _ model: any TranscriptionModel,
        previousModelName: String?
    ) {
        self.currentTranscriptionModel = model
        userDefaults.set(model.name, forKey: "CurrentTranscriptionModel")

        if model.provider != .local {
            whisperModelManager?.loadedLocalModel = nil
            whisperModelManager?.isModelLoaded = true
        }

        postModelChange(previousModelName: previousModelName, newModelName: model.name)
    }

    // MARK: - Refresh all available models

    func refreshAllAvailableModels() {
        let currentModelName = currentTranscriptionModel?.name
        allAvailableModels = availableModels()

        if let currentName = currentModelName,
           let updatedModel = allAvailableModels.first(where: { $0.name == currentName }) {
            setDefaultTranscriptionModel(updatedModel, previousModelName: currentName)
        } else if let currentModelName {
            if let fallbackModel = preferredDefaultModel() {
                setDefaultTranscriptionModel(
                    fallbackModel,
                    previousModelName: currentModelName
                )
            } else {
                currentTranscriptionModel = nil
                userDefaults.removeObject(forKey: "CurrentTranscriptionModel")
                postModelChange(previousModelName: currentModelName, newModelName: nil)
            }
        }
    }

    // MARK: - Clear current model

    func clearCurrentTranscriptionModel() {
        let previousModelName = currentTranscriptionModel?.name
        currentTranscriptionModel = nil
        userDefaults.removeObject(forKey: "CurrentTranscriptionModel")
        postModelChange(previousModelName: previousModelName, newModelName: nil)
    }

    // MARK: - Handle model deletion callback

    /// Called by WhisperModelManager.onModelDeleted.
    func handleModelDeleted(_ modelName: String) {
        let previousModelName = currentTranscriptionModel?.name
        if currentTranscriptionModel?.name == modelName {
            currentTranscriptionModel = nil
            userDefaults.removeObject(forKey: "CurrentTranscriptionModel")
            whisperModelManager?.loadedLocalModel = nil
            whisperModelManager?.isModelLoaded = false
            userDefaults.removeObject(forKey: "CurrentModel")
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

        if let preferredModel = preferredDefaultModel() {
            setDefaultTranscriptionModel(preferredModel)
        }
    }

    private func preferredDefaultModel() -> (any TranscriptionModel)? {
        for preferredName in Self.preferredDefaultModelNames {
            if let preferredModel = usableModels.first(where: { $0.name == preferredName }) {
                return preferredModel
            }
        }
        return nil
    }
}
