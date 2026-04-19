import SwiftUI
import SwiftData

struct ModelManagementView: View {
    @EnvironmentObject private var whisperModelManager: WhisperModelManager
    @EnvironmentObject private var parakeetModelManager: ParakeetModelManager
    @EnvironmentObject private var transcriptionModelManager: TranscriptionModelManager
    @State private var customModelToEdit: CustomCloudModel?
    @StateObject private var customModelManager = CustomModelManager.shared
    @StateObject private var whisperPrompt = WhisperPrompt()

    @State private var isShowingSettings = false
    
    // State for the unified alert
    @State private var isShowingDeleteAlert = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    @State private var deleteActionClosure: () -> Void = {}

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                defaultModelSection
                languageSelectionSection
                availableModelsSection
            }
            .padding(40)
        }
        .frame(minWidth: 600, minHeight: 500)
        .background(Color(NSColor.controlBackgroundColor))
        .alert(isPresented: $isShowingDeleteAlert) {
            Alert(
                title: Text(alertTitle),
                message: Text(alertMessage),
                primaryButton: .destructive(Text("Delete"), action: deleteActionClosure),
                secondaryButton: .cancel()
            )
        }
    }
    
    private var defaultModelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Default Model")
                .font(.headline)
                .foregroundColor(.secondary)
            Text(transcriptionModelManager.currentTranscriptionModel?.displayName ?? "No model selected")
                .font(.title2)
                .fontWeight(.bold)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CardBackground(isSelected: false))
        .cornerRadius(10)
    }

    private var languageSelectionSection: some View {
        LanguageSelectionView(transcriptionModelManager: transcriptionModelManager, displayMode: .full, whisperPrompt: whisperPrompt)
    }
    
    private var availableModelsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Models")
                    .font(.headline)
                
                Spacer()
                
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isShowingSettings.toggle()
                    }
                }) {
                    Image(systemName: "gear")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(isShowingSettings ? .accentColor : .primary.opacity(0.7))
                        .padding(12)
                        .background(
                            CardBackground(isSelected: isShowingSettings, cornerRadius: 22)
                        )
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding(.bottom, 12)
            
            if isShowingSettings {
                ModelSettingsView(whisperPrompt: whisperPrompt)
            } else {
                VStack(spacing: 12) {
                    ForEach(displayedModels, id: \.id) { model in
                        ModelCardRowView(
                            model: model,
                            parakeetModelManager: parakeetModelManager,
                            transcriptionModelManager: transcriptionModelManager,
                            isDownloaded: whisperModelManager.availableModels.contains { $0.name == model.name },
                            isCurrent: transcriptionModelManager.currentTranscriptionModel?.name == model.name,
                            downloadProgress: whisperModelManager.downloadProgress,
                            modelURL: whisperModelManager.availableModels.first { $0.name == model.name }?.url,
                            deleteAction: {
                                if let customModel = model as? CustomCloudModel {
                                    alertTitle = "Delete Custom Model"
                                    alertMessage = "Are you sure you want to delete the custom model '\(customModel.displayName)'?"
                                    deleteActionClosure = {
                                        customModelManager.removeCustomModel(withId: customModel.id)
                                        transcriptionModelManager.refreshAllAvailableModels()
                                    }
                                    isShowingDeleteAlert = true
                                } else if let downloadedModel = whisperModelManager.availableModels.first(where: { $0.name == model.name }) {
                                    alertTitle = "Delete Model"
                                    alertMessage = "Are you sure you want to delete the model '\(downloadedModel.name)'?"
                                    deleteActionClosure = {
                                        Task {
                                            await whisperModelManager.deleteModel(downloadedModel)
                                        }
                                    }
                                    isShowingDeleteAlert = true
                                } else if model is LocalVoxtralModel {
                                    alertTitle = "Delete Model"
                                    alertMessage = "Are you sure you want to delete the model '\(model.displayName)'?"
                                    deleteActionClosure = {
                                        Task {
                                            await VoxtralNativeModelManager.shared.deleteModelAssets(for: LocalVoxtralConfiguration.modelName)
                                            if transcriptionModelManager.currentTranscriptionModel?.name == model.name {
                                                transcriptionModelManager.clearCurrentTranscriptionModel()
                                            }
                                        }
                                    }
                                    isShowingDeleteAlert = true
                                } else if model is LocalCohereTranscribeModel {
                                    alertTitle = "Delete Model"
                                    alertMessage = "Delete the local MLX model files for '\(model.displayName)'?"
                                    deleteActionClosure = {
                                        Task {
                                            await CohereNativeModelManager.shared.deleteManagedAssets()
                                            if transcriptionModelManager.currentTranscriptionModel?.name == model.name {
                                                transcriptionModelManager.clearCurrentTranscriptionModel()
                                            }
                                        }
                                    }
                                    isShowingDeleteAlert = true
                                }
                            },
                            setDefaultAction: {
                                Task {
                                    transcriptionModelManager.setDefaultTranscriptionModel(model)
                                }
                            },
                            downloadAction: {
                                if let localModel = model as? LocalModel {
                                    Task { await whisperModelManager.downloadModel(localModel) }
                                }
                            },
                            editAction: model.provider == .custom ? { customModel in
                                customModelToEdit = customModel
                            } : nil
                        )
                    }

                    AddCustomModelCardView(
                        customModelManager: customModelManager,
                        onModelAdded: {
                        transcriptionModelManager.refreshAllAvailableModels()
                        customModelToEdit = nil
                    },
                        editingModel: customModelToEdit
                    )
                }
            }
        }
        .padding()
    }

    private var displayedModels: [any TranscriptionModel] {
        let preferredOrder = [
            "apple-speech",
            "whisper-large-v3-turbo",
            "parakeet-tdt-0.6b-v2",
            "parakeet-tdt-0.6b-v3",
            "voxtral-mini-realtime-local",
            "cohere-transcribe-03-2026-local",
            "scribe_v2",
            "xai-stt"
        ]
        let currentModelName = transcriptionModelManager.currentTranscriptionModel?.name
        
        return transcriptionModelManager.allAvailableModels.sorted { model1, model2 in
            let isCurrent1 = model1.name == currentModelName
            let isCurrent2 = model2.name == currentModelName

            if isCurrent1 != isCurrent2 {
                return isCurrent1
            }

            let index1 = preferredOrder.firstIndex(of: model1.name) ?? Int.max
            let index2 = preferredOrder.firstIndex(of: model2.name) ?? Int.max
            
            if index1 != index2 {
                return index1 < index2
            }
            
            if model1.provider != model2.provider {
                return model1.provider.rawValue < model2.provider.rawValue
            }
            
            return model1.displayName.localizedStandardCompare(model2.displayName) == .orderedAscending
        }
    }
}
