import SwiftUI
import SwiftData

struct ModelManagementView: View {
    @ObservedObject var whisperState: WhisperState
    @State private var customModelToEdit: CustomCloudModel?
    @StateObject private var aiService = AIService()
    @StateObject private var customModelManager = CustomModelManager.shared
    @EnvironmentObject private var enhancementService: AIEnhancementService
    @Environment(\.modelContext) private var modelContext
    @StateObject private var whisperPrompt = WhisperPrompt()

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
            Text(whisperState.currentTranscriptionModel?.displayName ?? "No model selected")
                .font(.title2)
                .fontWeight(.bold)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CardBackground(isSelected: false))
        .cornerRadius(10)
    }
    
    private var languageSelectionSection: some View {
        LanguageSelectionView(whisperState: whisperState, displayMode: .full, whisperPrompt: whisperPrompt)
    }
    
    private var availableModelsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Available Models")
                    .font(.title3)
                    .fontWeight(.semibold)
                
                Text("(\(whisperState.allAvailableModels.count))")
                    .foregroundColor(.secondary)
                    .font(.subheadline)
                
                Spacer()
            }
            
            VStack(spacing: 12) {
                ForEach(whisperState.allAvailableModels, id: \.id) { model in
                    ModelCardRowView(
                        model: model,
                        isDownloaded: whisperState.availableModels.contains { $0.name == model.name },
                        isCurrent: whisperState.currentTranscriptionModel?.name == model.name,
                        downloadProgress: whisperState.downloadProgress,
                        modelURL: whisperState.availableModels.first { $0.name == model.name }?.url,
                        deleteAction: {
                            if let customModel = model as? CustomCloudModel {
                                alertTitle = "Delete Custom Model"
                                alertMessage = "Are you sure you want to delete the custom model '\(customModel.displayName)'?"
                                deleteActionClosure = {
                                    customModelManager.removeCustomModel(withId: customModel.id)
                                    whisperState.refreshAllAvailableModels()
                                }
                                isShowingDeleteAlert = true
                            } else if let downloadedModel = whisperState.availableModels.first(where: { $0.name == model.name }) {
                                alertTitle = "Delete Model"
                                alertMessage = "Are you sure you want to delete the model '\(downloadedModel.name)'?"
                                deleteActionClosure = {
                                    Task {
                                        await whisperState.deleteModel(downloadedModel)
                                    }
                                }
                                isShowingDeleteAlert = true
                            }
                        },
                        setDefaultAction: {
                            Task {
                                await whisperState.setDefaultTranscriptionModel(model)
                            }
                        },
                        downloadAction: {
                            if let localModel = model as? LocalModel {
                                Task {
                                    await whisperState.downloadModel(localModel)
                                }
                            }
                        },
                        editAction: model.provider == .custom ? { customModel in
                            customModelToEdit = customModel
                        } : nil
                    )
                }
                
                // Add Custom Model Card at the bottom
                AddCustomModelCardView(
                    customModelManager: customModelManager,
                    editingModel: customModelToEdit
                ) {
                    // Refresh the models when a new custom model is added
                    whisperState.refreshAllAvailableModels()
                    customModelToEdit = nil // Clear editing state
                }
            }
        }
        .padding()
    }
}
