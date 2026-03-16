import SwiftUI

struct APIKeyManagementView: View {
    @EnvironmentObject private var aiService: AIService
    @State private var apiKey: String = ""
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var isVerifying = false

    var body: some View {
        Section("AI Provider Integration") {
            HStack {
                Text("Provider")
                Spacer()
                Text(aiService.selectedProvider.rawValue)
                    .foregroundColor(.secondary)

                if aiService.isAPIKeyValid {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                    Text("Connected")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 12) {
                Picker("Model", selection: Binding(
                    get: { aiService.currentModel },
                    set: { aiService.selectModel($0) }
                )) {
                    ForEach(aiService.availableModels, id: \.self) { model in
                        Text(model).tag(model)
                    }
                }

                Divider()

                if aiService.isAPIKeyValid {
                    HStack {
                        Text("API Key")
                        Spacer()
                        Text("••••••••")
                            .foregroundColor(.secondary)
                        Button("Remove", role: .destructive) {
                            aiService.clearAPIKey()
                        }
                    }
                } else {
                    SecureField("API Key", text: $apiKey)
                        .textFieldStyle(.roundedBorder)

                    HStack {
                        Link(destination: URL(string: "https://platform.openai.com/api-keys")!) {
                            HStack {
                                Image(systemName: "key.fill")
                                Text("Get API Key")
                            }
                            .font(.caption)
                            .foregroundColor(.blue)
                            .padding(.vertical, 4)
                            .padding(.horizontal, 8)
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)

                        Spacer()

                        Button(action: verifyAndSaveKey) {
                            HStack {
                                if isVerifying {
                                    ProgressView().controlSize(.small)
                                }
                                Text("Verify and Save")
                            }
                        }
                        .disabled(apiKey.isEmpty || isVerifying)
                    }
                }
            }
        }
        .alert("Error", isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }

    private func verifyAndSaveKey() {
        isVerifying = true
        aiService.saveAPIKey(apiKey) { success, errorMessage in
            isVerifying = false
            if !success {
                alertMessage = errorMessage ?? "Verification failed"
                showAlert = true
            }
            apiKey = ""
        }
    }
}
