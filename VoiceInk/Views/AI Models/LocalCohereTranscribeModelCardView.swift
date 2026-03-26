import SwiftUI
import AppKit

struct LocalCohereTranscribeModelCardView: View {
    let model: LocalCohereTranscribeModel
    let isCurrent: Bool
    var deleteAction: () -> Void
    var setDefaultAction: () -> Void

    @EnvironmentObject private var transcriptionModelManager: TranscriptionModelManager
    @ObservedObject private var environmentManager = CohereTranscribeEnvironmentManager.shared

    @State private var isExpanded = false
    @State private var accessToken = ""
    @State private var isVerifyingToken = false
    @State private var verificationError: String?
    @State private var verificationSucceeded = false

    private var modelDirectoryExists: Bool {
        FileManager.default.fileExists(atPath: environmentManager.modelAssetsDirectory.path)
    }

    private var isConfigured: Bool {
        environmentManager.isConfigured
    }

    private var runtimeInstalled: Bool {
        environmentManager.isRuntimeInstalled
    }

    private var hasSavedToken: Bool {
        environmentManager.hasHuggingFaceAccessToken
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    headerSection
                    metadataSection
                    descriptionSection
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                actionSection
            }
            .padding(16)

            if isExpanded {
                Divider()
                    .padding(.horizontal, 16)

                configurationSection
                    .padding(16)
            }
        }
        .background(CardBackground(isSelected: isCurrent, useAccentGradientWhenSelected: isCurrent))
        .onAppear {
            if let savedToken = APIKeyManager.shared.getAPIKey(forProvider: LocalCohereTranscribeConfiguration.huggingFaceProviderName) {
                accessToken = savedToken
            }
        }
    }

    private var headerSection: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(model.displayName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(.labelColor))

            statusBadge

            Spacer()
        }
    }

    private var statusBadge: some View {
        Group {
            if isCurrent {
                badge("Default", color: Color.accentColor, textColor: .white)
            } else if environmentManager.installState == .installing {
                badge("Installing", color: Color(.systemOrange).opacity(0.18), textColor: Color(.systemOrange))
            } else if isConfigured {
                badge("Ready", color: Color(.systemGreen).opacity(0.18), textColor: Color(.systemGreen))
            } else {
                badge("Setup Required", color: Color(.systemOrange).opacity(0.18), textColor: Color(.systemOrange))
            }
        }
    }

    private func badge(_ label: String, color: Color, textColor: Color) -> some View {
        Text(label)
            .font(.system(size: 11, weight: .medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(color))
            .foregroundColor(textColor)
    }

    private var metadataSection: some View {
        HStack(spacing: 12) {
            Label(model.provider.rawValue, systemImage: "cpu")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            Label(model.language, systemImage: "globe")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            Label(model.size, systemImage: "internaldrive")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            HStack(spacing: 3) {
                Text("Speed")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))
                progressDotsWithNumber(value: model.speed * 10)
            }
            .fixedSize(horizontal: true, vertical: false)

            HStack(spacing: 3) {
                Text("Accuracy")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))
                progressDotsWithNumber(value: model.accuracy * 10)
            }
            .fixedSize(horizontal: true, vertical: false)
        }
        .lineLimit(1)
    }

    private var descriptionSection: some View {
        Text(model.description)
            .font(.system(size: 11))
            .foregroundColor(Color(.secondaryLabelColor))
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 4)
    }

    private var actionSection: some View {
        HStack(spacing: 8) {
            if isCurrent {
                Text("Default Model")
                    .font(.system(size: 12))
                    .foregroundColor(Color(.secondaryLabelColor))
            } else if isConfigured {
                Button(action: setDefaultAction) {
                    Text("Set as Default")
                        .font(.system(size: 12))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button(action: {
                    withAnimation(.interpolatingSpring(stiffness: 170, damping: 20)) {
                        isExpanded.toggle()
                    }
                }) {
                    HStack(spacing: 4) {
                        Text("Configure")
                            .font(.system(size: 12, weight: .medium))
                        Image(systemName: "gear")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(Color(.controlAccentColor))
                            .shadow(color: Color(.controlAccentColor).opacity(0.2), radius: 2, x: 0, y: 1)
                    )
                }
                .buttonStyle(.plain)
            }

            if runtimeInstalled || hasSavedToken || modelDirectoryExists {
                Menu {
                    if hasSavedToken {
                        Button {
                            environmentManager.clearAccessToken()
                            verificationSucceeded = false
                            verificationError = nil
                            if isCurrent {
                                transcriptionModelManager.clearCurrentTranscriptionModel()
                            }
                        } label: {
                            Label("Remove Hugging Face Token", systemImage: "key.slash")
                        }
                    }

                    if modelDirectoryExists {
                        Button(action: deleteAction) {
                            Label("Delete Runtime and Cache", systemImage: "trash")
                        }

                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([environmentManager.modelAssetsDirectory])
                        } label: {
                            Label("Show in Finder", systemImage: "folder")
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 14))
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .frame(width: 20, height: 20)
            }
        }
    }

    private var configurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Local Setup")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(.labelColor))

            Text("Cohere Transcribe is a gated Hugging Face model. Accept access on Hugging Face once, save a token here, then install the local runtime.")
                .font(.caption)
                .foregroundColor(Color(.secondaryLabelColor))

            Link("Open Cohere Transcribe on Hugging Face", destination: URL(string: "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026")!)
                .font(.caption)

            HStack(spacing: 8) {
                SecureField("Enter your Hugging Face access token", text: $accessToken)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isVerifyingToken)

                Button(action: verifyAndSaveToken) {
                    HStack(spacing: 4) {
                        if isVerifyingToken {
                            ProgressView()
                                .scaleEffect(0.7)
                                .frame(width: 12, height: 12)
                        } else {
                            Image(systemName: verificationSucceeded ? "checkmark" : "checkmark.shield")
                                .font(.system(size: 12, weight: .medium))
                        }
                        Text(isVerifyingToken ? "Verifying..." : "Verify")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(verificationSucceeded ? Color(.systemGreen) : Color(.controlAccentColor))
                    )
                }
                .buttonStyle(.plain)
                .disabled(accessToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isVerifyingToken)
            }

            if verificationSucceeded {
                Text("Hugging Face access verified.")
                    .font(.caption)
                    .foregroundColor(Color(.systemGreen))
            } else if let verificationError {
                Text(verificationError)
                    .font(.caption)
                    .foregroundColor(Color(.systemRed))
            } else if hasSavedToken {
                Text("A Hugging Face token is already saved.")
                    .font(.caption)
                    .foregroundColor(Color(.systemGreen))
            }

            HStack(spacing: 8) {
                Button {
                    Task {
                        await environmentManager.installRuntimeIfNeeded()
                    }
                } label: {
                    HStack(spacing: 4) {
                        if environmentManager.installState == .installing {
                            ProgressView()
                                .scaleEffect(0.7)
                                .frame(width: 12, height: 12)
                        } else {
                            Image(systemName: runtimeInstalled ? "checkmark.circle" : "arrow.down.circle")
                                .font(.system(size: 12, weight: .medium))
                        }
                        Text(installButtonTitle)
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(runtimeInstalled ? Color(.systemGreen) : Color(.controlAccentColor))
                    )
                }
                .buttonStyle(.plain)
                .disabled(environmentManager.installState == .installing)

                if runtimeInstalled {
                    Text("Runtime ready")
                        .font(.caption)
                        .foregroundColor(Color(.systemGreen))
                }
            }

            if let installError = environmentManager.installState.errorMessage {
                Text(installError)
                    .font(.caption)
                    .foregroundColor(Color(.systemRed))
            }
        }
    }

    private var installButtonTitle: String {
        switch environmentManager.installState {
        case .installing:
            return "Installing..."
        case .installed:
            return "Runtime Installed"
        default:
            return "Install Runtime"
        }
    }

    private func verifyAndSaveToken() {
        let trimmedToken = accessToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else { return }

        isVerifyingToken = true
        verificationError = nil
        verificationSucceeded = false

        Task {
            do {
                try await environmentManager.verifyAndSaveAccessToken(trimmedToken)
                await MainActor.run {
                    verificationSucceeded = true
                    isVerifyingToken = false
                }
            } catch {
                await MainActor.run {
                    verificationError = error.localizedDescription
                    isVerifyingToken = false
                }
            }
        }
    }
}
