import SwiftUI

struct LocalVoxtralModelCardView: View {
    let model: LocalVoxtralModel
    let isCurrent: Bool
    var setDefaultAction: () -> Void

    @AppStorage(LocalVoxtralConfiguration.modelNameKey) private var modelName = LocalVoxtralConfiguration.defaultModelName
    @ObservedObject private var nativeModelManager = VoxtralNativeModelManager.shared
    @State private var isExpanded = false

    private var isConfigured: Bool {
        !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var modelAvailability: VoxtralNativeModelLocator.Availability {
        nativeModelManager.availability(for: modelName)
    }

    private var downloadState: VoxtralNativeModelManager.DownloadState {
        nativeModelManager.downloadState(for: modelName)
    }

    private var canDownloadModel: Bool {
        VoxtralNativeModelLocator.repositoryID(from: modelName) != nil
    }

    private var isDownloadingModel: Bool {
        downloadState == .downloading
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    headerSection
                    metadataSection
                    descriptionSection
                    compactStatusNotice
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
                Text("Default")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.accentColor))
                    .foregroundColor(.white)
            } else if isDownloadingModel {
                Text("Downloading")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.accentColor.opacity(0.16)))
                    .foregroundColor(Color.accentColor)
            } else if case .appManaged = modelAvailability {
                Text("Downloaded")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemGreen).opacity(0.18)))
                    .foregroundColor(Color(.systemGreen))
            } else if case .sharedCache = modelAvailability {
                Text("Cached")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemTeal).opacity(0.16)))
                    .foregroundColor(Color(.systemTeal))
            } else if case .externalLocalPath = modelAvailability {
                Text("Manual")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemOrange).opacity(0.16)))
                    .foregroundColor(Color(.systemOrange))
            } else {
                Text("Local")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemOrange).opacity(0.16)))
                    .foregroundColor(Color(.systemOrange))
            }
        }
    }

    private var metadataSection: some View {
        HStack(spacing: 12) {
            Label(model.provider.rawValue, systemImage: "desktopcomputer")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            Label(model.language, systemImage: "globe")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            HStack(spacing: 3) {
                Text("Speed")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))
                progressDotsWithNumber(value: model.speed * 10)
            }
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)

            HStack(spacing: 3) {
                Text("Accuracy")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))
                progressDotsWithNumber(value: model.accuracy * 10)
            }
            .lineLimit(1)
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

    private var compactStatusNotice: some View {
        Group {
            if shouldShowCompactStatusNotice {
                statusSummary
            }
        }
    }

    private var modelAssetSummary: some View {
        HStack(spacing: 6) {
            Image(systemName: modelAssetIconName)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(modelAssetColor)

            Text(modelAssetSummaryText)
                .font(.caption)
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)
        }
        .padding(.top, 6)
    }

    private var statusSummary: some View {
        HStack(spacing: 6) {
            Image(systemName: statusSummaryIconName)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(statusSummaryColor)

            Text(statusSummaryText)
                .font(.caption)
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)
        }
        .padding(.top, 6)
    }

    private var shouldShowCompactStatusNotice: Bool {
        isDownloadingModel || modelAvailability == .missing
    }

    private var actionSection: some View {
        VStack(alignment: .trailing, spacing: 8) {
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
                Text("Setup Required")
                    .font(.system(size: 12))
                    .foregroundColor(Color(.secondaryLabelColor))
            }

            if isDownloadingModel {
                HStack(spacing: 6) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("Downloading Model")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(Color(.secondaryLabelColor))
                }
            } else {
                secondaryActionButton
            }

            Button(action: toggleAdvanced) {
                Label(isExpanded ? "Hide Advanced" : "Advanced", systemImage: isExpanded ? "chevron.up.circle" : "gearshape.circle")
                    .font(.system(size: 11, weight: .medium))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    private var configurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Advanced")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(.labelColor))

            Text("You normally do not need to change these settings. VoiceInk runs Voxtral directly in-process using MLX.")
                .font(.caption)
                .foregroundColor(.secondary)

            TextField("Model Reference or Local Path", text: $modelName)
                .textFieldStyle(.roundedBorder)

            VStack(alignment: .leading, spacing: 6) {
                Text("Model Assets")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))

                Text(modelAssetSummaryText)
                    .font(.caption)
                    .foregroundColor(.secondary)

                if let directoryURL = modelAvailability.directoryURL {
                    Text(directoryURL.path)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(Color(.secondaryLabelColor))
                        .textSelection(.enabled)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(.windowBackgroundColor).opacity(0.45))
                        )
                }

                HStack(spacing: 8) {
                    if canDownloadModel && modelAvailability == .missing {
                        Button("Download Model") {
                            Task {
                                await nativeModelManager.downloadModelIfNeeded(modelName)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(isDownloadingModel)
                    }

                    if modelAvailability.directoryURL != nil {
                        Button("Show in Finder") {
                            nativeModelManager.showModelInFinder(modelName)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    if case .appManaged = modelAvailability {
                        Button("Delete Copy") {
                            nativeModelManager.deleteAppManagedModel(modelName)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }

            if modelName != LocalVoxtralConfiguration.resolvedModelReference {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Resolved Local Path")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(Color(.secondaryLabelColor))

                    Text(LocalVoxtralConfiguration.resolvedModelReference)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(Color(.secondaryLabelColor))
                        .textSelection(.enabled)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(.windowBackgroundColor).opacity(0.45))
                        )
                }
            }
        }
    }

    private func toggleAdvanced() {
        withAnimation(.interpolatingSpring(stiffness: 170, damping: 20)) {
            isExpanded.toggle()
        }
    }

    @ViewBuilder
    private var secondaryActionButton: some View {
        switch modelAvailability {
        case .missing:
            if canDownloadModel {
                Button("Download") {
                    Task {
                        await nativeModelManager.downloadModelIfNeeded(modelName)
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        case .appManaged, .sharedCache, .externalLocalPath:
            EmptyView()
        }
    }

    private var modelAssetSummaryText: String {
        switch modelAvailability {
        case .appManaged:
            return "VoiceInk is managing a local copy of the Voxtral model."
        case .sharedCache:
            return "Voxtral is already cached locally and ready to reuse."
        case .externalLocalPath:
            return "VoiceInk is pointed at a manual local model directory."
        case .missing:
            if let errorMessage = downloadState.errorMessage {
                return errorMessage
            }
            return canDownloadModel
                ? "The model is not cached yet. VoiceInk can download it directly."
                : "This model reference is managed manually."
        }
    }

    private var modelAssetIconName: String {
        switch modelAvailability {
        case .appManaged, .sharedCache, .externalLocalPath:
            return "internaldrive.fill"
        case .missing:
            return isDownloadingModel ? "arrow.down.circle.fill" : "tray.and.arrow.down.fill"
        }
    }

    private var modelAssetColor: Color {
        switch modelAvailability {
        case .appManaged, .sharedCache:
            return Color(.systemGreen)
        case .externalLocalPath:
            return Color(.systemOrange)
        case .missing:
            return isDownloadingModel ? Color.accentColor : Color(.systemOrange)
        }
    }

    private var statusSummaryText: String {
        if isDownloadingModel {
            return "Downloading native Voxtral assets."
        }

        switch modelAvailability {
        case .appManaged, .sharedCache, .externalLocalPath:
            return "Native Voxtral is ready for in-process MLX transcription."
        case .missing:
            return canDownloadModel
                ? "Download or cache the model to enable native Voxtral."
                : "Point VoiceInk at a local model directory to enable native Voxtral."
        }
    }

    private var statusSummaryIconName: String {
        if isDownloadingModel {
            return "arrow.down.circle.fill"
        }

        switch modelAvailability {
        case .appManaged, .sharedCache, .externalLocalPath:
            return "checkmark.circle.fill"
        case .missing:
            return canDownloadModel ? "tray.and.arrow.down.fill" : "wrench.and.screwdriver.fill"
        }
    }

    private var statusSummaryColor: Color {
        if isDownloadingModel {
            return Color.accentColor
        }

        switch modelAvailability {
        case .appManaged, .sharedCache:
            return Color(.systemGreen)
        case .externalLocalPath:
            return Color(.systemOrange)
        case .missing:
            return canDownloadModel ? Color(.systemOrange) : Color(.systemYellow)
        }
    }
}
