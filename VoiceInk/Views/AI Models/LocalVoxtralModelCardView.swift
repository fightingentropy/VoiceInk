import SwiftUI
import AppKit

struct LocalVoxtralModelCardView: View {
    let model: LocalVoxtralModel
    let isCurrent: Bool
    var deleteAction: () -> Void
    var setDefaultAction: () -> Void

    @ObservedObject private var nativeModelManager = VoxtralNativeModelManager.shared

    private var modelReference: String {
        LocalVoxtralConfiguration.modelName
    }

    private var modelAvailability: VoxtralNativeModelLocator.Availability {
        nativeModelManager.availability(for: modelReference)
    }

    private var isDownloaded: Bool {
        modelAvailability != .missing
    }

    private var downloadState: VoxtralNativeModelManager.DownloadState {
        nativeModelManager.downloadState(for: modelReference)
    }

    private var canDownloadModel: Bool {
        VoxtralNativeModelLocator.repositoryID(from: modelReference) != nil
    }

    private var isDownloadingModel: Bool {
        downloadState == .downloading
    }

    private var downloadFraction: Double {
        nativeModelManager.downloadProgress[modelReference] ?? 0
    }

    private var downloadButtonLabel: String {
        guard isDownloadingModel else { return "Download" }
        let percent = Int((downloadFraction * 100).rounded())
        return "Downloading \(percent)%"
    }

    private var modelURL: URL? {
        modelAvailability.directoryURL
    }

    private var showsDeleteAction: Bool {
        if case .appManaged = modelAvailability {
            return true
        }

        return false
    }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                headerSection
                metadataSection
                descriptionSection
                progressSection
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            actionSection
        }
        .padding(16)
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
            } else if isDownloaded {
                Text("Downloaded")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.quaternaryLabelColor)))
                    .foregroundColor(Color(.labelColor))
            }
        }
    }

    private var metadataSection: some View {
        HStack(spacing: 12) {
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

    private var progressSection: some View {
        Group {
            if isDownloadingModel {
                ProgressView(value: downloadFraction)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
            }
        }
    }

    private var actionSection: some View {
        HStack(spacing: 8) {
            if isDownloaded {
                if isCurrent {
                    Text("Default Model")
                        .font(.system(size: 12))
                        .foregroundColor(Color(.secondaryLabelColor))
                } else {
                    Button(action: setDefaultAction) {
                        Text("Set as Default")
                            .font(.system(size: 12))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            } else if canDownloadModel {
                Button(action: {
                    Task {
                        await nativeModelManager.downloadModelIfNeeded(modelReference)
                    }
                }) {
                    HStack(spacing: 4) {
                        Text(downloadButtonLabel)
                            .font(.system(size: 12, weight: .medium))
                        Image(systemName: "arrow.down.circle")
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
                .disabled(isDownloadingModel)
            } else {
                Text("Setup Required")
                    .font(.system(size: 12))
                    .foregroundColor(Color(.secondaryLabelColor))
            }

            if isDownloaded, let modelURL {
                Menu {
                    if showsDeleteAction {
                        Button(action: deleteAction) {
                            Label("Delete Model", systemImage: "trash")
                        }
                    }

                    Button {
                        NSWorkspace.shared.selectFile(modelURL.path, inFileViewerRootedAtPath: "")
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
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
}
