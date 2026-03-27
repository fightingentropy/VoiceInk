import SwiftUI
import AppKit

struct LocalCohereTranscribeModelCardView: View {
    let model: LocalCohereTranscribeModel
    let isCurrent: Bool
    var deleteAction: () -> Void
    var setDefaultAction: () -> Void

    @ObservedObject private var nativeModelManager = CohereNativeModelManager.shared

    private var availability: CohereNativeModelLocator.Availability {
        nativeModelManager.availability()
    }

    private var downloadState: CohereNativeModelManager.DownloadState {
        nativeModelManager.downloadState()
    }

    private var isDownloaded: Bool {
        nativeModelManager.isModelDownloaded()
    }

    private var isDownloading: Bool {
        downloadState == .downloading
    }

    private var modelDirectoryURL: URL? {
        switch availability {
        case .appManaged(let url), .externalLocalPath(let url):
            return url
        case .missing:
            return nil
        }
    }

    private var isAppManaged: Bool {
        if case .appManaged = availability {
            return true
        }
        return false
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

            if let errorMessage = downloadState.errorMessage {
                Divider()
                    .padding(.horizontal, 16)

                Text(errorMessage)
                    .font(.caption)
                    .foregroundColor(Color(.systemRed))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
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
                badge("Default", color: Color.accentColor, textColor: .white)
            } else if isDownloading {
                badge("Downloading", color: Color(.systemOrange).opacity(0.18), textColor: Color(.systemOrange))
            } else if isDownloaded {
                badge("Downloaded", color: Color(.quaternaryLabelColor), textColor: Color(.labelColor))
            } else {
                EmptyView()
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
            } else if isDownloaded {
                Button(action: setDefaultAction) {
                    Text("Set as Default")
                        .font(.system(size: 12))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button {
                    Task {
                        await nativeModelManager.downloadModelIfNeeded()
                    }
                } label: {
                    HStack(spacing: 6) {
                        if isDownloading {
                            ProgressView()
                                .scaleEffect(0.7)
                                .frame(width: 12, height: 12)
                        } else {
                            Image(systemName: "arrow.down.circle")
                                .font(.system(size: 12, weight: .medium))
                        }
                        Text(isDownloading ? "Downloading..." : "Download")
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
                .disabled(isDownloading)
            }

            if modelDirectoryURL != nil {
                Menu {
                    if isAppManaged {
                        Button(action: deleteAction) {
                            Label("Delete Model", systemImage: "trash")
                        }
                    }

                    if let modelDirectoryURL {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([modelDirectoryURL])
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
}
