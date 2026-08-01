import SwiftUI

enum ViewType: String, CaseIterable, Identifiable {
    case transcribeAudio = "Transcribe Audio"
    case models = "Models"
    case permissions = "Permissions"
    case audioInput = "Audio"
    case dictionary = "Dictionary"
    case settings = "Settings"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .transcribeAudio: return "waveform.badge.plus"
        case .models: return "brain.head.profile"
        case .permissions: return "lock.shield"
        case .audioInput: return "waveform"
        case .dictionary: return "book.closed"
        case .settings: return "gearshape"
        }
    }

    var subtitle: String {
        switch self {
        case .transcribeAudio: return "Transcribe an existing audio or video file"
        case .models: return "Choose and manage transcription models"
        case .permissions: return "Review the access VoiceInk needs"
        case .audioInput: return "Configure microphones and input priority"
        case .dictionary: return "Teach VoiceInk names, phrases, and replacements"
        case .settings: return "Shortcuts, recording behavior, and app preferences"
        }
    }
}

struct ContentView: View {
    @State private var selectedView: ViewType? = .settings

    private let visibleViewTypes: [ViewType] = [
        .models,
        .permissions,
        .audioInput,
        .dictionary,
        .settings
    ]

    private var currentView: ViewType {
        selectedView ?? .settings
    }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 220, ideal: 228, max: 270)
        } detail: {
            VStack(spacing: 0) {
                pageHeader

                detailView(for: currentView)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(MonochromeStyle.canvas)
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 760, minHeight: 540)
        .preferredColorScheme(.dark)
        .onReceive(NotificationCenter.default.publisher(for: .navigateToDestination)) { notification in
            guard let destination = notification.userInfo?["destination"] as? String else {
                return
            }

            switch destination {
            case "Settings":
                selectedView = .settings
            case "AI Models":
                selectedView = .models
            case "Permissions":
                selectedView = .permissions
            case "Transcribe Audio":
                selectedView = .transcribeAudio
            default:
                break
            }
        }
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            brandHeader

            List(visibleViewTypes, selection: $selectedView) { viewType in
                Label {
                    Text(viewType.rawValue)
                } icon: {
                    Image(systemName: viewType.icon)
                        .symbolRenderingMode(.hierarchical)
                        .font(.system(size: 17, weight: .medium))
                        .frame(width: 22)
                }
                .font(.system(size: 16, weight: selectedView == viewType ? .semibold : .regular))
                .foregroundStyle(selectedView == viewType ? .primary : .secondary)
                .padding(.vertical, 6)
                .padding(.horizontal, 2)
                .tag(viewType)
                .accessibilityIdentifier("sidebar.\(viewType.id)")
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)
            .safeAreaPadding(.horizontal, 7)
        }
        .background(.ultraThinMaterial)
    }

    private var brandHeader: some View {
        HStack(spacing: 11) {
            Group {
                if let appIcon = NSImage(named: "AppIcon") {
                    Image(nsImage: appIcon)
                        .resizable()
                        .interpolation(.high)
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                } else {
                    Image(systemName: "waveform")
                        .symbolRenderingMode(.hierarchical)
                }
            }
            .frame(width: 34, height: 34)

            Text("VoiceInk")
                .font(.system(size: 17, weight: .semibold))

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 18)
        .padding(.top, 14)
        .padding(.bottom, 20)
    }

    private var pageHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(currentView.rawValue)
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.primary)

                Text(currentView.subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.horizontal, 28)
        .padding(.top, 40)
        .padding(.bottom, 18)
        .background(.bar)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    @ViewBuilder
    private func detailView(for viewType: ViewType) -> some View {
        switch viewType {
        case .models:
            ModelManagementView()
        case .transcribeAudio:
            AudioTranscribeView()
        case .audioInput:
            AudioInputSettingsView()
        case .dictionary:
            DictionarySettingsView()
        case .settings:
            SettingsView()
        case .permissions:
            PermissionsView()
        }
    }
}
