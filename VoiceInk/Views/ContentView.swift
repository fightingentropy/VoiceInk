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
        case .transcribeAudio: return "waveform.circle"
        case .models: return "waveform.badge.magnifyingglass"
        case .permissions: return "lock.shield"
        case .audioInput: return "mic"
        case .dictionary: return "text.book.closed"
        case .settings: return "slider.horizontal.3"
        }
    }
}

struct ContentView: View {
    @State private var selectedView: ViewType = .settings

    private let visibleViewTypes: [ViewType] = [
        .models,
        .permissions,
        .audioInput,
        .dictionary,
        .settings
    ]

    var body: some View {
        ZStack {
            MonochromeStyle.canvas
                .ignoresSafeArea()

            HStack(spacing: 0) {
                sidebar

                Rectangle()
                    .fill(MonochromeStyle.hairline)
                    .frame(width: 0.75)

                VStack(spacing: 0) {
                    pageHeader

                    Rectangle()
                        .fill(MonochromeStyle.hairline)
                        .frame(height: 0.75)

                    detailView(for: selectedView)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .frame(minWidth: 760, minHeight: 540)
        .preferredColorScheme(.dark)
        .tint(.white)
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
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 9) {
                Group {
                    if let appIcon = NSImage(named: "AppIcon") {
                        Image(nsImage: appIcon)
                            .resizable()
                            .interpolation(.high)
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 24, height: 24)
                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    }
                }
                .frame(width: 30, height: 30)

                Text("VoiceInk")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(MonochromeStyle.primaryText)
            }
            .padding(.horizontal, 17)
            .padding(.top, 51)
            .padding(.bottom, 18)

            VStack(spacing: 4) {
                ForEach(visibleViewTypes) { viewType in
                    navigationButton(for: viewType)
                }
            }
            .padding(.horizontal, 10)

            Spacer()
        }
        .frame(width: 178)
        .background(MonochromeStyle.sidebar)
    }

    private func navigationButton(for viewType: ViewType) -> some View {
        let isSelected = selectedView == viewType

        return Button {
            withAnimation(.easeOut(duration: 0.16)) {
                selectedView = viewType
            }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: viewType.icon)
                    .font(.system(size: 12, weight: .medium))
                    .frame(width: 16)

                Text(viewType.rawValue)
                    .font(.system(size: 12.5, weight: isSelected ? .semibold : .medium))

                Spacer()
            }
            .foregroundStyle(
                isSelected
                    ? MonochromeStyle.primaryText
                    : MonochromeStyle.secondaryText
            )
            .padding(.horizontal, 11)
            .frame(height: 34)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(isSelected ? MonochromeStyle.selectedFill : Color.clear)
        )
        .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                .stroke(isSelected ? MonochromeStyle.hairline : Color.clear, lineWidth: 0.75)
        )
    }

    private var pageHeader: some View {
        HStack {
            Text(selectedView.rawValue)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(MonochromeStyle.primaryText)

            Spacer()

            Circle()
                .fill(MonochromeStyle.secondaryText)
                .frame(width: 5, height: 5)
        }
        .padding(.leading, 19)
        .padding(.trailing, 20)
        .padding(.top, 45)
        .padding(.bottom, 13)
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
