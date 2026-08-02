import SwiftUI
import KeyboardShortcuts
import LaunchAtLogin

struct SettingsView: View {
    @Environment(\.modelContext) private var modelContext
    @EnvironmentObject private var hotkeyManager: HotkeyManager
    @EnvironmentObject private var transcriptionModelManager: TranscriptionModelManager
    @ObservedObject private var soundManager = SoundManager.shared
    @ObservedObject private var mediaController = MediaController.shared
    @ObservedObject private var playbackController = PlaybackController.shared
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = true
    @AppStorage("restoreClipboardAfterPaste") private var restoreClipboardAfterPaste = true
    @AppStorage("clipboardRestoreDelay") private var clipboardRestoreDelay = 2.0
    @AppStorage(AppDefaults.pasteLiveTranscriptImmediatelyKey)
    private var pasteLiveTranscriptImmediately = AppDefaults.pasteLiveTranscriptImmediatelyDefault

    @State private var showResetOnboardingAlert = false
    @State private var isCustomCancelEnabled = KeyboardShortcuts.getShortcut(for: .cancelRecorder) != nil
    @State private var isCustomCancelExpanded = false
    @State private var isSoundFeedbackExpanded = false
    @State private var isMuteSystemExpanded = false
    @State private var isRestoreClipboardExpanded = false
    @State private var isPauseMediaExpanded = false
    @State private var isMaintenanceExpanded = false

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 22) {
                shortcutsSection
                recordingSection
                applicationSection
                maintenanceSection
            }
            .padding(.horizontal, 28)
            .padding(.top, 24)
            .padding(.bottom, 32)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity, alignment: .center)
        }
        .scrollIndicators(.hidden)
        .background(Color.clear)
        .alert("Reset Onboarding", isPresented: $showResetOnboardingAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Reset", role: .destructive) {
                DispatchQueue.main.async {
                    hasCompletedOnboarding = false
                }
            }
        } message: {
            Text("You'll see the introduction screens again the next time you launch the app.")
        }
    }

    private var shortcutsSection: some View {
        MonochromeSettingsSection(title: "Shortcuts") {
            compactLabeledRow("Primary") {
                HStack(spacing: 7) {
                    hotkeyPicker(binding: $hotkeyManager.selectedHotkey1)
                    if hotkeyManager.selectedHotkey1 == .custom {
                        KeyboardShortcuts.Recorder(for: .toggleMiniRecorder)
                            .controlSize(.small)
                    }
                }
            }

            if hotkeyManager.selectedHotkey2 != .none {
                MonochromeDivider()

                compactLabeledRow("Secondary") {
                    HStack(spacing: 7) {
                        hotkeyPicker(binding: $hotkeyManager.selectedHotkey2)
                        if hotkeyManager.selectedHotkey2 == .custom {
                            KeyboardShortcuts.Recorder(for: .toggleMiniRecorder2)
                                .controlSize(.small)
                        }
                        Button {
                            withAnimation(.easeOut(duration: 0.16)) {
                                hotkeyManager.selectedHotkey2 = .none
                            }
                        } label: {
                            Image(systemName: "minus")
                                .font(.system(size: 10, weight: .semibold))
                        }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                    }
                }
            } else if hotkeyManager.selectedHotkey1 != .none {
                MonochromeDivider()

                Button {
                    withAnimation(.easeOut(duration: 0.16)) {
                        hotkeyManager.selectedHotkey2 = .rightOption
                    }
                } label: {
                    Label("Add shortcut", systemImage: "plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .padding(.vertical, 10)
            }

            MonochromeDivider()

            ExpandableSettingsRow(
                isExpanded: $isCustomCancelExpanded,
                isEnabled: $isCustomCancelEnabled,
                label: "Custom cancel"
            ) {
                compactLabeledRow("Shortcut") {
                    KeyboardShortcuts.Recorder(for: .cancelRecorder)
                        .controlSize(.small)
                }
            }
            .onChange(of: isCustomCancelEnabled) { _, isEnabled in
                if !isEnabled {
                    KeyboardShortcuts.setShortcut(nil, for: .cancelRecorder)
                    isCustomCancelExpanded = false
                }
            }

        }
    }

    private var recordingSection: some View {
        MonochromeSettingsSection(title: "Recording") {
            ExpandableSettingsRow(
                isExpanded: $isSoundFeedbackExpanded,
                isEnabled: $soundManager.isEnabled,
                label: "Sound feedback"
            ) {
                CustomSoundSettingsView()
            }

            MonochromeDivider()

            ExpandableSettingsRow(
                isExpanded: $isMuteSystemExpanded,
                isEnabled: $mediaController.isSystemMuteEnabled,
                label: "Mute system audio"
            ) {
                compactLabeledRow("Resume delay") {
                    delayPicker(selection: $mediaController.audioResumptionDelay)
                }
            }

            MonochromeDivider()

            ExpandableSettingsRow(
                isExpanded: $isRestoreClipboardExpanded,
                isEnabled: $restoreClipboardAfterPaste,
                label: "Restore clipboard"
            ) {
                compactLabeledRow("Restore after") {
                    Picker("", selection: $clipboardRestoreDelay) {
                        Text("250 ms").tag(0.25)
                        Text("500 ms").tag(0.5)
                        Text("1 sec").tag(1.0)
                        Text("2 sec").tag(2.0)
                        Text("3 sec").tag(3.0)
                        Text("4 sec").tag(4.0)
                        Text("5 sec").tag(5.0)
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .controlSize(.small)
                    .frame(width: 95)
                }
            }

            MonochromeDivider()

            HStack(spacing: 10) {
                HStack(spacing: 5) {
                    Text("Paste live text immediately")
                        .font(.system(size: 13, weight: .regular, design: .rounded))
                        .fontWidth(.condensed)

                    InfoTip(
                        "Pastes exactly the live words shown when you release the shortcut. This skips the provider's final correction, so the last word or punctuation may be incomplete. If no live text is available yet, VoiceInk waits for the final result instead."
                    )
                }
                .foregroundStyle(MonochromeStyle.primaryText)

                Spacer(minLength: 14)

                Toggle("", isOn: $pasteLiveTranscriptImmediately)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .tint(.accentColor)

                Color.clear
                    .frame(width: 26, height: 26)
                    .accessibilityHidden(true)
            }
            .frame(minHeight: 46)

            MonochromeDivider()

            ExpandableSettingsRow(
                isExpanded: $isPauseMediaExpanded,
                isEnabled: $playbackController.isPauseMediaEnabled,
                label: "Pause playing media",
                infoMessage: "Pauses media during recording and resumes it afterwards."
            ) {
                compactLabeledRow("Resume delay") {
                    delayPicker(selection: $mediaController.audioResumptionDelay)
                }
            }
        }
    }

    private var applicationSection: some View {
        MonochromeSettingsSection(title: "App") {
            LaunchAtLogin.Toggle("Launch at login")
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .fontWidth(.condensed)
                .foregroundStyle(MonochromeStyle.primaryText)
                .toggleStyle(.switch)
                .controlSize(.small)
                .tint(.accentColor)
                .frame(minHeight: 44)

            MonochromeDivider()

            HStack {
                Text("Onboarding")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .fontWidth(.condensed)
                    .foregroundStyle(MonochromeStyle.primaryText)

                Spacer()

                Button("Reset") {
                    showResetOnboardingAlert = true
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .frame(minHeight: 46)
        }
    }

    private var maintenanceSection: some View {
        MonochromeSettingsSection(title: "Maintenance") {
            Button {
                withAnimation(.easeOut(duration: 0.16)) {
                    isMaintenanceExpanded.toggle()
                }
            } label: {
                HStack {
                    Text("Backup & diagnostics")
                        .font(.system(size: 13, weight: .regular, design: .rounded))
                        .fontWidth(.condensed)
                        .foregroundStyle(MonochromeStyle.primaryText)

                    Spacer()

                    Image(systemName: "chevron.forward")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isMaintenanceExpanded ? 90 : 0))
                }
                .contentShape(Rectangle())
                .frame(minHeight: 46)
            }
            .buttonStyle(.plain)

            if isMaintenanceExpanded {
                MonochromeDivider()

                HStack(spacing: 8) {
                    Button("Export") {
                        ImportExportService.shared.exportSettings(
                            hotkeyManager: hotkeyManager,
                            mediaController: mediaController,
                            playbackController: playbackController,
                            soundManager: soundManager,
                            modelContext: modelContext
                        )
                    }

                    Button("Import") {
                        ImportExportService.shared.importSettings(
                            hotkeyManager: hotkeyManager,
                            mediaController: mediaController,
                            playbackController: playbackController,
                            soundManager: soundManager,
                            modelContext: modelContext,
                            transcriptionModelManager: transcriptionModelManager
                        )
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .padding(.vertical, 10)

                MonochromeDivider()

                DiagnosticsSettingsView()
                    .padding(.vertical, 8)
            }
        }
    }

    private func compactLabeledRow<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        HStack(spacing: 12) {
            Text(title)
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .fontWidth(.condensed)
                .foregroundStyle(MonochromeStyle.primaryText)

            Spacer(minLength: 16)

            content()
        }
        .frame(minHeight: 46)
    }

    private func hotkeyPicker(
        binding: Binding<HotkeyManager.HotkeyOption>
    ) -> some View {
        Picker("", selection: binding) {
            ForEach(HotkeyManager.HotkeyOption.allCases, id: \.self) { option in
                Text(option.displayName).tag(option)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .controlSize(.small)
        .frame(width: 155)
    }

    private func delayPicker(
        selection: Binding<Double>
    ) -> some View {
        Picker("", selection: selection) {
            Text("0 sec").tag(0.0)
            Text("1 sec").tag(1.0)
            Text("2 sec").tag(2.0)
            Text("3 sec").tag(3.0)
            Text("4 sec").tag(4.0)
            Text("5 sec").tag(5.0)
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .controlSize(.small)
        .frame(width: 82)
    }

}

struct ExpandableSettingsRow<Content: View>: View {
    @Binding var isExpanded: Bool
    @Binding var isEnabled: Bool
    let label: String
    var infoMessage: String? = nil
    var infoURL: String? = nil
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                HStack(spacing: 5) {
                    Text(label)
                        .font(.system(size: 13, weight: .regular, design: .rounded))
                        .fontWidth(.condensed)

                    if let infoMessage {
                        if let infoURL {
                            InfoTip(infoMessage, learnMoreURL: infoURL)
                        } else {
                            InfoTip(infoMessage)
                        }
                    }
                }
                .foregroundStyle(MonochromeStyle.primaryText)

                Spacer(minLength: 14)

                Toggle("", isOn: $isEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .tint(.accentColor)

                Button {
                    guard isEnabled else { return }
                    withAnimation(.easeOut(duration: 0.16)) {
                        isExpanded.toggle()
                    }
                } label: {
                    Image(systemName: "chevron.forward")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isEnabled && isExpanded ? 90 : 0))
                        .opacity(isEnabled ? 1 : 0.25)
                }
                .buttonStyle(MonochromeIconButtonStyle())
                .disabled(!isEnabled)
                .help(isExpanded ? "Hide options" : "Show options")
            }
            .frame(minHeight: 46)

            if isEnabled && isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    MonochromeDivider()
                    content()
                }
                .padding(.bottom, 9)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .onChange(of: isEnabled) { _, newValue in
            withAnimation(.easeOut(duration: 0.16)) {
                isExpanded = newValue
            }
        }
    }
}

extension Text {
    func settingsDescription() -> some View {
        self
            .font(.system(size: 11))
            .foregroundStyle(MonochromeStyle.secondaryText)
            .fixedSize(horizontal: false, vertical: true)
    }
}
