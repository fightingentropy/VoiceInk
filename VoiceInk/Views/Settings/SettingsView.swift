import SwiftUI
import Cocoa
import KeyboardShortcuts
import LaunchAtLogin
import AVFoundation

struct SettingsView: View {
    @EnvironmentObject private var updaterViewModel: UpdaterViewModel
    @EnvironmentObject private var menuBarManager: MenuBarManager
    @EnvironmentObject private var hotkeyManager: HotkeyManager
    @EnvironmentObject private var whisperState: WhisperState
    @EnvironmentObject private var enhancementService: AIEnhancementService
    @StateObject private var deviceManager = AudioDeviceManager.shared
    @ObservedObject private var soundManager = SoundManager.shared
    @ObservedObject private var mediaController = MediaController.shared
    @ObservedObject private var playbackController = PlaybackController.shared
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = true
    @AppStorage("autoUpdateCheck") private var autoUpdateCheck = true
    @AppStorage("enableAnnouncements") private var enableAnnouncements = true
    @AppStorage("restoreClipboardAfterPaste") private var restoreClipboardAfterPaste = true
    @AppStorage("clipboardRestoreDelay") private var clipboardRestoreDelay = 1.0
    @State private var showResetOnboardingAlert = false
    @State private var currentShortcut = KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder)
    @State private var isCustomCancelEnabled = false

    // Expansion states - all collapsed by default
    @State private var isCustomCancelExpanded = false
    @State private var isMiddleClickExpanded = false
    @State private var isSoundFeedbackExpanded = false
    @State private var isMuteSystemExpanded = false
    @State private var isRestoreClipboardExpanded = false

    var body: some View {
        Form {
            // MARK: - Shortcuts
            Section {
                LabeledContent("Hotkey 1") {
                    HStack(spacing: 8) {
                        hotkeyPicker(binding: $hotkeyManager.selectedHotkey1)
                        if hotkeyManager.selectedHotkey1 == .custom {
                            KeyboardShortcuts.Recorder(for: .toggleMiniRecorder)
                                .controlSize(.small)
                        }
                    }
                }

                if hotkeyManager.selectedHotkey2 != .none {
                    LabeledContent("Hotkey 2") {
                        HStack(spacing: 8) {
                            hotkeyPicker(binding: $hotkeyManager.selectedHotkey2)
                            if hotkeyManager.selectedHotkey2 == .custom {
                                KeyboardShortcuts.Recorder(for: .toggleMiniRecorder2)
                                    .controlSize(.small)
                            }
                            Button {
                                withAnimation { hotkeyManager.selectedHotkey2 = .none }
                            } label: {
                                Image(systemName: "minus.circle.fill")
                                    .foregroundColor(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                if hotkeyManager.selectedHotkey1 != .none && hotkeyManager.selectedHotkey2 == .none {
                    Button("Add Second Hotkey") {
                        withAnimation { hotkeyManager.selectedHotkey2 = .rightOption }
                    }
                }
            } header: {
                Text("Shortcuts")
            } footer: {
                Text("Quick tap for hands-free recording, hold for push-to-talk.")
            }

            // MARK: - Additional Shortcuts
            Section("Additional Shortcuts") {
                LabeledContent("Paste Last Transcription (Original)") {
                    KeyboardShortcuts.Recorder(for: .pasteLastTranscription)
                        .controlSize(.small)
                }

                LabeledContent("Paste Last Transcription (Enhanced)") {
                    KeyboardShortcuts.Recorder(for: .pasteLastEnhancement)
                        .controlSize(.small)
                }

                LabeledContent("Retry Last Transcription") {
                    KeyboardShortcuts.Recorder(for: .retryLastTranscription)
                        .controlSize(.small)
                }

                // Custom Cancel - hierarchical
                ExpandableSettingsRow(
                    isExpanded: $isCustomCancelExpanded,
                    isEnabled: $isCustomCancelEnabled,
                    label: "Custom Cancel Shortcut"
                ) {
                    LabeledContent("Shortcut") {
                        KeyboardShortcuts.Recorder(for: .cancelRecorder)
                            .controlSize(.small)
                    }
                }
                .onChange(of: isCustomCancelEnabled) { _, newValue in
                    if !newValue {
                        KeyboardShortcuts.setShortcut(nil, for: .cancelRecorder)
                        isCustomCancelExpanded = false
                    }
                }

                // Middle-Click
                ExpandableSettingsRow(
                    isExpanded: $isMiddleClickExpanded,
                    isEnabled: $hotkeyManager.isMiddleClickToggleEnabled,
                    label: "Middle-Click Recording"
                ) {
                    LabeledContent("Activation Delay") {
                        HStack {
                            TextField("", value: $hotkeyManager.middleClickActivationDelay, formatter: NumberFormatter())
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 60)
                            Text("ms")
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }

            // MARK: - Recording Feedback
            Section("Recording Feedback") {
                // Sound Feedback
                ExpandableSettingsRow(
                    isExpanded: $isSoundFeedbackExpanded,
                    isEnabled: $soundManager.isEnabled,
                    label: "Sound Feedback"
                ) {
                    CustomSoundSettingsView()
                }

                // Mute System Audio
                ExpandableSettingsRow(
                    isExpanded: $isMuteSystemExpanded,
                    isEnabled: $mediaController.isSystemMuteEnabled,
                    label: "Mute Audio While Recording"
                ) {
                    HStack(spacing: 4) {
                        Picker("Resume Delay", selection: $mediaController.audioResumptionDelay) {
                            Text("0s").tag(0.0)
                            Text("1s").tag(1.0)
                            Text("2s").tag(2.0)
                            Text("3s").tag(3.0)
                            Text("4s").tag(4.0)
                            Text("5s").tag(5.0)
                        }
                        InfoTip("Delay before unmuting. Use 2s for Bluetooth headphones, 0s for wired.")
                    }
                }

                // Restore Clipboard
                ExpandableSettingsRow(
                    isExpanded: $isRestoreClipboardExpanded,
                    isEnabled: $restoreClipboardAfterPaste,
                    label: "Restore Clipboard After Paste"
                ) {
                    Picker("Restore Delay", selection: $clipboardRestoreDelay) {
                        Text("1s").tag(1.0)
                        Text("2s").tag(2.0)
                        Text("3s").tag(3.0)
                        Text("4s").tag(4.0)
                        Text("5s").tag(5.0)
                    }
                }
            }

            // MARK: - Power Mode
            PowerModeSection()

            // MARK: - Experimental
            ExperimentalSection()

            // MARK: - Interface
            Section("Interface") {
                Picker("Recorder Style", selection: $whisperState.recorderType) {
                    Text("Notch").tag("notch")
                    Text("Mini").tag("mini")
                }
                .pickerStyle(.segmented)

                Toggle(isOn: Binding(
                    get: { UserDefaults.standard.bool(forKey: "UseAppleScriptPaste") },
                    set: { UserDefaults.standard.set($0, forKey: "UseAppleScriptPaste") }
                )) {
                    HStack(spacing: 4) {
                        Text("AppleScript Paste")
                        InfoTip("Paste via System Events instead of direct keystrokes. Try this if paste isn't working in some apps.")
                    }
                }
            }

            // MARK: - General
            Section("General") {
                Toggle("Hide Dock Icon", isOn: $menuBarManager.isMenuBarOnly)

                LaunchAtLogin.Toggle("Launch at Login")

                Toggle("Auto-check Updates", isOn: $autoUpdateCheck)
                    .onChange(of: autoUpdateCheck) { _, newValue in
                        updaterViewModel.toggleAutoUpdates(newValue)
                    }

                Toggle("Show Announcements", isOn: $enableAnnouncements)
                    .onChange(of: enableAnnouncements) { _, newValue in
                        if newValue {
                            AnnouncementsService.shared.start()
                        } else {
                            AnnouncementsService.shared.stop()
                        }
                    }

                HStack {
                    Button("Check for Updates") {
                        updaterViewModel.checkForUpdates()
                    }
                    .disabled(!updaterViewModel.canCheckForUpdates)

                    Button("Reset Onboarding") {
                        showResetOnboardingAlert = true
                    }
                }
            }

            // MARK: - Privacy
            Section {
                AudioCleanupSettingsView()
            } header: {
                Text("Privacy")
            } footer: {
                Text("Control how VoiceInk handles your transcription data and audio recordings.")
            }

            // MARK: - Data Management
            Section {
                LabeledContent {
                    Button("Import") {
                        ImportExportService.shared.importSettings(
                            enhancementService: enhancementService,
                            whisperPrompt: whisperState.whisperPrompt,
                            hotkeyManager: hotkeyManager,
                            menuBarManager: menuBarManager,
                            mediaController: MediaController.shared,
                            playbackController: PlaybackController.shared,
                            soundManager: SoundManager.shared,
                            whisperState: whisperState
                        )
                    }
                } label: {
                    Text("Import Settings")
                }

                LabeledContent {
                    Button("Export") {
                        ImportExportService.shared.exportSettings(
                            enhancementService: enhancementService,
                            whisperPrompt: whisperState.whisperPrompt,
                            hotkeyManager: hotkeyManager,
                            menuBarManager: menuBarManager,
                            mediaController: MediaController.shared,
                            playbackController: PlaybackController.shared,
                            soundManager: SoundManager.shared,
                            whisperState: whisperState
                        )
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Export Settings")
                        InfoTip("Export prompts, power modes, word replacements, shortcuts, and preferences. API keys are never included.")
                    }
                }
            } header: {
                Text("Data Management")
            }

            // MARK: - Diagnostics
            Section("Diagnostics") {
                DiagnosticsSettingsView()
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(NSColor.controlBackgroundColor))
        .onAppear {
            isCustomCancelEnabled = KeyboardShortcuts.getShortcut(for: .cancelRecorder) != nil
        }
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

    @ViewBuilder
    private func hotkeyPicker(binding: Binding<HotkeyManager.HotkeyOption>) -> some View {
        Picker("", selection: binding) {
            ForEach(HotkeyManager.HotkeyOption.allCases, id: \.self) { option in
                Text(option.displayName).tag(option)
            }
        }
        .labelsHidden()
        .frame(width: 140)
    }
}

// MARK: - Expandable Settings Row (entire row clickable)

struct ExpandableSettingsRow<Content: View>: View {
    @Binding var isExpanded: Bool
    @Binding var isEnabled: Bool
    let label: String
    var infoMessage: String? = nil
    @ViewBuilder let content: () -> Content

    @State private var isHandlingToggleChange = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Main row - entire area is tappable
            HStack {
                Toggle(isOn: $isEnabled) {
                    HStack(spacing: 4) {
                        Text(label)
                        if let message = infoMessage {
                            InfoTip(message)
                        }
                    }
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                    .rotationEffect(.degrees(isEnabled && isExpanded ? 90 : 0))
                    .opacity(isEnabled ? 1 : 0.4)
            }
            .contentShape(Rectangle())
            .onTapGesture {
                guard !isHandlingToggleChange else { return }
                if isEnabled {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                }
            }

            // Expanded content with proper spacing
            if isEnabled && isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    content()
                }
                .padding(.top, 12)
                .padding(.leading, 4)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isExpanded)
        .onChange(of: isEnabled) { _, newValue in
            isHandlingToggleChange = true
            if newValue {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded = true
                }
            } else {
                isExpanded = false
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                isHandlingToggleChange = false
            }
        }
    }
}

// MARK: - Power Mode Section

struct PowerModeSection: View {
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    @AppStorage("powerModeUIFlag") private var powerModeUIFlag = false
    @AppStorage(PowerModeDefaults.autoRestoreKey) private var powerModeAutoRestoreEnabled = false
    @State private var showDisableAlert = false
    @State private var isExpanded = false
    @State private var isHandlingToggleChange = false

    var body: some View {
        Section {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Toggle(isOn: toggleBinding) {
                        HStack(spacing: 4) {
                            Text("Power Mode")
                            InfoTip(
                                "Apply custom settings based on active app or website.",
                                learnMoreURL: "https://tryvoiceink.com/docs/power-mode"
                            )
                        }
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)
                        .rotationEffect(.degrees(powerModeUIFlag && isExpanded ? 90 : 0))
                        .opacity(powerModeUIFlag ? 1 : 0.4)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    guard !isHandlingToggleChange else { return }
                    if powerModeUIFlag {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isExpanded.toggle()
                        }
                    }
                }

                if powerModeUIFlag && isExpanded {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Auto-Restore Preferences", isOn: $powerModeAutoRestoreEnabled)
                    }
                    .padding(.top, 12)
                    .padding(.leading, 4)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .animation(.easeInOut(duration: 0.2), value: isExpanded)
        } header: {
            Text("Power Mode")
        }
        .alert("Power Mode Still Active", isPresented: $showDisableAlert) {
            Button("Got it", role: .cancel) { }
        } message: {
            Text("Disable or remove your Power Modes first.")
        }
        .onChange(of: powerModeUIFlag) { _, newValue in
            isHandlingToggleChange = true
            if newValue {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded = true
                }
            } else {
                isExpanded = false
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                isHandlingToggleChange = false
            }
        }
    }

    private var toggleBinding: Binding<Bool> {
        Binding(
            get: { powerModeUIFlag },
            set: { newValue in
                if newValue {
                    powerModeUIFlag = true
                } else if powerModeManager.configurations.allSatisfy({ !$0.isEnabled }) {
                    powerModeUIFlag = false
                } else {
                    showDisableAlert = true
                }
            }
        )
    }
}

// MARK: - Experimental Section

struct ExperimentalSection: View {
    @AppStorage("isExperimentalFeaturesEnabled") private var isExperimentalFeaturesEnabled = false
    @ObservedObject private var playbackController = PlaybackController.shared
    @ObservedObject private var mediaController = MediaController.shared
    @State private var isExpanded = false
    @State private var isPauseMediaExpanded = false
    @State private var isHandlingToggleChange = false
    @State private var isHandlingPauseMediaToggle = false

    var body: some View {
        Section {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Toggle(isOn: $isExperimentalFeaturesEnabled) {
                        HStack(spacing: 4) {
                            Text("Experimental Features")
                            InfoTip("Features in development that may be unstable.")
                        }
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)
                        .rotationEffect(.degrees(isExperimentalFeaturesEnabled && isExpanded ? 90 : 0))
                        .opacity(isExperimentalFeaturesEnabled ? 1 : 0.4)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    guard !isHandlingToggleChange else { return }
                    if isExperimentalFeaturesEnabled {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isExpanded.toggle()
                        }
                    }
                }

                if isExperimentalFeaturesEnabled && isExpanded {
                    VStack(alignment: .leading, spacing: 0) {
                        // Pause Media sub-option
                        HStack {
                            Toggle("Pause Media While Recording", isOn: $playbackController.isPauseMediaEnabled)

                            Spacer()

                            Image(systemName: "chevron.right")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundColor(.secondary)
                                .rotationEffect(.degrees(playbackController.isPauseMediaEnabled && isPauseMediaExpanded ? 90 : 0))
                                .opacity(playbackController.isPauseMediaEnabled ? 1 : 0.4)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            guard !isHandlingPauseMediaToggle else { return }
                            if playbackController.isPauseMediaEnabled {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    isPauseMediaExpanded.toggle()
                                }
                            }
                        }

                        if playbackController.isPauseMediaEnabled && isPauseMediaExpanded {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack(spacing: 4) {
                                    Picker("Resume Delay", selection: $mediaController.audioResumptionDelay) {
                                        Text("0s").tag(0.0)
                                        Text("1s").tag(1.0)
                                        Text("2s").tag(2.0)
                                        Text("3s").tag(3.0)
                                        Text("4s").tag(4.0)
                                        Text("5s").tag(5.0)
                                    }
                                    InfoTip("Delay before resuming playback. Use 2s for Bluetooth headphones, 0s for wired.")
                                }
                            }
                            .padding(.top, 12)
                            .padding(.leading, 4)
                            .transition(.opacity.combined(with: .move(edge: .top)))
                        }
                    }
                    .padding(.top, 12)
                    .padding(.leading, 4)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                    .animation(.easeInOut(duration: 0.2), value: isPauseMediaExpanded)
                }
            }
            .animation(.easeInOut(duration: 0.2), value: isExpanded)
            .onChange(of: isExperimentalFeaturesEnabled) { _, newValue in
                isHandlingToggleChange = true
                if newValue {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded = true
                    }
                } else {
                    playbackController.isPauseMediaEnabled = false
                    isExpanded = false
                    isPauseMediaExpanded = false
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    isHandlingToggleChange = false
                }
            }
            .onChange(of: playbackController.isPauseMediaEnabled) { _, newValue in
                isHandlingPauseMediaToggle = true
                if newValue {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isPauseMediaExpanded = true
                    }
                } else {
                    isPauseMediaExpanded = false
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    isHandlingPauseMediaToggle = false
                }
            }
        } header: {
            Text("Experimental")
        }
    }
}

// MARK: - Legacy SettingsSection (kept for other views that may use it)

struct SettingsSection<Content: View>: View {
    let icon: String
    let title: String
    let subtitle: String
    let content: Content
    var showWarning: Bool = false

    init(icon: String, title: String, subtitle: String, showWarning: Bool = false, @ViewBuilder content: () -> Content) {
        self.icon = icon
        self.title = title
        self.subtitle = subtitle
        self.showWarning = showWarning
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundColor(showWarning ? .red : .accentColor)
                    .frame(width: 20, height: 20)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline)
                    if !subtitle.isEmpty {
                        Text(subtitle)
                            .font(.subheadline)
                            .foregroundColor(showWarning ? .red : .secondary)
                    }
                }

                if showWarning {
                    Spacer()
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                }
            }

            Divider()

            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CardBackground(isSelected: showWarning, useAccentGradientWhenSelected: true))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(showWarning ? Color.red.opacity(0.5) : Color.clear, lineWidth: 1)
        )
    }
}

// MARK: - Text Extension

extension Text {
    func settingsDescription() -> some View {
        self
            .font(.system(size: 12))
            .foregroundColor(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
