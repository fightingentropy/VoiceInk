import SwiftUI
import KeyboardShortcuts
import AppKit

struct RecordView: View {
    @EnvironmentObject var whisperState: WhisperState
    @EnvironmentObject var hotkeyManager: HotkeyManager
    @Environment(\.colorScheme) private var colorScheme
    @ObservedObject private var mediaController = MediaController.shared
    
    private var hasShortcutSet: Bool {
        KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder) != nil
    }
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            mainContent
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private var mainContent: some View {
        VStack(spacing: 48) {
            heroSection
            controlsSection
        }
        .padding(32)
    }
    
    private var heroSection: some View {
        VStack(spacing: 20) {
            AppIconView()
            titleSection
        }
    }
    
    private var titleSection: some View {
        VStack(spacing: 8) {
            Text("VOICEINK")
                .font(.system(size: 42, weight: .bold))
            
            if whisperState.currentTranscriptionModel != nil {
                Text("Powered by Whisper AI")
                    .font(.system(size: 15))
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private var controlsSection: some View {
        VStack(spacing: 32) {
            compactControlsCard
            instructionsCard
        }
    }
    
    private var compactControlsCard: some View {
        HStack(spacing: 32) {
            shortcutSection
            
            if hasShortcutSet {
                Divider()
                    .frame(height: 40)
                pushToTalkSection
                
                Divider()
                    .frame(height: 40)
                
                // Settings section
                VStack(alignment: .leading, spacing: 12) {
                    Toggle(isOn: $whisperState.isAutoCopyEnabled) {
                        HStack {
                            Image(systemName: "doc.on.clipboard")
                                .foregroundColor(.secondary)
                            Text("Auto-copy to clipboard")
                                .font(.subheadline.weight(.medium))
                        }
                    }
                    .toggleStyle(.switch)
                    
                    Toggle(isOn: .init(
                        get: { SoundManager.shared.isEnabled },
                        set: { SoundManager.shared.isEnabled = $0 }
                    )) {
                        HStack {
                            Image(systemName: "speaker.wave.2")
                                .foregroundColor(.secondary)
                            Text("Sound feedback")
                                .font(.subheadline.weight(.medium))
                        }
                    }
                    .toggleStyle(.switch)
                    
                    Toggle(isOn: $mediaController.isSystemMuteEnabled) {
                        HStack {
                            Image(systemName: "speaker.slash")
                                .foregroundColor(.secondary)
                            Text("Mute system audio during recording")
                                .font(.subheadline.weight(.medium))
                        }
                    }
                    .toggleStyle(.switch)
                    .help("Automatically mute system audio when recording starts and restore when recording stops")
                }
            }
        }
        .padding(24)
        .background(CardBackground(isSelected: false))
    }
    
    private var shortcutSection: some View {
        VStack(spacing: 12) {
            if hasShortcutSet {
                if let shortcut = KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder) {
                    KeyboardShortcutView(shortcut: shortcut)
                        .scaleEffect(1.2)
                }
            } else {
                Image(systemName: "keyboard.badge.exclamationmark")
                    .font(.system(size: 28))
                    .foregroundColor(.orange)
            }
            
            Button(action: {
                NotificationCenter.default.post(
                    name: .navigateToDestination,
                    object: nil,
                    userInfo: ["destination": "Settings"]
                )
            }) {
                Text(hasShortcutSet ? "Change" : "Set Shortcut")
                    .font(.subheadline.weight(.medium))
                    .foregroundColor(.accentColor)
            }
            .buttonStyle(.plain)
        }
    }
    
    private var pushToTalkSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Push-to-Talk")
                    .font(.subheadline.weight(.medium))
                
                if hotkeyManager.isPushToTalkEnabled {
                    SelectableKeyCapView(
                        text: getKeySymbol(for: hotkeyManager.pushToTalkKey),
                        subtext: getKeyText(for: hotkeyManager.pushToTalkKey),
                        isSelected: true
                    )
                }
            }
        }
    }
    
    private func getKeySymbol(for key: HotkeyManager.PushToTalkKey) -> String {
        switch key {
        case .rightOption: return "⌥"
        case .leftOption: return "⌥"
        case .leftControl: return "⌃"
        case .rightControl: return "⌃"
        case .fn: return "Fn"
        case .rightCommand: return "⌘"
        case .rightShift: return "⇧"
        }
    }
    
    private func getKeyText(for key: HotkeyManager.PushToTalkKey) -> String {
        switch key {
        case .rightOption: return "Right Option"
        case .leftOption: return "Left Option"
        case .leftControl: return "Left Control"
        case .rightControl: return "Right Control"
        case .fn: return "Function"
        case .rightCommand: return "Right Command"
        case .rightShift: return "Right Shift"
        }
    }
    
    private var instructionsCard: some View {
        VStack(alignment: .leading, spacing: 28) {
            Text("How it works")
                .font(.title3.weight(.bold))
            
            VStack(alignment: .leading, spacing: 24) {
                ForEach(getInstructions(), id: \.title) { instruction in
                    InstructionRow(instruction: instruction)
                }
                
                Divider()
                    .padding(.vertical, 4)
                
                afterRecordingSection
            }
        }
        .padding(28)
        .background(CardBackground(isSelected: false))
    }
    
    private var afterRecordingSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("After recording")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 12) {
                if whisperState.isAutoCopyEnabled {
                    InfoRow(icon: "doc.on.clipboard", text: "Copied to clipboard")
                }
                InfoRow(icon: "text.cursor", text: "Pasted at cursor position")
            }
        }
    }
    
    private func getInstructions() -> [(icon: String, title: String, description: String)] {
        let keyName: String
        switch hotkeyManager.pushToTalkKey {
        case .rightOption:
            keyName = "right Option (⌥)"
        case .leftOption:
            keyName = "left Option (⌥)"
        case .leftControl:
            keyName = "left Control (⌃)"
        case .rightControl:
            keyName = "right Control (⌃)"
        case .fn:
            keyName = "Fn"
        case .rightCommand:
            keyName = "right Command (⌘)"
        case .rightShift:
            keyName = "right Shift (⇧)"
        }
        
        let activateDescription = hotkeyManager.isPushToTalkEnabled ?
            "Hold the \(keyName) key" :
            "Press your configured shortcut"
        
        let finishDescription = hotkeyManager.isPushToTalkEnabled ?
            "Release the \(keyName) key to stop and process" :
            "Press the shortcut again to stop"
        
        return [
            (
                icon: "mic.circle.fill",
                title: "Start Recording",
                description: activateDescription
            ),
            (
                icon: "waveform",
                title: "Speak Clearly",
                description: "Talk into your microphone naturally"
            ),
            (
                icon: "stop.circle.fill",
                title: "Finish Up",
                description: finishDescription
            )
        ]
    }
}

// Simplified InstructionRow
struct InstructionRow: View {
    let instruction: (icon: String, title: String, description: String)
    
    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: instruction.icon)
                .font(.system(size: 20))
                .foregroundColor(.accentColor)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(instruction.title)
                    .font(.subheadline.weight(.medium))
                Text(instruction.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// Simplified InfoRow
struct InfoRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(.secondary)
            Text(text)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}
