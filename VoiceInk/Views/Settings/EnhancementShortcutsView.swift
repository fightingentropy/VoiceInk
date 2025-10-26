import SwiftUI

struct EnhancementShortcutsView: View {
    private let shortcuts: [ShortcutRowData] = [
        .toggleEnhancement,
        .switchPrompt
    ]
    
    var body: some View {
        VStack(spacing: 12) {
            ForEach(shortcuts) { shortcut in
                ShortcutRow(data: shortcut)
            }
        }
        .background(Color.clear)
    }
}

struct EnhancementShortcutsSection: View {
    @State private var isExpanded = false
    
    var body: some View {
        VStack(spacing: 0) {
            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "command")
                        .font(.system(size: 20))
                        .foregroundColor(.accentColor)
                        .frame(width: 24, height: 24)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Enhancement Shortcuts")
                            .font(.headline)
                            .foregroundColor(.primary)
                        Text("Keep enhancement prompts handy")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Image(systemName: "chevron.down")
                        .rotationEffect(.degrees(isExpanded ? 0 : -90))
                        .foregroundColor(.secondary)
                        .font(.system(size: 14, weight: .semibold))
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                Divider()
                    .transition(.opacity)
                
                VStack(alignment: .leading, spacing: 16) {
                    EnhancementShortcutsView()
                    
                    Text("Enhancement shortcuts are available only when the recorder is visible and VoiceInk is running.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(16)
                .transition(
                    .asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.98, anchor: .top)),
                        removal: .opacity
                    )
                )
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CardBackground(isSelected: false))
    }
}

// MARK: - Supporting Views
private struct ShortcutRowData: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let keyDisplay: [String]
    
    static let toggleEnhancement = ShortcutRowData(
        title: "Toggle AI Enhancement",
        description: "Quickly enable or disable enhancement while recording.",
        keyDisplay: ["⌘", "E"]
    )
    
    static let switchPrompt = ShortcutRowData(
        title: "Switch Enhancement Prompt",
        description: "Switch between your saved prompts without touching the UI. Use ⌘1–⌘0 to activate the corresponding prompt in the order they are saved.",
        keyDisplay: ["⌘", "1 – 0"]
    )
}

private struct ShortcutRow: View {
    let data: ShortcutRowData
    
    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(data.title)
                        .font(.system(size: 14, weight: .semibold))
                    InfoTip(title: data.title, message: data.description, learnMoreURL: "https://tryvoiceink.com/docs/switching-enhancement-prompts")
                }
            }
            
            Spacer(minLength: 0)
            
            HStack(spacing: 8) {
                ForEach(data.keyDisplay, id: \.self) { key in
                    KeyChip(label: key)
                }
            }
        }
        .padding(.horizontal, 8)
    }
}

private struct KeyChip: View {
    let label: String
    
    var body: some View {
        Text(label)
            .font(.system(size: 13, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color(NSColor.controlBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Color(NSColor.separatorColor).opacity(0.7), lineWidth: 0.8)
                    )
            )
            .shadow(color: Color(NSColor.shadowColor).opacity(0.1), radius: 4, x: 0, y: 2)
    }
}
