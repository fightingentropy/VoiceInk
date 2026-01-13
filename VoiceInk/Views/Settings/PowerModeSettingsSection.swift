import SwiftUI

struct PowerModeSettingsSection: View {
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    @AppStorage("powerModeUIFlag") private var powerModeUIFlag = false
    @AppStorage(PowerModeDefaults.autoRestoreKey) private var powerModeAutoRestoreEnabled = false
    @State private var showDisableAlert = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Compact header
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
                    .frame(width: 16, height: 16)

                Text("Power Mode")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)

                Spacer()

                Toggle("", isOn: toggleBinding)
                    .toggleStyle(.switch)
                    .scaleEffect(0.8)
            }
            .padding(.horizontal, 12)
            .padding(.top, 10)
            .padding(.bottom, 8)

            // Content
            VStack(alignment: .leading, spacing: 6) {
                Text("Auto-apply configs based on active app or website.")
                    .font(.system(size: 11))
                    .foregroundColor(Color(NSColor.tertiaryLabelColor))

                if powerModeUIFlag {
                    HStack(spacing: 6) {
                        Toggle("", isOn: $powerModeAutoRestoreEnabled)
                            .toggleStyle(.switch)
                            .scaleEffect(0.75)
                            .frame(width: 36)

                        Text("Auto-Restore Preferences")
                            .font(.system(size: 12))

                        InfoTip("Revert preferences after each recording session.")

                        Spacer()
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 12)
        }
        .animation(.easeInOut(duration: 0.2), value: powerModeUIFlag)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(NSColor.controlBackgroundColor).opacity(0.5))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color(NSColor.separatorColor).opacity(0.5), lineWidth: 0.5)
        )
        .alert("Power Mode Still Active", isPresented: $showDisableAlert) {
            Button("Got it", role: .cancel) { }
        } message: {
            Text("Disable or remove your Power Modes first.")
        }
    }
    
    private var toggleBinding: Binding<Bool> {
        Binding(
            get: { powerModeUIFlag },
            set: { newValue in
                if newValue {
                    powerModeUIFlag = true
                } else if powerModeManager.configurations.noneEnabled {
                    powerModeUIFlag = false
                } else {
                    showDisableAlert = true
                }
            }
        )
    }
    
}

private extension Array where Element == PowerModeConfig {
    var noneEnabled: Bool {
        allSatisfy { !$0.isEnabled }
    }
}

enum PowerModeDefaults {
    static let autoRestoreKey = "powerModeAutoRestoreEnabled"
}
