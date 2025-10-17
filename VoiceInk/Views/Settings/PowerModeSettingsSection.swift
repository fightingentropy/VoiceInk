import SwiftUI

struct PowerModeSettingsSection: View {
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    @AppStorage("powerModeUIFlag") private var powerModeUIFlag = false
    @State private var showDisableAlert = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "sparkles.square.fill.on.square")
                    .font(.system(size: 20))
                    .foregroundColor(.accentColor)
                    .frame(width: 24, height: 24)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Power Mode")
                        .font(.headline)
                    Text("Enable to automatically apply custom configurations based on the app or website you are using.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Toggle("Enable Power Mode", isOn: toggleBinding)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CardBackground(isSelected: false, useAccentGradientWhenSelected: true))
        .alert("Power Mode Still Active", isPresented: $showDisableAlert) {
            Button("Got it", role: .cancel) { }
        } message: {
            Text("Power Mode can't be disabled while any configuration is still enabled. Disable or remove your Power Modes first.")
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
