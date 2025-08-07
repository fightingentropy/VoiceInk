import SwiftUI

struct PowerModePopover: View {
    @ObservedObject var powerModeManager = PowerModeManager.shared
    @State private var selectedConfig: PowerModeConfig?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Select Power Mode")
                .font(.headline)
                .foregroundColor(.white.opacity(0.9))
                .padding(.horizontal)
                .padding(.top, 8)
            
            Divider()
                .background(Color.white.opacity(0.1))
            
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(powerModeManager.configurations.filter { $0.isEnabled }) { config in
                        PowerModeRow(
                            config: config,
                            isSelected: selectedConfig?.id == config.id,
                            action: {
                                powerModeManager.setActiveConfiguration(config)
                                selectedConfig = config
                                applySelectedConfiguration()
                            }
                        )
                    }
                }
                .padding(.horizontal)
            }
        }
        .frame(width: 180)
        .frame(maxHeight: 300)
        .padding(.vertical, 8)
        .background(Color.black)
        .environment(\.colorScheme, .dark)
        .onAppear {
            selectedConfig = powerModeManager.activeConfiguration
        }
    }
    
    private func applySelectedConfiguration() {
        Task {
            if let config = selectedConfig {
                await PowerModeSessionManager.shared.beginSession(with: config)
            }
        }
    }
}

struct PowerModeRow: View {
    let config: PowerModeConfig
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Text(config.emoji)
                    .font(.system(size: 14))
                
                Text(config.name)
                    .foregroundColor(.white.opacity(0.9))
                    .font(.system(size: 13))
                    .lineLimit(1)
                
                if isSelected {
                    Spacer()
                    Image(systemName: "checkmark")
                        .foregroundColor(.green)
                        .font(.system(size: 10))
                }
            }
            .contentShape(Rectangle())
            .padding(.vertical, 4)
            .padding(.horizontal, 8)
        }
        .buttonStyle(.plain)
        .background(isSelected ? Color.white.opacity(0.1) : Color.clear)
        .cornerRadius(4)
    }
} 