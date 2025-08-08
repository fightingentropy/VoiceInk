import SwiftUI

extension View {
    func placeholder<Content: View>(
        when shouldShow: Bool,
        alignment: Alignment = .center,
        @ViewBuilder placeholder: () -> Content) -> some View {

        ZStack(alignment: alignment) {
            placeholder().opacity(shouldShow ? 1 : 0)
            self
        }
    }
}

enum ConfigurationMode: Hashable {
    case add
    case edit(PowerModeConfig)
    
    var isAdding: Bool {
        if case .add = self { return true }
        return false
    }
    
    var title: String {
        switch self {
        case .add: return "Add Power Mode"
        case .edit: return "Edit Power Mode"
        }
    }
    
    func hash(into hasher: inout Hasher) {
        switch self {
        case .add:
            hasher.combine(0)
        case .edit(let config):
            hasher.combine(1)
            hasher.combine(config.id)
        }
    }
    
    static func == (lhs: ConfigurationMode, rhs: ConfigurationMode) -> Bool {
        switch (lhs, rhs) {
        case (.add, .add):
            return true
        case (.edit(let lhsConfig), .edit(let rhsConfig)):
            return lhsConfig.id == rhsConfig.id
        default:
            return false
        }
    }
}

enum ConfigurationType {
    case application
    case website
}

let commonEmojis = ["🏢", "🏠", "💼", "🎮", "📱", "📺", "🎵", "📚", "✏️", "🎨", "🧠", "⚙️", "💻", "🌐", "📝", "📊", "🔍", "💬", "📈", "🔧"]

struct PowerModeView: View {
    @StateObject private var powerModeManager = PowerModeManager.shared
    @EnvironmentObject private var enhancementService: AIEnhancementService
    @EnvironmentObject private var aiService: AIService
    @State private var configurationMode: ConfigurationMode?
    @State private var navigationPath = NavigationPath()
    
    var body: some View {
        NavigationStack(path: $navigationPath) {
            VStack(spacing: 0) {
                VStack(spacing: 12) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text("Power Modes")
                                    .font(.system(size: 28, weight: .bold, design: .default))
                                    .foregroundColor(.primary)
                                
                                                                 InfoTip(
                                     title: "What is Power Mode?",
                                     message: "Automatically apply custom configurations based on the app/website you are using",
                                     learnMoreURL: "https://www.youtube.com/@tryvoiceink/videos"
                                 )
                            }
                            
                            Text("Automate your workflows with context-aware configurations.")
                                .font(.system(size: 14))
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Button(action: {
                            configurationMode = .add
                            navigationPath.append(configurationMode!)
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "plus")
                                    .font(.system(size: 12, weight: .medium))
                                Text("Add Power Mode")
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.accentColor)
                            .cornerRadius(6)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
                .padding(.horizontal, 24)
                .padding(.top, 20)
                .padding(.bottom, 16)
                
                Rectangle()
                    .fill(Color(NSColor.separatorColor))
                    .frame(height: 1)
                    .padding(.horizontal, 24)
                
                GeometryReader { geometry in
                    ScrollView {
                        VStack(spacing: 0) {
                            if powerModeManager.configurations.isEmpty {
                                VStack(spacing: 24) {
                                    Spacer()
                                        .frame(height: geometry.size.height * 0.2)
                                    
                                    VStack(spacing: 16) {
                                        Image(systemName: "square.grid.2x2.fill")
                                            .font(.system(size: 48, weight: .regular))
                                            .foregroundColor(.secondary.opacity(0.6))
                                        
                                        VStack(spacing: 8) {
                                            Text("No Power Modes Yet")
                                                .font(.system(size: 20, weight: .medium))
                                                .foregroundColor(.primary)
                                            
                                            Text("Create first power mode to automate your VoiceInk workflow based on apps/website you are using")
                                                .font(.system(size: 14))
                                                .foregroundColor(.secondary)
                                                .multilineTextAlignment(.center)
                                                .lineSpacing(2)
                                        }
                                    }
                                    
                                    Spacer()
                                }
                                .frame(maxWidth: .infinity)
                                .frame(minHeight: geometry.size.height)
                            } else {
                                VStack(spacing: 0) {
                                    PowerModeConfigurationsGrid(
                                        powerModeManager: powerModeManager,
                                        onEditConfig: { config in
                                            configurationMode = .edit(config)
                                            navigationPath.append(configurationMode!)
                                        }
                                    )
                                    .padding(.horizontal, 24)
                                    .padding(.vertical, 20)
                                    
                                    Spacer()
                                        .frame(height: 40)
                                }
                            }
                        }
                    }
                }
            }
            .background(Color(NSColor.controlBackgroundColor))
            .navigationDestination(for: ConfigurationMode.self) { mode in
                ConfigurationView(mode: mode, powerModeManager: powerModeManager)
            }
        }
    }
}



// New component for section headers
struct SectionHeader: View {
    let title: String
    
    var body: some View {
        Text(title)
            .font(.system(size: 16, weight: .bold))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.bottom, 8)
    }
}
