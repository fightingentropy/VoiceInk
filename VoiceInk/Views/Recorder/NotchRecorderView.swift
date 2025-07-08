import SwiftUI

struct NotchRecorderView: View {
    @ObservedObject var whisperState: WhisperState
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: NotchWindowManager
    @State private var isHovering = false
    @State private var showPowerModePopover = false
    @State private var showEnhancementPromptPopover = false
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    
    @EnvironmentObject private var enhancementService: AIEnhancementService
    
    private var menuBarHeight: CGFloat {
        if let screen = NSScreen.main {
            if screen.safeAreaInsets.top > 0 {
                return screen.safeAreaInsets.top
            }
            return NSApplication.shared.mainMenu?.menuBarHeight ?? NSStatusBar.system.thickness
        }
        return NSStatusBar.system.thickness
    }
    
    private var exactNotchWidth: CGFloat {
        if let screen = NSScreen.main {
            if screen.safeAreaInsets.left > 0 {
                return screen.safeAreaInsets.left * 2
            }
            return 200
        }
        return 200
    }
    
    var body: some View {
        Group {
            if windowManager.isVisible {
                HStack(spacing: 0) {
                    HStack(spacing: 8) {
                        NotchRecordButton(
                            isRecording: whisperState.isRecording,
                            isProcessing: whisperState.isProcessing
                        ) {
                            Task { await whisperState.toggleRecord() }
                        }
                        .frame(width: 22)
                        
                        if powerModeManager.isPowerModeEnabled {
                            NotchToggleButton(
                                isEnabled: powerModeManager.isPowerModeEnabled,
                                icon: powerModeManager.currentActiveConfiguration.emoji,
                                color: .orange,
                                disabled: false
                            ) {
                                showPowerModePopover.toggle()
                            }
                            .frame(width: 22)
                            .popover(isPresented: $showPowerModePopover, arrowEdge: .bottom) {
                                PowerModePopover()
                            }
                        } else {
                            NotchToggleButton(
                                isEnabled: enhancementService.isEnhancementEnabled,
                                icon: enhancementService.activePrompt?.icon.rawValue ?? "brain",
                                color: .blue,
                                disabled: false
                            ) {
                                if enhancementService.isEnhancementEnabled {
                                    showEnhancementPromptPopover.toggle()
                                } else {
                                    enhancementService.isEnhancementEnabled = true
                                }
                            }
                            .frame(width: 22)
                            .popover(isPresented: $showEnhancementPromptPopover, arrowEdge: .bottom) {
                                EnhancementPromptPopover()
                                    .environmentObject(enhancementService)
                            }
                        }
                        
                        Spacer()
                    }
                    .frame(width: 64)
                    .padding(.leading, 16)
                    
                    Rectangle()
                        .fill(Color.clear)
                        .frame(width: exactNotchWidth)
                        .contentShape(Rectangle())
                    
                    HStack(spacing: 0) {
                        Spacer()
                        
                        Group {
                            if whisperState.isEnhancing {
                                Text("Enhancing")
                                    .foregroundColor(.white)
                                    .font(.system(size: 10, weight: .medium, design: .default))
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.5)
                            } else if whisperState.isTranscribing {
                                Text("Transcribing")
                                    .foregroundColor(.white)
                                    .font(.system(size: 10, weight: .medium, design: .default))
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.5)
                            } else if whisperState.isRecording {
                                AudioVisualizer(
                                    audioMeter: recorder.audioMeter,
                                    color: .white,
                                    isActive: whisperState.isRecording
                                )
                                .scaleEffect(y: min(1.0, (menuBarHeight - 8) / 25), anchor: .center)
                            } else {
                                StaticVisualizer(color: .white)
                                    .scaleEffect(y: min(1.0, (menuBarHeight - 8) / 25), anchor: .center)
                            }
                        }
                        .frame(width: 70)
                        .padding(.trailing, 8)
                    }
                    .frame(width: 84)
                    .padding(.trailing, 16)
                }
                .frame(height: menuBarHeight)
                .frame(maxWidth: windowManager.isVisible ? .infinity : 0)
                .background(Color.black)
                .mask {
                    NotchShape(cornerRadius: 10)
                }
                .clipped()
                .onHover { hovering in
                    isHovering = hovering
                }
                .opacity(windowManager.isVisible ? 1 : 0)
            }
        }
    }
}



struct NotchToggleButton: View {
    let isEnabled: Bool
    let icon: String
    let color: Color
    let disabled: Bool
    let action: () -> Void
    
    init(isEnabled: Bool, icon: String, color: Color, disabled: Bool = false, action: @escaping () -> Void) {
        self.isEnabled = isEnabled
        self.icon = icon
        self.color = color
        self.disabled = disabled
        self.action = action
    }
    
    private var isEmoji: Bool {
        return !icon.contains(".") && !icon.contains("-") && icon.unicodeScalars.contains { !$0.isASCII }
    }
    
    var body: some View {
        Button(action: action) {
            Group {
                if isEmoji {
                    Text(icon)
                        .font(.system(size: 12))
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 11))
                }
            }
            .foregroundColor(disabled ? .white.opacity(0.3) : (isEnabled ? .white : .white.opacity(0.6)))
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(disabled)
    }
}



// Notch-specific button styles
struct NotchRecordButton: View {
    let isRecording: Bool
    let isProcessing: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(buttonColor)
                    .frame(width: 22, height: 22)
                
                if isProcessing {
                    ProcessingIndicator(color: .white)
                        .frame(width: 14, height: 14)
                } else if isRecording {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.white)
                        .frame(width: 8, height: 8)
                } else {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 8, height: 8)
                }
            }
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(isProcessing)
    }
    
    private var buttonColor: Color {
        if isProcessing {
            return Color(red: 0.4, green: 0.4, blue: 0.45)
        } else if isRecording {
            return .red
        } else {
            return Color(red: 0.3, green: 0.3, blue: 0.35)
        }
    }
}

struct ProcessingIndicator: View {
    @State private var rotation: Double = 0
    let color: Color
    
    var body: some View {
        Circle()
            .trim(from: 0.1, to: 0.9)
            .stroke(color, lineWidth: 1.5)
            .frame(width: 12, height: 12)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
    }
} 
