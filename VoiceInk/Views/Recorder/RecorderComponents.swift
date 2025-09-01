import SwiftUI

// MARK: - Hover Interaction Manager
class HoverInteraction: ObservableObject {
    @Published var isHovered: Bool = false
    private var timer: Timer?

    func setHover(on: Bool, delay: TimeInterval = 0.1) {
        if on {
            timer?.invalidate()
            if !isHovered {
                isHovered = true
            }
        } else {
            timer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
                self?.isHovered = false
            }
        }
    }
}

// MARK: - Generic Toggle Button Component
struct RecorderToggleButton: View {
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
                        .font(.system(size: 14))
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 13))
                }
            }
            .foregroundColor(disabled ? .white.opacity(0.3) : (isEnabled ? .white : .white.opacity(0.6)))
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(disabled)
    }
}

// MARK: - Generic Record Button Component
struct RecorderRecordButton: View {
    let isRecording: Bool
    let isProcessing: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(buttonColor)
                    .frame(width: 25, height: 25)
                
                if isProcessing {
                    ProcessingIndicator(color: .white)
                        .frame(width: 16, height: 16)
                } else if isRecording {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.white)
                        .frame(width: 9, height: 9)
                } else {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 9, height: 9)
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

// MARK: - Processing Indicator Component
struct ProcessingIndicator: View {
    @State private var rotation: Double = 0
    let color: Color
    
    var body: some View {
        Circle()
            .trim(from: 0.1, to: 0.9)
            .stroke(color, lineWidth: 1.7)
            .frame(width: 14, height: 14)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
    }
}

// MARK: - Progress Animation Component
struct ProgressAnimation: View {
    @State private var currentDot = 0
    @State private var timer: Timer?
    let animationSpeed: Double
    
    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<5, id: \.self) { index in
                Circle()
                    .fill(Color.white.opacity(index <= currentDot ? 0.8 : 0.2))
                    .frame(width: 3.5, height: 3.5)
            }
        }
        .onAppear {
            timer = Timer.scheduledTimer(withTimeInterval: animationSpeed, repeats: true) { _ in
                currentDot = (currentDot + 1) % 7
                if currentDot >= 5 { currentDot = -1 }
            }
        }
        .onDisappear {
            timer?.invalidate()
            timer = nil
        }
    }
}

// MARK: - Prompt Button Component
struct RecorderPromptButton: View {
    @EnvironmentObject private var enhancementService: AIEnhancementService
    @Binding var showPopover: Bool
    let buttonSize: CGFloat
    let padding: EdgeInsets
    @StateObject private var hoverInteraction = HoverInteraction()

    init(showPopover: Binding<Bool>, buttonSize: CGFloat = 28, padding: EdgeInsets = EdgeInsets(top: 0, leading: 7, bottom: 0, trailing: 0)) {
        self._showPopover = showPopover
        self.buttonSize = buttonSize
        self.padding = padding
    }
    
    var body: some View {
        RecorderToggleButton(
            isEnabled: enhancementService.isEnhancementEnabled,
            icon: enhancementService.activePrompt?.icon.rawValue ?? enhancementService.allPrompts.first(where: { $0.id == PredefinedPrompts.defaultPromptId })?.icon.rawValue ?? "checkmark.seal.fill",
            color: .blue,
            disabled: false
        ) {
            if enhancementService.isEnhancementEnabled {
                showPopover.toggle()
            } else {
                enhancementService.isEnhancementEnabled = true
            }
        }
        .frame(width: buttonSize)
        .padding(padding)
        .onHover { hoverInteraction.setHover(on: $0) }
        .popover(isPresented: $showPopover, arrowEdge: .bottom) {
            EnhancementPromptPopover()
                .environmentObject(enhancementService)
                .onHover { hoverInteraction.setHover(on: $0) }
        }
        .onChange(of: hoverInteraction.isHovered) { isHovered in
            if isHovered != showPopover {
                showPopover = isHovered
            }
        }
    }
}

// MARK: - Power Mode Button Component
struct RecorderPowerModeButton: View {
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    @Binding var showPopover: Bool
    let buttonSize: CGFloat
    let padding: EdgeInsets
    @StateObject private var hoverInteraction = HoverInteraction()
    
    init(showPopover: Binding<Bool>, buttonSize: CGFloat = 28, padding: EdgeInsets = EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 7)) {
        self._showPopover = showPopover
        self.buttonSize = buttonSize
        self.padding = padding
    }
    
    var body: some View {
        RecorderToggleButton(
            isEnabled: !powerModeManager.enabledConfigurations.isEmpty,
            icon: powerModeManager.enabledConfigurations.isEmpty ? "✨" : (powerModeManager.currentActiveConfiguration?.emoji ?? "✨"),
            color: .orange,
            disabled: powerModeManager.enabledConfigurations.isEmpty
        ) {
            showPopover.toggle()
        }
        .frame(width: buttonSize)
        .padding(padding)
        .onHover { hoverInteraction.setHover(on: $0) }
        .popover(isPresented: $showPopover, arrowEdge: .bottom) {
            PowerModePopover()
                .onHover { hoverInteraction.setHover(on: $0) }
        }
        .onChange(of: hoverInteraction.isHovered) { isHovered in
            if isHovered != showPopover {
                showPopover = isHovered
            }
        }
    }
}

// MARK: - Status Display Component
struct RecorderStatusDisplay: View {
    let currentState: RecordingState
    let audioMeter: AudioMeter
    let menuBarHeight: CGFloat?
    
    init(currentState: RecordingState, audioMeter: AudioMeter, menuBarHeight: CGFloat? = nil) {
        self.currentState = currentState
        self.audioMeter = audioMeter
        self.menuBarHeight = menuBarHeight
    }
    
    var body: some View {
        Group {
            if currentState == .enhancing {
                VStack(spacing: 2) {
                    Text("Enhancing")
                        .foregroundColor(.white)
                        .font(.system(size: 11, weight: .medium, design: .default))
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)
                    
                    ProgressAnimation(animationSpeed: 0.15)
                }
            } else if currentState == .transcribing {
                VStack(spacing: 2) {
                    Text("Transcribing")
                        .foregroundColor(.white)
                        .font(.system(size: 11, weight: .medium, design: .default))
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)
                    
                    ProgressAnimation(animationSpeed: 0.12)
                }
            } else if currentState == .recording {
                AudioVisualizer(
                    audioMeter: audioMeter,
                    color: .white,
                    isActive: currentState == .recording
                )
                .scaleEffect(y: menuBarHeight != nil ? min(1.0, (menuBarHeight! - 8) / 25) : 1.0, anchor: .center)
            } else {
                StaticVisualizer(color: .white)
                    .scaleEffect(y: menuBarHeight != nil ? min(1.0, (menuBarHeight! - 8) / 25) : 1.0, anchor: .center)
            }
        }
    }
}