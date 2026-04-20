import SwiftUI

// MARK: - Hover Interaction Manager
class HoverInteraction: ObservableObject {
    @Published var isHovered: Bool = false

    func setHover(on: Bool) {
        if on {
            if !isHovered {
                isHovered = true
            }
        } else {
            isHovered = false
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
    let color: Color
    let animationSpeed: Double

    private let dotCount = 5
    private let dotSize: CGFloat = 3
    private let dotSpacing: CGFloat = 2

    @State private var currentDot = 0
    @State private var animationTask: Task<Void, Never>?

    init(color: Color = .white, animationSpeed: Double = 0.3) {
        self.color = color
        self.animationSpeed = animationSpeed
    }

    var body: some View {
        HStack(spacing: dotSpacing) {
            ForEach(0..<dotCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: dotSize / 2)
                    .fill(color.opacity(index <= currentDot ? 0.85 : 0.25))
                    .frame(width: dotSize, height: dotSize)
            }
        }
        .onAppear {
            startAnimation()
        }
        .onDisappear {
            animationTask?.cancel()
            animationTask = nil
        }
    }

    private func startAnimation() {
        animationTask?.cancel()
        currentDot = 0
        animationTask = Task { @MainActor in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(animationSpeed * 1_000_000_000))
                guard !Task.isCancelled else { break }
                currentDot = (currentDot + 1) % (dotCount + 2)
                if currentDot > dotCount { currentDot = -1 }
            }
        }
    }
}

// MARK: - Status Display Component
struct RecorderStatusDisplay: View {
    enum Style {
        case regular
        case compact
    }

    let currentState: RecordingState
    let audioMeter: AudioMeter
    let menuBarHeight: CGFloat?
    let style: Style

    init(currentState: RecordingState, audioMeter: AudioMeter, menuBarHeight: CGFloat? = nil, style: Style = .regular) {
        self.currentState = currentState
        self.audioMeter = audioMeter
        self.menuBarHeight = menuBarHeight
        self.style = style
    }

    var body: some View {
        Group {
            if currentState == .transcribing {
                ProcessingStatusDisplay(mode: .transcribing, color: .white, isCompact: style == .compact)
                    .transition(.opacity)
            } else if currentState == .recording {
                AudioVisualizer(
                    audioMeter: audioMeter,
                    color: .white,
                    isActive: currentState == .recording,
                    barCount: visualizerBarCount,
                    barWidth: visualizerBarWidth,
                    barSpacing: visualizerBarSpacing,
                    minHeight: visualizerMinHeight,
                    maxHeight: visualizerMaxHeight,
                    opacity: visualizerOpacity
                )
                .scaleEffect(y: menuBarHeight != nil ? min(1.0, (menuBarHeight! - 8) / 25) : 1.0, anchor: .center)
                .transition(.opacity)
            } else {
                StaticVisualizer(
                    color: .white,
                    barCount: visualizerBarCount,
                    barWidth: visualizerBarWidth,
                    staticHeight: visualizerMinHeight,
                    barSpacing: visualizerBarSpacing,
                    opacity: style == .compact ? 0.36 : 0.5
                )
                    .scaleEffect(y: menuBarHeight != nil ? min(1.0, (menuBarHeight! - 8) / 25) : 1.0, anchor: .center)
                    .transition(.opacity)
            }
        }
        .frame(width: style == .compact ? 42 : nil, height: style == .compact ? 18 : nil)
        .animation(.easeInOut(duration: 0.2), value: currentState)
    }

    private var visualizerBarCount: Int {
        style == .compact ? 9 : 15
    }

    private var visualizerBarWidth: CGFloat {
        style == .compact ? 2 : 3
    }

    private var visualizerBarSpacing: CGFloat {
        2
    }

    private var visualizerMinHeight: CGFloat {
        style == .compact ? 3 : 4
    }

    private var visualizerMaxHeight: CGFloat {
        style == .compact ? 16 : 28
    }

    private var visualizerOpacity: Double {
        style == .compact ? 0.72 : 0.85
    }
}
