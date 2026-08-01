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
    let partialTranscript: String
    let menuBarHeight: CGFloat?
    let style: Style

    init(currentState: RecordingState, partialTranscript: String, menuBarHeight: CGFloat? = nil, style: Style = .regular) {
        self.currentState = currentState
        self.partialTranscript = partialTranscript
        self.menuBarHeight = menuBarHeight
        self.style = style
    }

    var body: some View {
        Group {
            if currentState == .transcribing {
                ProcessingStatusDisplay(mode: .transcribing, color: .white, isCompact: style == .compact)
                    .transition(.opacity)
            } else {
                LiveTranscriptStatusDisplay(
                    text: partialTranscript,
                    isListening: currentState == .recording,
                    isCompact: style == .compact
                )
                .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: style == .compact ? 18 : 28)
        .animation(.easeInOut(duration: 0.2), value: currentState)
    }
}

private struct LiveTranscriptStatusDisplay: View {
    let text: String
    let isListening: Bool
    let isCompact: Bool

    @State private var isCaretVisible = true

    private var displayText: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var hasTranscript: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        Group {
            if hasTranscript {
                HStack(spacing: 4) {
                    Text(displayText)
                        .font(.system(size: isCompact ? 12 : 13, weight: .medium))
                        .foregroundStyle(.white.opacity(0.96))
                        .lineLimit(1)
                        .truncationMode(.head)
                        .frame(maxWidth: .infinity, alignment: .trailing)

                    if isListening {
                        RoundedRectangle(cornerRadius: 0.5)
                            .fill(.white.opacity(0.9))
                            .frame(width: 1, height: isCompact ? 12 : 14)
                            .opacity(isCaretVisible ? 1 : 0.18)
                    }
                }
            } else {
                Color.clear
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.55).repeatForever(autoreverses: true)) {
                isCaretVisible = false
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(hasTranscript ? displayText : (isListening ? "Listening" : "Starting"))
        .accessibilityAddTraits(.updatesFrequently)
        .animation(nil, value: text)
    }
}
