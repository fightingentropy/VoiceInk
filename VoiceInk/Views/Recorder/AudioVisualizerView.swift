import SwiftUI

struct AudioVisualizer: View {
    let audioMeter: AudioMeter
    let color: Color
    let isActive: Bool

    private let barCount: Int
    private let barWidth: CGFloat
    private let barSpacing: CGFloat
    private let minHeight: CGFloat
    private let maxHeight: CGFloat
    private let opacity: Double

    private let phases: [Double]

    init(
        audioMeter: AudioMeter,
        color: Color,
        isActive: Bool,
        barCount: Int = 15,
        barWidth: CGFloat = 3,
        barSpacing: CGFloat = 2,
        minHeight: CGFloat = 4,
        maxHeight: CGFloat = 28,
        opacity: Double = 0.85
    ) {
        self.audioMeter = audioMeter
        self.color = color
        self.isActive = isActive
        self.barCount = barCount
        self.barWidth = barWidth
        self.barSpacing = barSpacing
        self.minHeight = minHeight
        self.maxHeight = maxHeight
        self.opacity = opacity

        // Create smooth wave phases
        self.phases = (0..<barCount).map { Double($0) * 0.4 }
    }

    var body: some View {
        // TimelineView with 60Hz updates (native approach recommended by Apple WWDC 2021+)
        TimelineView(.animation(minimumInterval: 0.016)) { context in
            HStack(spacing: barSpacing) {
                ForEach(0..<barCount, id: \.self) { index in
                    RoundedRectangle(cornerRadius: barWidth / 2)
                        .fill(color.opacity(opacity))
                        .frame(width: barWidth, height: calculateHeight(for: index, at: context.date))
                }
            }
        }
    }

    private func calculateHeight(for index: Int, at date: Date) -> CGFloat {
        guard isActive else { return minHeight }

        let time = date.timeIntervalSince1970
        let level = audioMeter.averagePower
        let amplitude = max(0, min(1, level))

        // Boost lower levels for better visibility
        let boosted = pow(amplitude, 0.7)

        // Wave calculation
        let wave = sin(time * 8 + phases[index]) * 0.5 + 0.5
        let centerDistance = abs(Double(index) - Double(barCount) / 2) / Double(barCount / 2)
        let centerBoost = 1.0 - (centerDistance * 0.4)

        let height = minHeight + CGFloat(boosted * wave * centerBoost) * (maxHeight - minHeight)
        return max(minHeight, height)
    }
}

struct StaticVisualizer: View {
    private let barCount: Int
    private let barWidth: CGFloat
    private let staticHeight: CGFloat
    private let barSpacing: CGFloat
    private let opacity: Double
    let color: Color

    init(
        color: Color,
        barCount: Int = 15,
        barWidth: CGFloat = 3,
        staticHeight: CGFloat = 4,
        barSpacing: CGFloat = 2,
        opacity: Double = 0.5
    ) {
        self.color = color
        self.barCount = barCount
        self.barWidth = barWidth
        self.staticHeight = staticHeight
        self.barSpacing = barSpacing
        self.opacity = opacity
    }

    var body: some View {
        HStack(spacing: barSpacing) {
            ForEach(0..<barCount, id: \.self) { _ in
                RoundedRectangle(cornerRadius: barWidth / 2)
                    .fill(color.opacity(opacity))
                    .frame(width: barWidth, height: staticHeight)
            }
        }
    }
}

// MARK: - Processing Status Display
struct ProcessingStatusDisplay: View {
    enum Mode {
        case transcribing
    }

    let mode: Mode
    let color: Color
    let isCompact: Bool

    init(mode: Mode, color: Color, isCompact: Bool = false) {
        self.mode = mode
        self.color = color
        self.isCompact = isCompact
    }

    private var label: String {
        "Transcribing"
    }

    private var animationSpeed: Double {
        0.18
    }

    var body: some View {
        Group {
            if isCompact {
                ProgressAnimation(color: color.opacity(0.8), animationSpeed: animationSpeed)
            } else {
                VStack(spacing: 4) {
                    Text(label)
                        .foregroundColor(color)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)

                    ProgressAnimation(color: color, animationSpeed: animationSpeed)
                }
            }
        }
        .frame(height: isCompact ? 18 : 28)
    }
}
