import SwiftUI

/// A compact voice waveform driven by the microphone meter while recording.
/// During final transcription it settles into a subtle travelling pulse, so the
/// same visual communicates both states without labels or transcript text.
struct AudioVisualizer: View {
    let audioMeter: AudioMeter
    let color: Color
    let isActive: Bool
    let isProcessing: Bool

    private let barCount: Int
    private let barWidth: CGFloat
    private let barSpacing: CGFloat
    private let minHeight: CGFloat
    private let maxHeight: CGFloat
    private let opacity: Double

    init(
        audioMeter: AudioMeter,
        color: Color,
        isActive: Bool,
        isProcessing: Bool = false,
        barCount: Int = 19,
        barWidth: CGFloat = 2.25,
        barSpacing: CGFloat = 2.25,
        minHeight: CGFloat = 2.5,
        maxHeight: CGFloat = 20,
        opacity: Double = 0.92
    ) {
        self.audioMeter = audioMeter
        self.color = color
        self.isActive = isActive
        self.isProcessing = isProcessing
        self.barCount = barCount
        self.barWidth = barWidth
        self.barSpacing = barSpacing
        self.minHeight = minHeight
        self.maxHeight = maxHeight
        self.opacity = opacity
    }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { context in
            HStack(spacing: barSpacing) {
                ForEach(0..<barCount, id: \.self) { index in
                    Capsule(style: .continuous)
                        .fill(color.opacity(barOpacity(for: index, at: context.date)))
                        .frame(
                            width: barWidth,
                            height: barHeight(for: index, at: context.date)
                        )
                }
            }
            .frame(maxHeight: maxHeight)
        }
        .animation(.snappy(duration: 0.12), value: audioMeter)
    }

    private func barHeight(for index: Int, at date: Date) -> CGFloat {
        if isProcessing {
            return processingHeight(for: index, at: date)
        }

        guard isActive else { return minHeight }

        let average = clamped(audioMeter.averagePower)
        let peak = clamped(audioMeter.peakPower)
        let boostedLevel = pow((average * 0.72) + (peak * 0.28), 0.62)
        let position = normalizedPosition(for: index)

        // A soft centre envelope makes the meter read as one coherent waveform.
        let envelope = 0.58 + (1 - abs(position)) * 0.42
        let time = date.timeIntervalSinceReferenceDate
        let detail = 0.72
            + 0.18 * sin(time * 12 + Double(index) * 1.37)
            + 0.10 * sin(time * 7 - Double(index) * 0.81)
        let amplitude = boostedLevel * envelope * max(0.42, detail)

        return minHeight + CGFloat(amplitude) * (maxHeight - minHeight)
    }

    private func processingHeight(for index: Int, at date: Date) -> CGFloat {
        let time = date.timeIntervalSinceReferenceDate
        let position = normalizedPosition(for: index)
        let travellingWave = (sin(time * 5.5 - position * 5.2) + 1) / 2
        let centreEnvelope = 0.65 + (1 - abs(position)) * 0.35
        let amplitude = 0.12 + travellingWave * centreEnvelope * 0.26

        return minHeight + CGFloat(amplitude) * (maxHeight - minHeight)
    }

    private func barOpacity(for index: Int, at date: Date) -> Double {
        guard isProcessing else {
            return isActive ? opacity : opacity * 0.45
        }

        let position = normalizedPosition(for: index)
        let pulse = (sin(date.timeIntervalSinceReferenceDate * 5.5 - position * 5.2) + 1) / 2
        return opacity * (0.50 + pulse * 0.42)
    }

    private func normalizedPosition(for index: Int) -> Double {
        guard barCount > 1 else { return 0 }
        return (Double(index) / Double(barCount - 1)) * 2 - 1
    }

    private func clamped(_ value: Double) -> Double {
        min(max(value, 0), 1)
    }
}
