import SwiftUI

struct MiniRecorderWaveformLayout {
    static let pillSize = CGSize(width: 104, height: 34)
    static let windowPadding: CGFloat = 16

    static var panelSize: CGSize {
        CGSize(
            width: pillSize.width + windowPadding,
            height: pillSize.height + windowPadding
        )
    }
}

struct MiniRecorderView<S: RecorderStateProvider & ObservableObject>: View {
    @ObservedObject var stateProvider: S
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager

    private var isRecording: Bool {
        stateProvider.recordingState == .recording
    }

    private var isProcessing: Bool {
        stateProvider.recordingState == .transcribing
    }

    private var recorderPill: some View {
        let shape = Capsule(style: .continuous)

        return AudioVisualizer(
            audioMeter: recorder.audioMeter,
            color: .white,
            isActive: isRecording,
            isProcessing: isProcessing,
            barCount: 19,
            barWidth: 2.25,
            barSpacing: 2.25,
            minHeight: 2.5,
            maxHeight: 20,
            opacity: 0.92
        )
        .frame(width: MiniRecorderWaveformLayout.pillSize.width - 24, height: 22)
        .frame(
            width: MiniRecorderWaveformLayout.pillSize.width,
            height: MiniRecorderWaveformLayout.pillSize.height
        )
        .background {
            shape
                .fill(.regularMaterial)
                .overlay(shape.fill(Color.black.opacity(0.20)))
        }
        .clipShape(shape)
        .overlay(shape.stroke(.white.opacity(0.13), lineWidth: 0.75))
        .shadow(color: .black.opacity(0.24), radius: 11, y: 4)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(isRecording ? "Listening" : "Transcribing")
        .accessibilityAddTraits(.updatesFrequently)
    }

    var body: some View {
        if windowManager.isVisible {
            recorderPill
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .onAppear {
                    windowManager.resize(to: MiniRecorderWaveformLayout.panelSize)
                }
        }
    }
}
