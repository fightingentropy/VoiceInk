import SwiftUI

struct MiniRecorderView<S: RecorderStateProvider & ObservableObject>: View {
    @ObservedObject var stateProvider: S
    let recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager

    // MARK: - Design Constants
    private let collapsedWidth: CGFloat = 48
    private let expandedWidth: CGFloat = 320
    private let collapsedHeight: CGFloat = 20
    private let expandedHeight: CGFloat = 32

    private var hasLiveTranscript: Bool {
        stateProvider.recordingState == .recording
            && !stateProvider.partialTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var pillWidth: CGFloat {
        hasLiveTranscript ? expandedWidth : collapsedWidth
    }

    private var pillHeight: CGFloat {
        hasLiveTranscript ? expandedHeight : collapsedHeight
    }

    private var contentLayout: some View {
        HStack(spacing: 0) {
            RecorderStatusDisplay(
                currentState: stateProvider.recordingState,
                partialTranscript: stateProvider.partialTranscript,
                style: .compact
            )
        }
        .frame(height: pillHeight)
        .padding(.horizontal, hasLiveTranscript ? 14 : 0)
    }

    private var recorderPill: some View {
        let shape = RoundedRectangle(cornerRadius: pillHeight / 2, style: .continuous)

        return contentLayout
            .frame(width: pillWidth, height: pillHeight)
            .background(Color(red: 0.025, green: 0.025, blue: 0.03).opacity(0.98), in: shape)
            .overlay(shape.stroke(.white.opacity(0.14), lineWidth: 0.75))
            .shadow(color: .black.opacity(0.48), radius: 8, y: 3)
    }

    var body: some View {
        if windowManager.isVisible {
            recorderPill
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .animation(.spring(response: 0.32, dampingFraction: 0.86), value: hasLiveTranscript)
        }
    }
}
