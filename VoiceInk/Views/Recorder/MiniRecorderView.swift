import SwiftUI

struct MiniRecorderView<S: RecorderStateProvider & ObservableObject>: View {
    @ObservedObject var stateProvider: S
    let recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager

    // MARK: - Design Constants
    private let mainContentHeight: CGFloat = 32
    private let width: CGFloat = 320
    private let cornerRadius: CGFloat = 16

    private var contentLayout: some View {
        HStack(spacing: 0) {
            RecorderStatusDisplay(
                currentState: stateProvider.recordingState,
                partialTranscript: stateProvider.partialTranscript,
                style: .compact
            )
        }
        .frame(height: mainContentHeight)
        .padding(.horizontal, 14)
    }

    private var recorderPill: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)

        return contentLayout
            .frame(width: width)
            .background(MonochromeStyle.subtleFill, in: shape)
            .overlay(shape.stroke(MonochromeStyle.hairline, lineWidth: 0.75))
    }

    var body: some View {
        if windowManager.isVisible {
            recorderPill
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
    }
}
