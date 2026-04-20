import SwiftUI

struct MiniRecorderView<S: RecorderStateProvider & ObservableObject>: View {
    @ObservedObject var stateProvider: S
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager

    // MARK: - Design Constants
    private let mainContentHeight: CGFloat = 32
    private let width: CGFloat = 76
    private let cornerRadius: CGFloat = 16

    private var contentLayout: some View {
        HStack(spacing: 0) {
            RecorderStatusDisplay(
                currentState: stateProvider.recordingState,
                audioMeter: recorder.audioMeter,
                style: .compact
            )
        }
        .frame(height: mainContentHeight)
        .padding(.horizontal, 10)
    }

    var body: some View {
        if windowManager.isVisible {
            contentLayout
                .frame(width: width)
                .background(Color.black.opacity(0.9))
                .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
    }
}
