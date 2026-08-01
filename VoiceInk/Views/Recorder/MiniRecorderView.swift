import SwiftUI
import AppKit

struct MiniRecorderPillLayout {
    static let collapsedSize = CGSize(width: 48, height: 20)
    static let expandedHeight: CGFloat = 32
    static let maximumWidth: CGFloat = 720

    private static let horizontalPadding: CGFloat = 28
    private static let caretAllowance: CGFloat = 5
    private static let verticalPadding: CGFloat = 12

    static func size(
        for transcript: String,
        isRecording: Bool,
        maximumWidth: CGFloat = maximumWidth
    ) -> CGSize {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isRecording, !trimmed.isEmpty else {
            return collapsedSize
        }

        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12, weight: .medium)
        ]
        let attributedText = NSAttributedString(string: trimmed, attributes: attributes)
        let totalHorizontalInset = horizontalPadding + caretAllowance
        let constrainedMaximumWidth = max(collapsedSize.width, maximumWidth)
        let maximumTextWidth = max(1, constrainedMaximumWidth - totalHorizontalInset)
        let singleLineBounds = attributedText.boundingRect(
            with: CGSize(width: 100_000, height: 100_000),
            options: [.usesLineFragmentOrigin, .usesFontLeading]
        )
        let textWidth = min(ceil(singleLineBounds.width), maximumTextWidth)
        let pillWidth = max(collapsedSize.width, textWidth + totalHorizontalInset)
        let wrappedTextBounds = attributedText.boundingRect(
            with: CGSize(width: max(1, pillWidth - totalHorizontalInset), height: 100_000),
            options: [.usesLineFragmentOrigin, .usesFontLeading]
        )
        let pillHeight = max(expandedHeight, ceil(wrappedTextBounds.height) + verticalPadding)

        return CGSize(width: pillWidth, height: pillHeight)
    }
}

struct MiniRecorderView<S: RecorderStateProvider & ObservableObject>: View {
    @ObservedObject var stateProvider: S
    let recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager

    private let windowPadding: CGFloat = 16

    private var hasLiveTranscript: Bool {
        stateProvider.recordingState == .recording
            && !stateProvider.partialTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func contentLayout(pillSize: CGSize) -> some View {
        HStack(spacing: 0) {
            RecorderStatusDisplay(
                currentState: stateProvider.recordingState,
                partialTranscript: stateProvider.partialTranscript,
                style: .compact
            )
        }
        .frame(height: pillSize.height)
        .padding(.horizontal, hasLiveTranscript ? 14 : 0)
    }

    private func recorderPill(pillSize: CGSize) -> some View {
        let shape = RoundedRectangle(cornerRadius: pillSize.height / 2, style: .continuous)

        return contentLayout(pillSize: pillSize)
            .frame(width: pillSize.width, height: pillSize.height)
            .background(Color(red: 0.025, green: 0.025, blue: 0.03).opacity(0.98), in: shape)
            .overlay(shape.stroke(.white.opacity(0.14), lineWidth: 0.75))
            .shadow(color: .black.opacity(0.48), radius: 8, y: 3)
    }

    var body: some View {
        if windowManager.isVisible {
            let pillSize = MiniRecorderPillLayout.size(
                for: stateProvider.partialTranscript,
                isRecording: stateProvider.recordingState == .recording
            )
            let panelSize = CGSize(
                width: pillSize.width + windowPadding,
                height: pillSize.height + windowPadding
            )

            recorderPill(pillSize: pillSize)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .animation(.spring(response: 0.28, dampingFraction: 0.88), value: pillSize)
                .onAppear {
                    windowManager.resize(to: panelSize)
                }
                .onChange(of: panelSize) { _, newSize in
                    windowManager.resize(to: newSize)
                }
        }
    }
}
