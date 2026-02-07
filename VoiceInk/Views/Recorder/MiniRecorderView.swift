import SwiftUI

struct MiniRecorderView: View {
    @ObservedObject var whisperState: WhisperState
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager
    @EnvironmentObject private var enhancementService: AIEnhancementService

    @State private var activePopover: ActivePopoverState = .none

    // MARK: - Design Constants
    private let mainContentHeight: CGFloat = 40
    private let compactWidth: CGFloat = 184
    private let expandedWidth: CGFloat = 300
    private let compactCornerRadius: CGFloat = 20
    private let expandedCornerRadius: CGFloat = 12

    private var hasPartialText: Bool {
        whisperState.recordingState == .recording && !whisperState.partialTranscript.isEmpty
    }

    private var currentWidth: CGFloat {
        hasPartialText ? expandedWidth : compactWidth
    }

    private var currentCornerRadius: CGFloat {
        hasPartialText ? expandedCornerRadius : compactCornerRadius
    }

    private var contentLayout: some View {
        HStack(spacing: 0) {
            RecorderPromptButton(
                activePopover: $activePopover,
                buttonSize: 22,
                padding: EdgeInsets()
            )
            .padding(.leading, 12)

            Spacer(minLength: 0)

            RecorderStatusDisplay(
                currentState: whisperState.recordingState,
                audioMeter: recorder.audioMeter
            )

            Spacer(minLength: 0)

            RecorderPowerModeButton(
                activePopover: $activePopover,
                buttonSize: 22,
                padding: EdgeInsets()
            )
            .padding(.trailing, 12)
        }
        .frame(height: mainContentHeight)
    }

    private var partialTextSection: some View {
        // Polls at 10Hz to detect transcript updates
        TimelineView(.animation(minimumInterval: 0.1)) { _ in
            VStack(spacing: 0) {
                Divider()
                    .background(Color.white.opacity(0.15))

                Text(whisperState.partialTranscript)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.8))
                    .lineLimit(1)
                    .truncationMode(.head)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 5)
            }
            .opacity(hasPartialText ? 1 : 0)
            .frame(height: hasPartialText ? nil : 0)
            .clipped()
        }
    }

    var body: some View {
        if windowManager.isVisible {
            VStack(spacing: 0) {
                contentLayout

                partialTextSection
            }
            .frame(width: currentWidth)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: currentCornerRadius, style: .continuous))
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .animation(.snappy(duration: 0.35), value: currentWidth)
            .animation(.snappy(duration: 0.35), value: currentCornerRadius)
        }
    }
}

