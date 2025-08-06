import SwiftUI

struct MiniRecorderView: View {
    @ObservedObject var whisperState: WhisperState
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager
    @EnvironmentObject private var enhancementService: AIEnhancementService
    
    @State private var showPowerModePopover = false
    @State private var showEnhancementPromptPopover = false
    @State private var isHovering = false
    
    private var backgroundView: some View {
        ZStack {
            Color.black.opacity(0.9)
            LinearGradient(
                colors: [
                    Color.black.opacity(0.95),
                    Color(red: 0.15, green: 0.15, blue: 0.15).opacity(0.9)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            VisualEffectView(material: .hudWindow, blendingMode: .withinWindow)
                .opacity(0.05)
        }
        .clipShape(Capsule())
    }
    
    private var statusView: some View {
        RecorderStatusDisplay(
            currentState: whisperState.recordingState,
            audioMeter: recorder.audioMeter
        )
    }
    
    var body: some View {
        Group {
            if windowManager.isVisible {
                Capsule()
                    .fill(.clear)
                    .background(backgroundView)
                    .overlay {
                        Capsule()
                            .strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5)
                    }
                    .overlay {
                        HStack(spacing: 0) {
                            // Left button zone
                            Group {
                                if windowManager.isExpanded {
                                    RecorderPromptButton(showPopover: $showEnhancementPromptPopover)
                                        .transition(.scale(scale: 0.5).combined(with: .opacity))
                                }
                            }
                            .frame(width: windowManager.isExpanded ? nil : 0)
                            .frame(maxWidth: windowManager.isExpanded ? .infinity : 0)
                            .clipped()
                            .opacity(windowManager.isExpanded ? 1 : 0)
                            .animation(.easeInOut(duration: 0.25), value: windowManager.isExpanded)
                            
                            if windowManager.isExpanded {
                                Spacer()
                            }
                            
                            // Fixed visualizer zone  
                            statusView
                                .frame(maxWidth: .infinity)
                            
                            if windowManager.isExpanded {
                                Spacer()
                            }
                            
                            // Right button zone
                            Group {
                                if windowManager.isExpanded {
                                    RecorderPowerModeButton(showPopover: $showPowerModePopover)
                                        .transition(.scale(scale: 0.5).combined(with: .opacity))
                                }
                            }
                            .frame(width: windowManager.isExpanded ? nil : 0)
                            .frame(maxWidth: windowManager.isExpanded ? .infinity : 0)
                            .clipped()
                            .opacity(windowManager.isExpanded ? 1 : 0)
                            .animation(.easeInOut(duration: 0.25), value: windowManager.isExpanded)
                        }
                        .padding(.vertical, 8)
                    }
                    .onHover { hovering in
                        isHovering = hovering
                        if hovering {
                            windowManager.expand()
                        } else {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                                if !isHovering {
                                    windowManager.collapse()
                                }
                            }
                        }
                    }
            }
        }
    }
}


