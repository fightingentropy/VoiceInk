import SwiftUI

struct MiniRecorderView: View {
    @ObservedObject var whisperState: WhisperState
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager
    @State private var showPowerModePopover = false
    @State private var showEnhancementPromptPopover = false
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    
    @EnvironmentObject private var enhancementService: AIEnhancementService
    
    var body: some View {
        Group {
            if windowManager.isVisible {
                Capsule()
                    .fill(.clear)
                    .background(
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
                    )
                    .overlay {
                        Capsule()
                            .strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5)
                    }
                    .overlay {
                        HStack(spacing: 0) {
                            NotchRecordButton(
                                isRecording: whisperState.isRecording,
                                isProcessing: whisperState.isProcessing
                            ) {
                                Task { await whisperState.toggleRecord() }
                            }
                            .frame(width: 24)
                            .padding(.leading, 8)
                            
                            Group {
                                if whisperState.isEnhancing {
                                    Text("Enhancing")
                                        .foregroundColor(.white)
                                        .font(.system(size: 10, weight: .medium, design: .default))
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.5)
                                } else if whisperState.isTranscribing {
                                    Text("Transcribing")
                                        .foregroundColor(.white)
                                        .font(.system(size: 10, weight: .medium, design: .default))
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.5)
                                } else if whisperState.isRecording {
                                    AudioVisualizer(
                                        audioMeter: recorder.audioMeter,
                                        color: .white,
                                        isActive: whisperState.isRecording
                                    )
                                } else {
                                    StaticVisualizer(color: .white)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.horizontal, 8)
                            
                            if powerModeManager.isPowerModeEnabled {
                                NotchToggleButton(
                                    isEnabled: powerModeManager.isPowerModeEnabled,
                                    icon: powerModeManager.currentActiveConfiguration.emoji,
                                    color: .orange,
                                    disabled: false
                                ) {
                                    showPowerModePopover.toggle()
                                }
                                .frame(width: 24)
                                .padding(.trailing, 8)
                                .popover(isPresented: $showPowerModePopover, arrowEdge: .bottom) {
                                    PowerModePopover()
                                }
                            } else {
                                NotchToggleButton(
                                    isEnabled: enhancementService.isEnhancementEnabled,
                                    icon: enhancementService.activePrompt?.icon.rawValue ?? "brain",
                                    color: .blue,
                                    disabled: false
                                ) {
                                    if enhancementService.isEnhancementEnabled {
                                        showEnhancementPromptPopover.toggle()
                                    } else {
                                        enhancementService.isEnhancementEnabled = true
                                    }
                                }
                                .frame(width: 24)
                                .padding(.trailing, 8)
                                .popover(isPresented: $showEnhancementPromptPopover, arrowEdge: .bottom) {
                                    EnhancementPromptPopover()
                                        .environmentObject(enhancementService)
                                }
                            }
                        }
                        .padding(.vertical, 8)
                    }
                    .opacity(windowManager.isVisible ? 1 : 0)
            }
        }
    }
}


