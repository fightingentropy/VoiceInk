import SwiftUI
import LaunchAtLogin

struct MenuBarView: View {
    @EnvironmentObject var engine: VoiceInkEngine
    @EnvironmentObject var transcriptionModelManager: TranscriptionModelManager
    @EnvironmentObject var recorderUIManager: RecorderUIManager
    @EnvironmentObject var menuBarManager: MenuBarManager
    @EnvironmentObject var enhancementService: AIEnhancementService
    @State private var launchAtLoginEnabled = LaunchAtLogin.isEnabled
    
    var body: some View {
        VStack {
            Button("Toggle Recorder") {
                recorderUIManager.handleToggleMiniRecorder()
            }

            Divider()

            Button("Retry Last Transcription") {
                LastTranscriptionService.retryLastTranscription(
                    from: engine.modelContext,
                    transcriptionModelManager: transcriptionModelManager,
                    serviceRegistry: engine.serviceRegistry,
                    enhancementService: enhancementService
                )
            }

            Button("Copy Last Transcription") {
                LastTranscriptionService.copyLastTranscription(from: engine.modelContext)
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])
            
            Button("History") {
                menuBarManager.openHistoryWindow()
            }
            .keyboardShortcut("h", modifiers: [.command, .shift])
            
            Button("Settings") {
                menuBarManager.openMainWindowAndNavigate(to: "Settings")
            }
            .keyboardShortcut(",", modifiers: .command)
            
            Button(menuBarManager.isMenuBarOnly ? "Show Dock Icon" : "Hide Dock Icon") {
                menuBarManager.toggleMenuBarOnly()
            }
            .keyboardShortcut("d", modifiers: [.command, .shift])
            
            Toggle("Launch at Login", isOn: $launchAtLoginEnabled)
                .onChange(of: launchAtLoginEnabled) { oldValue, newValue in
                    LaunchAtLogin.isEnabled = newValue
                }
            
            Divider()

            Button("Quit VoiceInk") {
                NSApplication.shared.terminate(nil)
            }
        }
    }
}
