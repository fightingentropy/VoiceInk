import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject var recorderUIManager: RecorderUIManager
    @EnvironmentObject var menuBarManager: MenuBarManager

    var body: some View {
        VStack {
            Button("Toggle Recorder") {
                recorderUIManager.handleToggleMiniRecorder()
            }

            Divider()

            Button("Settings") {
                menuBarManager.openMainWindowAndNavigate(to: "Settings")
            }
            .keyboardShortcut(",", modifiers: .command)

            Divider()

            Button("Quit VoiceInk") {
                NSApplication.shared.terminate(nil)
            }
        }
    }
}
