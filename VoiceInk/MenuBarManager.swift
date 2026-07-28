import SwiftUI
import AppKit
import os

@MainActor
class MenuBarManager: ObservableObject {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "MenuBarManager")

    init() {
        updateAppActivationPolicy()

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowDidClose),
            name: NSWindow.willCloseNotification,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    @objc private func windowDidClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              window.identifier != nil else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let hasVisibleWindows = NSApplication.shared.windows.contains {
                $0.isVisible && $0.level == .normal && !$0.styleMask.contains(.nonactivatingPanel)
            }
            if !hasVisibleWindows {
                NSApplication.shared.setActivationPolicy(.accessory)
            }
        }
    }

    func applyActivationPolicy() {
        updateAppActivationPolicy()
    }
    
    func focusMainWindow() {
        if WindowManager.shared.showMainWindow() == nil {
            logger.warning("Unable to locate main window to focus")
        }
    }
    
    private func updateAppActivationPolicy() {
        NSApplication.shared.setActivationPolicy(.accessory)
    }
    
    func openMainWindowAndNavigate(to destination: String) {
        logger.debug("Navigating to \(destination, privacy: .public)")

        guard WindowManager.shared.showMainWindow() != nil else {
            logger.warning("Unable to show main window for navigation")
            return
        }

        // Post a notification to navigate to the desired destination
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [logger] in
            NotificationCenter.default.post(
                name: .navigateToDestination,
                object: nil,
                userInfo: ["destination": destination]
            )
            logger.debug("Posted navigation notification for \(destination, privacy: .public)")
        }
    }

}
