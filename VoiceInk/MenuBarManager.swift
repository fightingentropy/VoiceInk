import SwiftUI
import SwiftData
import AppKit
import os

@MainActor
class MenuBarManager: ObservableObject {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "MenuBarManager")
    @Published var isMenuBarOnly: Bool {
        didSet {
            UserDefaults.standard.set(isMenuBarOnly, forKey: "IsMenuBarOnly")
            updateAppActivationPolicy()
        }
    }

    private var modelContainer: ModelContainer?
    private var engine: VoiceInkEngine?

    init() {
        self.isMenuBarOnly = UserDefaults.standard.bool(forKey: "IsMenuBarOnly")
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
        guard isMenuBarOnly else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let hasVisibleWindows = NSApplication.shared.windows.contains {
                $0.isVisible && $0.level == .normal && !$0.styleMask.contains(.nonactivatingPanel)
            }
            if !hasVisibleWindows {
                NSApplication.shared.setActivationPolicy(.accessory)
            }
        }
    }

    func configure(modelContainer: ModelContainer, engine: VoiceInkEngine) {
        self.modelContainer = modelContainer
        self.engine = engine
    }
    
    func toggleMenuBarOnly() {
        isMenuBarOnly.toggle()
    }
    
    func applyActivationPolicy() {
        updateAppActivationPolicy()
    }
    
    func focusMainWindow() {
        NSApplication.shared.setActivationPolicy(.regular)
        if WindowManager.shared.showMainWindow() == nil {
            logger.warning("Unable to locate main window to focus")
        }
    }
    
    private func updateAppActivationPolicy() {
        let application = NSApplication.shared
        if isMenuBarOnly {
            application.setActivationPolicy(.accessory)
            WindowManager.shared.hideMainWindow()
        } else {
            application.setActivationPolicy(.regular)
            _ = WindowManager.shared.showMainWindow()
        }
    }
    
    func openMainWindowAndNavigate(to destination: String) {
        logger.debug("Navigating to \(destination, privacy: .public)")

        NSApplication.shared.setActivationPolicy(.regular)

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

    func openHistoryWindow() {
        guard let modelContainer = modelContainer,
              let engine = engine else {
            logger.error("Dependencies not configured")
            return
        }
        NSApplication.shared.setActivationPolicy(.regular)
        HistoryWindowController.shared.showHistoryWindow(
            modelContainer: modelContainer,
            engine: engine
        )
    }
}
