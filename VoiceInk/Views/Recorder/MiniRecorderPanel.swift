import SwiftUI
import AppKit

class MiniRecorderPanel: NSPanel {
    static let initialContentSize = CGSize(width: 64, height: 36)

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
    
    init(contentRect: NSRect) {
        super.init(
            contentRect: contentRect,
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        configurePanel()
    }
    
    private func configurePanel() {
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        isMovable = true
        isMovableByWindowBackground = true
        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        titlebarAppearsTransparent = true
        titleVisibility = .hidden
        standardWindowButton(.closeButton)?.isHidden = true
    }
    
    static func calculateWindowMetrics(
        contentSize: CGSize = initialContentSize,
        screen: NSScreen? = NSScreen.main
    ) -> NSRect {
        guard let screen else {
            return NSRect(origin: .zero, size: contentSize)
        }

        let padding: CGFloat = 24

        let visibleFrame = screen.visibleFrame
        let centerX = visibleFrame.midX
        let xPosition = centerX - (contentSize.width / 2)
        let yPosition = visibleFrame.minY + padding

        return NSRect(
            x: xPosition,
            y: yPosition,
            width: contentSize.width,
            height: contentSize.height
        )
    }

    func show() {
        orderFrontRegardless()
    }

    func resize(to requestedSize: CGSize) {
        guard let screen = screen ?? NSScreen.main else { return }

        let visibleFrame = screen.visibleFrame
        let width = min(requestedSize.width, visibleFrame.width - 32)
        let height = min(requestedSize.height, visibleFrame.height - 32)
        let currentCenterX = frame.midX
        let currentMinY = frame.minY
        let x = min(
            max(currentCenterX - (width / 2), visibleFrame.minX + 16),
            visibleFrame.maxX - width - 16
        )
        let y = min(
            max(currentMinY, visibleFrame.minY + 16),
            visibleFrame.maxY - height - 16
        )
        let targetFrame = NSRect(x: x, y: y, width: width, height: height)

        guard abs(frame.width - width) > 0.5 || abs(frame.height - height) > 0.5 else {
            return
        }
        setFrame(targetFrame, display: true)
    }
    
    func hide(completion: @MainActor @escaping () -> Void) {
        completion()
    }
}
