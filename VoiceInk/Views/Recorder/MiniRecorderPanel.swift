import SwiftUI
import AppKit

class MiniRecorderPanel: NSPanel {
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
    
    static func calculateWindowMetrics(expanded: Bool = false) -> NSRect {
        guard let screen = NSScreen.main else {
            return NSRect(x: 0, y: 0, width: expanded ? 160 : 70, height: 34)
        }
        
        let compactWidth: CGFloat = 100
        let expandedWidth: CGFloat = 160
        let width = expanded ? expandedWidth : compactWidth
        let height: CGFloat = 34
        let padding: CGFloat = 24
        
        let visibleFrame = screen.visibleFrame
        let centerX = visibleFrame.midX - 5
        let xPosition = centerX - (width / 2)
        let yPosition = visibleFrame.minY + padding
        
        return NSRect(
            x: xPosition,
            y: yPosition,
            width: width,
            height: height
        )
    }
    
    func show() {
        let metrics = MiniRecorderPanel.calculateWindowMetrics(expanded: false)
        setFrame(metrics, display: true)
        orderFrontRegardless()
    }
    
    func expandWindow(completion: (() -> Void)? = nil) {
        let expandedMetrics = MiniRecorderPanel.calculateWindowMetrics(expanded: true)
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.25
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            animator().setFrame(expandedMetrics, display: true)
        }, completionHandler: completion)
    }
    
    func collapseWindow(completion: (() -> Void)? = nil) {
        let compactMetrics = MiniRecorderPanel.calculateWindowMetrics(expanded: false)
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.25
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            animator().setFrame(compactMetrics, display: true)
        }, completionHandler: completion)
    }
    
    func hide(completion: @escaping () -> Void) {
        completion()
    }
} 