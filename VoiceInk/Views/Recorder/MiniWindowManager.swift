import SwiftUI
import AppKit

class MiniWindowManager: ObservableObject {
    @Published var isVisible = false
    @Published var isExpanded = false
    private var windowController: NSWindowController?
    private var miniPanel: MiniRecorderPanel?
    private let whisperState: WhisperState
    private let recorder: Recorder
    
    // Callback to check if collapse should be prevented (e.g., when popovers are showing)
    var shouldPreventCollapse: (() -> Bool)?
    
    // Debounced timer for auto-collapse
    private var debounceTimer: Timer?
    
    init(whisperState: WhisperState, recorder: Recorder) {
        self.whisperState = whisperState
        self.recorder = recorder
        setupNotifications()
    }
    
    deinit {
        debounceTimer?.invalidate()
        NotificationCenter.default.removeObserver(self)
    }
    
    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleHideNotification),
            name: NSNotification.Name("HideMiniRecorder"),
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleFeedbackNotification),
            name: .promptSelectionChanged,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleFeedbackNotification),
            name: .powerModeConfigurationApplied,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleFeedbackNotification),
            name: .enhancementToggleChanged,
            object: nil
        )
    }
    
    @objc private func handleHideNotification() {
        hide()
    }
    
    @objc private func handleFeedbackNotification() {
        guard isVisible else { return }
        
        // Only expand if not already expanded
        if !isExpanded {
            expand()
        }
        
        // Reset debounce timer - this cancels any existing timer and starts a new one
        debounceTimer?.invalidate()
        debounceTimer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { [weak self] _ in
            guard let self = self else { return }
            if self.isExpanded && !(self.shouldPreventCollapse?() ?? false) {
                self.collapse()
            }
        }
    }
    func show() {
        if isVisible { return }
        
        let activeScreen = NSApp.keyWindow?.screen ?? NSScreen.main ?? NSScreen.screens[0]
        
        initializeWindow(screen: activeScreen)
        self.isVisible = true
        miniPanel?.show()
    }
    
    func expand() {
        guard isVisible, !isExpanded else { return }
        
        withAnimation(.easeInOut(duration: 0.25)) {
            isExpanded = true
        }
        
        miniPanel?.expandWindow()
    }
    
    func collapse() {
        guard isVisible, isExpanded else { return }
        
        withAnimation(.easeInOut(duration: 0.25)) {
            isExpanded = false
        }
        
        miniPanel?.collapseWindow()
    }
    
    func hide() {
        guard isVisible else { return }
        
        // Cancel any pending auto-collapse timer
        debounceTimer?.invalidate()
        debounceTimer = nil
        
        self.isVisible = false
        self.isExpanded = false  
        self.miniPanel?.hide { [weak self] in
            guard let self = self else { return }
            self.deinitializeWindow()
        }
    }
    
    private func initializeWindow(screen: NSScreen) {
        deinitializeWindow()
        
        let metrics = MiniRecorderPanel.calculateWindowMetrics()
        let panel = MiniRecorderPanel(contentRect: metrics)
        
        let miniRecorderView = MiniRecorderView(whisperState: whisperState, recorder: recorder)
            .environmentObject(self)
            .environmentObject(whisperState.enhancementService!)
        
        let hostingController = NSHostingController(rootView: miniRecorderView)
        panel.contentView = hostingController.view
        
        self.miniPanel = panel
        self.windowController = NSWindowController(window: panel)
        
        panel.orderFrontRegardless()
    }
    
    private func deinitializeWindow() {
        windowController?.close()
        windowController = nil
        miniPanel = nil
    }
    
    func toggle() {
        if isVisible {
            hide()
        } else {
            show()
        }
    }
} 
