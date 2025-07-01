import Foundation
import KeyboardShortcuts
import Carbon
import AppKit

extension KeyboardShortcuts.Name {
    static let toggleMiniRecorder = Self("toggleMiniRecorder")
}

@MainActor
class HotkeyManager: ObservableObject {
    @Published var isListening = false
    @Published var isShortcutConfigured = false
    @Published var isPushToTalkEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isPushToTalkEnabled, forKey: "isPushToTalkEnabled")
            resetKeyStates()
            setupKeyMonitor()
        }
    }
    @Published var pushToTalkKey: PushToTalkKey {
        didSet {
            UserDefaults.standard.set(pushToTalkKey.rawValue, forKey: "pushToTalkKey")
            resetKeyStates()
        }
    }
    
    private var whisperState: WhisperState
    private var currentKeyState = false
    private var miniRecorderShortcutManager: MiniRecorderShortcutManager
    
    // Change from single monitor to separate local and global monitors
    private var globalEventMonitor: Any?
    private var localEventMonitor: Any?
    
    // Key handling properties
    private var keyPressStartTime: Date?
    private let briefPressThreshold = 1.0 // 1 second threshold for brief press
    private var isHandsFreeMode = false   // Track if we're in hands-free recording mode

    // Add cooldown management
    private var lastShortcutTriggerTime: Date?
    private let shortcutCooldownInterval: TimeInterval = 0.5 // 500ms cooldown
    
    private var fnDebounceTask: Task<Void, Never>?
    private var pendingFnKeyState: Bool? = nil
    
    enum PushToTalkKey: String, CaseIterable {
        case rightOption = "rightOption"
        case leftOption = "leftOption"
        case leftControl = "leftControl"
        case rightControl = "rightControl"
        case fn = "fn"
        case rightCommand = "rightCommand"
        case rightShift = "rightShift"
        
        var displayName: String {
            switch self {
            case .rightOption: return "Right Option (⌥)"
            case .leftOption: return "Left Option (⌥)"
            case .leftControl: return "Left Control (⌃)"
            case .rightControl: return "Right Control (⌃)"
            case .fn: return "Fn"
            case .rightCommand: return "Right Command (⌘)"
            case .rightShift: return "Right Shift (⇧)"
            }
        }
        
        var keyCode: CGKeyCode {
            switch self {
            case .rightOption: return 0x3D
            case .leftOption: return 0x3A
            case .leftControl: return 0x3B
            case .rightControl: return 0x3E
            case .fn: return 0x3F
            case .rightCommand: return 0x36
            case .rightShift: return 0x3C
            }
        }
    }
    
    init(whisperState: WhisperState) {
        self.isPushToTalkEnabled = UserDefaults.standard.bool(forKey: "isPushToTalkEnabled")
        self.pushToTalkKey = PushToTalkKey(rawValue: UserDefaults.standard.string(forKey: "pushToTalkKey") ?? "") ?? .rightCommand
        self.whisperState = whisperState
        self.miniRecorderShortcutManager = MiniRecorderShortcutManager(whisperState: whisperState)
        
        updateShortcutStatus()
    }
    
    private func resetKeyStates() {
        currentKeyState = false
        keyPressStartTime = nil
        isHandsFreeMode = false
    }
    
    private func setupKeyMonitor() {
        removeKeyMonitor()
        
        guard isPushToTalkEnabled else { return }
        
        // Global monitor for capturing flags when app is in background
        globalEventMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            guard let self = self else { return }
            
            Task { @MainActor in
                await self.handleNSKeyEvent(event)
            }
        }
        
        // Local monitor for capturing flags when app has focus
        localEventMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            guard let self = self else { return event }
            
            Task { @MainActor in
                await self.handleNSKeyEvent(event)
            }
            
            return event // Return the event to allow normal processing
        }
    }
    
    private func removeKeyMonitor() {
        if let monitor = globalEventMonitor {
            NSEvent.removeMonitor(monitor)
            globalEventMonitor = nil
        }
        
        if let monitor = localEventMonitor {
            NSEvent.removeMonitor(monitor)
            localEventMonitor = nil
        }
    }
    
    private func handleNSKeyEvent(_ event: NSEvent) async {
        let keycode = event.keyCode
        let flags = event.modifierFlags
        
        // Check if the target key is pressed based on the modifier flags
        var isKeyPressed = false
        var isTargetKey = false
        
        switch pushToTalkKey {
        case .rightOption, .leftOption:
            isKeyPressed = flags.contains(.option)
            isTargetKey = keycode == pushToTalkKey.keyCode
        case .leftControl, .rightControl:
            isKeyPressed = flags.contains(.control)
            isTargetKey = keycode == pushToTalkKey.keyCode
        case .fn:
            isKeyPressed = flags.contains(.function)
            isTargetKey = keycode == pushToTalkKey.keyCode
            // Debounce only for Fn key
            if isTargetKey {
                pendingFnKeyState = isKeyPressed
                fnDebounceTask?.cancel()
                fnDebounceTask = Task { [pendingState = isKeyPressed] in
                    try? await Task.sleep(nanoseconds: 75_000_000) // 75ms
                    // Only act if the state hasn't changed during debounce
                    if pendingFnKeyState == pendingState {
                        await MainActor.run {
                            self.processPushToTalkKey(isKeyPressed: pendingState)
                        }
                    }
                }
                return
            }
        case .rightCommand:
            isKeyPressed = flags.contains(.command)
            isTargetKey = keycode == pushToTalkKey.keyCode
        case .rightShift:
            isKeyPressed = flags.contains(.shift)
            isTargetKey = keycode == pushToTalkKey.keyCode
        }
        
        guard isTargetKey else { return }
        processPushToTalkKey(isKeyPressed: isKeyPressed)
    }
    
    private func processPushToTalkKey(isKeyPressed: Bool) {
        guard isKeyPressed != currentKeyState else { return }
        currentKeyState = isKeyPressed
        
        // Key is pressed down
        if isKeyPressed {
            keyPressStartTime = Date()
            
            // If we're in hands-free mode, stop recording
            if isHandsFreeMode {
                isHandsFreeMode = false
                Task { @MainActor in await whisperState.handleToggleMiniRecorder() }
                return
            }
            
            // Show recorder if not already visible
            if !whisperState.isMiniRecorderVisible {
                Task { @MainActor in await whisperState.handleToggleMiniRecorder() }
            }
        } 
        // Key is released
        else {
            let now = Date()
            
            // Calculate press duration
            if let startTime = keyPressStartTime {
                let pressDuration = now.timeIntervalSince(startTime)
                
                if pressDuration < briefPressThreshold {
                    // For brief presses, enter hands-free mode
                    isHandsFreeMode = true
                    // Continue recording - do nothing on release
                } else {
                    // For longer presses, stop and transcribe
                    Task { @MainActor in await whisperState.handleToggleMiniRecorder() }
                }
            }
            
            keyPressStartTime = nil
        }
    }
    
    func updateShortcutStatus() {
        isShortcutConfigured = KeyboardShortcuts.getShortcut(for: .toggleMiniRecorder) != nil
        if isShortcutConfigured {
            setupShortcutHandler()
            setupKeyMonitor()
        } else {
            removeKeyMonitor()
        }
    }
    
    private func setupShortcutHandler() {
        KeyboardShortcuts.onKeyUp(for: .toggleMiniRecorder) { [weak self] in
            Task { @MainActor in
                await self?.handleShortcutTriggered()
            }
        }
    }
    
    private func handleShortcutTriggered() async {
        // Check cooldown
        if let lastTrigger = lastShortcutTriggerTime,
           Date().timeIntervalSince(lastTrigger) < shortcutCooldownInterval {
            return // Still in cooldown period
        }
        
        // Update last trigger time
        lastShortcutTriggerTime = Date()
        
        // Handle the shortcut
        await whisperState.handleToggleMiniRecorder()
    }
    
    deinit {
        Task { @MainActor in
            removeKeyMonitor()
        }
    }
}
