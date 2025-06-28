import Foundation
import KeyboardShortcuts
import Carbon
import AppKit

extension KeyboardShortcuts.Name {
    static let toggleMiniRecorder = Self("toggleMiniRecorder")
    static let escapeRecorder = Self("escapeRecorder")
    static let toggleEnhancement = Self("toggleEnhancement")
    // Power Mode selection shortcuts
    static let selectPowerMode1 = Self("selectPowerMode1")
    static let selectPowerMode2 = Self("selectPowerMode2")
    static let selectPowerMode3 = Self("selectPowerMode3")
    static let selectPowerMode4 = Self("selectPowerMode4")
    static let selectPowerMode5 = Self("selectPowerMode5")
    static let selectPowerMode6 = Self("selectPowerMode6")
    static let selectPowerMode7 = Self("selectPowerMode7")
    static let selectPowerMode8 = Self("selectPowerMode8")
    static let selectPowerMode9 = Self("selectPowerMode9")
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
    private var visibilityTask: Task<Void, Never>?
    
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
    
    // Add double-press Escape handling properties
    private var escFirstPressTime: Date? = nil
    private let escSecondPressThreshold: TimeInterval = 1.5 // seconds
    private var isEscapeHandlerSetup = false
    
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
        
        updateShortcutStatus()
        setupEnhancementShortcut()
        setupVisibilityObserver()
    }
    
    private func resetKeyStates() {
        currentKeyState = false
        keyPressStartTime = nil
        isHandsFreeMode = false
    }
    
    private func setupVisibilityObserver() {
        visibilityTask = Task { @MainActor in
            for await isVisible in whisperState.$isMiniRecorderVisible.values {
                if isVisible {
                    setupEscapeShortcut()
                    KeyboardShortcuts.setShortcut(.init(.e, modifiers: .command), for: .toggleEnhancement)
                    setupPowerModeShortcuts()
                } else {
                    removeEscapeShortcut()
                    removeEnhancementShortcut()
                    removePowerModeShortcuts()
                }
            }
        }
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
    
    private func setupEscapeShortcut() {
        KeyboardShortcuts.setShortcut(.init(.escape), for: .escapeRecorder)
        guard !isEscapeHandlerSetup else { return }
        isEscapeHandlerSetup = true
        KeyboardShortcuts.onKeyDown(for: .escapeRecorder) { [weak self] in
            Task { @MainActor in
                guard let self = self,
                      await self.whisperState.isMiniRecorderVisible else { return }
                
                let now = Date()
                if let firstTime = self.escFirstPressTime,
                   now.timeIntervalSince(firstTime) <= self.escSecondPressThreshold {
                    self.escFirstPressTime = nil
                    SoundManager.shared.playEscSound()
                    await self.whisperState.dismissMiniRecorder()
                } else {
                    self.escFirstPressTime = now
                    SoundManager.shared.playEscSound()
                    NotificationManager.shared.showNotification(
                        title: "Press ESC again to cancel recording",
                        type: .info,
                        duration: self.escSecondPressThreshold
                    )
                    Task { [weak self] in
                        try? await Task.sleep(nanoseconds: UInt64((self?.escSecondPressThreshold ?? 1.5) * 1_000_000_000))
                        await MainActor.run {
                            self?.escFirstPressTime = nil
                        }
                    }
                }
            }
        }
    }
    
    private func removeEscapeShortcut() {
        KeyboardShortcuts.setShortcut(nil, for: .escapeRecorder)
        escFirstPressTime = nil
    }
    
    private func setupEnhancementShortcut() {
        KeyboardShortcuts.onKeyDown(for: .toggleEnhancement) { [weak self] in
            Task { @MainActor in
                guard let self = self,
                      await self.whisperState.isMiniRecorderVisible,
                      let enhancementService = await self.whisperState.getEnhancementService() else { return }
                enhancementService.isEnhancementEnabled.toggle()
            }
        }
    }
    
    private func setupPowerModeShortcuts() {
        // Set up Command+1 through Command+9 shortcuts with proper key definitions
        KeyboardShortcuts.setShortcut(.init(.one, modifiers: .command), for: .selectPowerMode1)
        KeyboardShortcuts.setShortcut(.init(.two, modifiers: .command), for: .selectPowerMode2)
        KeyboardShortcuts.setShortcut(.init(.three, modifiers: .command), for: .selectPowerMode3)
        KeyboardShortcuts.setShortcut(.init(.four, modifiers: .command), for: .selectPowerMode4)
        KeyboardShortcuts.setShortcut(.init(.five, modifiers: .command), for: .selectPowerMode5)
        KeyboardShortcuts.setShortcut(.init(.six, modifiers: .command), for: .selectPowerMode6)
        KeyboardShortcuts.setShortcut(.init(.seven, modifiers: .command), for: .selectPowerMode7)
        KeyboardShortcuts.setShortcut(.init(.eight, modifiers: .command), for: .selectPowerMode8)
        KeyboardShortcuts.setShortcut(.init(.nine, modifiers: .command), for: .selectPowerMode9)
        
        // Setup handlers for each shortcut
        setupPowerModeHandler(for: .selectPowerMode1, index: 0)
        setupPowerModeHandler(for: .selectPowerMode2, index: 1)
        setupPowerModeHandler(for: .selectPowerMode3, index: 2)
        setupPowerModeHandler(for: .selectPowerMode4, index: 3)
        setupPowerModeHandler(for: .selectPowerMode5, index: 4)
        setupPowerModeHandler(for: .selectPowerMode6, index: 5)
        setupPowerModeHandler(for: .selectPowerMode7, index: 6)
        setupPowerModeHandler(for: .selectPowerMode8, index: 7)
        setupPowerModeHandler(for: .selectPowerMode9, index: 8)
    }
    
    private func setupPowerModeHandler(for shortcutName: KeyboardShortcuts.Name, index: Int) {
        KeyboardShortcuts.onKeyDown(for: shortcutName) { [weak self] in
            Task { @MainActor in
                guard let self = self,
                      await self.whisperState.isMiniRecorderVisible else { return }
                
                let powerModeManager = PowerModeManager.shared
                
                if powerModeManager.isPowerModeEnabled {
                    let availableConfigurations = powerModeManager.getAllAvailableConfigurations()
                    if index < availableConfigurations.count {
                        let selectedConfig = availableConfigurations[index]
                        powerModeManager.setActiveConfiguration(selectedConfig)
                        await ActiveWindowService.shared.applyConfiguration(selectedConfig)
                    }
                } else {
                    guard let enhancementService = await self.whisperState.getEnhancementService() else { return }
                    
                    let availablePrompts = enhancementService.allPrompts
                    if index < availablePrompts.count {
                        if !enhancementService.isEnhancementEnabled {
                            enhancementService.isEnhancementEnabled = true
                        }
                        
                        enhancementService.setActivePrompt(availablePrompts[index])
                    }
                }
            }
        }
    }
    
    private func removePowerModeShortcuts() {
        // Remove Command+1 through Command+9 shortcuts
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode1)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode2)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode3)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode4)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode5)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode6)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode7)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode8)
        KeyboardShortcuts.setShortcut(nil, for: .selectPowerMode9)
    }
    
    private func removeEnhancementShortcut() {
        KeyboardShortcuts.setShortcut(nil, for: .toggleEnhancement)
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
        visibilityTask?.cancel()
        Task { @MainActor in
            removeKeyMonitor()
            removeEscapeShortcut()
            removeEnhancementShortcut()
            removePowerModeShortcuts()
        }
    }
}
