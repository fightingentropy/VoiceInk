import Foundation
import CoreAudio
import AVFoundation
import os

struct PrioritizedDevice: Codable, Identifiable {
    let id: String
    let name: String
    let priority: Int
}

enum AudioInputMode: String, CaseIterable {
    case custom = "Custom Device"
    case prioritized = "Prioritized"
}

class AudioDeviceManager: ObservableObject {
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "AudioDeviceManager")
    @Published var availableDevices: [(id: AudioDeviceID, uid: String, name: String)] = []
    @Published var selectedDeviceID: AudioDeviceID?
    @Published var inputMode: AudioInputMode = .custom
    @Published var prioritizedDevices: [PrioritizedDevice] = []
    var fallbackDeviceID: AudioDeviceID?

    var isRecordingActive: Bool = false

    static let shared = AudioDeviceManager()

    init() {
        setupFallbackDevice()
        loadPrioritizedDevices()

        if let savedMode = UserDefaults.standard.audioInputModeRawValue,
           let mode = AudioInputMode(rawValue: savedMode) {
            inputMode = mode
        } else {
            inputMode = .custom
        }

        loadAvailableDevices { [weak self] in
            self?.migrateFromSystemDefaultIfNeeded()
            self?.initializeSelectedDevice()
        }

        setupDeviceChangeNotifications()
    }

    private func migrateFromSystemDefaultIfNeeded() {
        if let savedModeRaw = UserDefaults.standard.audioInputModeRawValue,
           savedModeRaw == "System Default" {
            logger.notice("🎙️ Migrating from System Default mode to Custom mode")

            if let fallbackID = fallbackDeviceID {
                selectedDeviceID = fallbackID
                if let device = availableDevices.first(where: { $0.id == fallbackID }) {
                    UserDefaults.standard.selectedAudioDeviceUID = device.uid
                    logger.notice("🎙️ Migrated to Custom mode with device: \(device.name)")
                }
            }

            UserDefaults.standard.audioInputModeRawValue = AudioInputMode.custom.rawValue
        }
    }
    
    func setupFallbackDevice() {
        let deviceID: AudioDeviceID? = getDeviceProperty(
            deviceID: AudioObjectID(kAudioObjectSystemObject),
            selector: kAudioHardwarePropertyDefaultInputDevice
        )
        
        if let deviceID = deviceID {
            fallbackDeviceID = deviceID
        } else {
            logger.error("Failed to get fallback device")
        }
    }
    
    private func initializeSelectedDevice() {
        if inputMode == .prioritized {
            selectHighestPriorityAvailableDevice()
            return
        }
        
        if let savedUID = UserDefaults.standard.selectedAudioDeviceUID {
            if let device = availableDevices.first(where: { $0.uid == savedUID }) {
                selectedDeviceID = device.id
            } else {
                logger.warning("🎙️ Saved device UID \(savedUID) is no longer available")
                UserDefaults.standard.removeObject(forKey: UserDefaults.Keys.selectedAudioDeviceUID)
                fallbackToDefaultDevice()
            }
        } else {
            fallbackToDefaultDevice()
        }
    }
    
    private func isDeviceAvailable(_ deviceID: AudioDeviceID) -> Bool {
        return availableDevices.contains { $0.id == deviceID }
    }
    
    private func fallbackToDefaultDevice() {
        logger.notice("🎙️ Current device unavailable, selecting new device...")

        guard let newDeviceID = findBestAvailableDevice() else {
            logger.error("No input devices available!")
            selectedDeviceID = nil
            notifyDeviceChange()
            return
        }

        let newDeviceName = getDeviceName(deviceID: newDeviceID) ?? "Unknown Device"
        logger.notice("🎙️ Auto-selecting new device: \(newDeviceName)")
        selectDevice(id: newDeviceID)
    }

    func findBestAvailableDevice() -> AudioDeviceID? {
        if let device = availableDevices.first(where: { isBuiltInDevice($0.id) }) {
            return device.id
        }
        if let device = availableDevices.first {
            logger.warning("🎙️ No built-in device found, using: \(device.name)")
            return device.id
        }
        return nil
    }

    private func isBuiltInDevice(_ deviceID: AudioDeviceID) -> Bool {
        guard let uid = getDeviceUID(deviceID: deviceID) else {
            return false
        }
        return uid.contains("BuiltIn")
    }
    
    func loadAvailableDevices(completion: (() -> Void)? = nil) {
        var propertySize: UInt32 = 0
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        
        var result = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propertySize
        )
        
        let deviceCount = Int(propertySize) / MemoryLayout<AudioDeviceID>.size
        
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        
        result = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propertySize,
            &deviceIDs
        )
        
        if result != noErr {
            logger.error("Error getting audio devices: \(result)")
            return
        }
        
        let devices = deviceIDs.compactMap { deviceID -> (id: AudioDeviceID, uid: String, name: String)? in
            guard let name = getDeviceName(deviceID: deviceID),
                  let uid = getDeviceUID(deviceID: deviceID),
                  isValidInputDevice(deviceID: deviceID) else {
                return nil
            }
            return (id: deviceID, uid: uid, name: name)
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.availableDevices = devices.map { ($0.id, $0.uid, $0.name) }
            if let currentID = self.selectedDeviceID, !devices.contains(where: { $0.id == currentID }) {
                self.logger.warning("🎙️ Currently selected device is no longer available")
                // Skip auto-fallback during recording; handleDeviceListChange manages it
                if !self.isRecordingActive {
                    if self.inputMode == .prioritized {
                        self.selectHighestPriorityAvailableDevice()
                    } else {
                        self.fallbackToDefaultDevice()
                    }
                }
            }
            completion?()
        }
    }
    
    func getDeviceName(deviceID: AudioDeviceID) -> String? {
        let name: CFString? = getDeviceProperty(deviceID: deviceID,
                                              selector: kAudioDevicePropertyDeviceNameCFString)
        return name as String?
    }
    
    private func isValidInputDevice(deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )

        var propertySize: UInt32 = 0
        var result = AudioObjectGetPropertyDataSize(
            deviceID,
            &address,
            0,
            nil,
            &propertySize
        )

        if result != noErr {
            logger.error("Error checking input capability for device \(deviceID): \(result)")
            return false
        }

        let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: Int(propertySize))
        defer { bufferList.deallocate() }

        result = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            bufferList
        )

        if result != noErr {
            logger.error("Error getting stream configuration for device \(deviceID): \(result)")
            return false
        }

        let bufferCount = Int(bufferList.pointee.mNumberBuffers)
        return bufferCount > 0
    }

    func selectDevice(id: AudioDeviceID) {
        if let deviceToSelect = availableDevices.first(where: { $0.id == id }) {
            let uid = deviceToSelect.uid
            DispatchQueue.main.async {
                self.selectedDeviceID = id
                UserDefaults.standard.selectedAudioDeviceUID = uid
                self.notifyDeviceChange()
            }
        } else {
            logger.error("Attempted to select unavailable device: \(id)")
            fallbackToDefaultDevice()
        }
    }

    func selectDeviceAndSwitchToCustomMode(id: AudioDeviceID) {
        if let deviceToSelect = availableDevices.first(where: { $0.id == id }) {
            let uid = deviceToSelect.uid
            DispatchQueue.main.async {
                self.inputMode = .custom
                self.selectedDeviceID = id
                UserDefaults.standard.audioInputModeRawValue = AudioInputMode.custom.rawValue
                UserDefaults.standard.selectedAudioDeviceUID = uid
                self.notifyDeviceChange()
            }
        } else {
            logger.error("Attempted to select unavailable device: \(id)")
            fallbackToDefaultDevice()
        }
    }
    
    func selectInputMode(_ mode: AudioInputMode) {
        inputMode = mode
        UserDefaults.standard.audioInputModeRawValue = mode.rawValue

        if selectedDeviceID == nil {
            if inputMode == .custom {
                if let firstDevice = availableDevices.first {
                    selectDevice(id: firstDevice.id)
                }
            } else if inputMode == .prioritized {
                selectHighestPriorityAvailableDevice()
            }
        }

        notifyDeviceChange()
    }
    
    func getCurrentDevice() -> AudioDeviceID {
        switch inputMode {
        case .custom:
            if let id = selectedDeviceID, isDeviceAvailable(id) {
                return id
            } else {
                // Use smart device finding instead of stale fallback
                return findBestAvailableDevice() ?? 0
            }
        case .prioritized:
            let sortedDevices = prioritizedDevices.sorted { $0.priority < $1.priority }
            for device in sortedDevices {
                if let available = availableDevices.first(where: { $0.uid == device.id }) {
                    return available.id
                }
            }
            // Use smart device finding instead of stale fallback
            return findBestAvailableDevice() ?? 0
        }
    }
    
    private func loadPrioritizedDevices() {
        if let data = UserDefaults.standard.prioritizedDevicesData,
           let devices = try? JSONDecoder().decode([PrioritizedDevice].self, from: data) {
            prioritizedDevices = devices
        }
    }
    
    func savePrioritizedDevices() {
        if let data = try? JSONEncoder().encode(prioritizedDevices) {
            UserDefaults.standard.prioritizedDevicesData = data
        }
    }
    
    func addPrioritizedDevice(uid: String, name: String) {
        guard !prioritizedDevices.contains(where: { $0.id == uid }) else { return }
        let nextPriority = (prioritizedDevices.map { $0.priority }.max() ?? -1) + 1
        let device = PrioritizedDevice(id: uid, name: name, priority: nextPriority)
        prioritizedDevices.append(device)
        savePrioritizedDevices()
    }
    
    func removePrioritizedDevice(id: String) {
        let wasSelected = selectedDeviceID == availableDevices.first(where: { $0.uid == id })?.id
        prioritizedDevices.removeAll { $0.id == id }
        
        let updatedDevices = prioritizedDevices.enumerated().map { index, device in
            PrioritizedDevice(id: device.id, name: device.name, priority: index)
        }
        
        prioritizedDevices = updatedDevices
        savePrioritizedDevices()
        
        if wasSelected && inputMode == .prioritized {
            selectHighestPriorityAvailableDevice()
        }
    }
    
    func updatePriorities(devices: [PrioritizedDevice]) {
        prioritizedDevices = devices
        savePrioritizedDevices()
        
        if inputMode == .prioritized {
            selectHighestPriorityAvailableDevice()
        }
        
        notifyDeviceChange()
    }
    
    private func selectHighestPriorityAvailableDevice() {
        let sortedDevices = prioritizedDevices.sorted { $0.priority < $1.priority }

        for device in sortedDevices {
            if let availableDevice = availableDevices.first(where: { $0.uid == device.id }) {
                selectedDeviceID = availableDevice.id
                logger.notice("🎙️ Selected prioritized device: \(device.name)")
                notifyDeviceChange()
                return
            }
        }

        fallbackToDefaultDevice()
    }
    
    private func setupDeviceChangeNotifications() {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        
        let systemObjectID = AudioObjectID(kAudioObjectSystemObject)
        
        let status = AudioObjectAddPropertyListener(
            systemObjectID,
            &address,
            { (_, _, _, userData) -> OSStatus in
                let manager = Unmanaged<AudioDeviceManager>.fromOpaque(userData!).takeUnretainedValue()
                DispatchQueue.main.async {
                    manager.handleDeviceListChange()
                }
                return noErr
            },
            UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
        )
        
        if status != noErr {
            logger.error("Failed to add device change listener: \(status)")
        }
    }
    
    private func handleDeviceListChange() {
        logger.notice("🎙️ Device list change detected")

        loadAvailableDevices { [weak self] in
            guard let self = self else { return }

            // If recording is active, check if we need to switch devices
            if self.isRecordingActive {
                guard let currentID = self.selectedDeviceID else { return }

                // Check if current recording device is still available
                if !self.isDeviceAvailable(currentID) {
                    self.logger.warning("🎙️ Recording device \(currentID) no longer available - requesting switch")

                    // Find next device based on input mode
                    let newDeviceID: AudioDeviceID?

                    if self.inputMode == .prioritized {
                        // Respect priority order: find next available prioritized device
                        let sortedDevices = self.prioritizedDevices.sorted { $0.priority < $1.priority }
                        let priorityDeviceID = sortedDevices.compactMap { device in
                            self.availableDevices.first(where: { $0.uid == device.id })?.id
                        }.first

                        if let deviceID = priorityDeviceID {
                            newDeviceID = deviceID
                        } else {
                            // No priority devices available, fall back to best available
                            self.logger.warning("🎙️ No priority devices available, using fallback")
                            newDeviceID = self.findBestAvailableDevice()
                        }
                    } else {
                        // Custom mode: use best available device
                        newDeviceID = self.findBestAvailableDevice()
                    }

                    if let deviceID = newDeviceID {
                        self.selectedDeviceID = deviceID

                        // Notify recorder to switch devices mid-recording
                        NotificationCenter.default.post(
                            name: .audioDeviceSwitchRequired,
                            object: nil,
                            userInfo: ["newDeviceID": deviceID]
                        )
                    } else {
                        self.logger.error("No audio input devices available!")
                        // Stop recording since no device is available
                        NotificationCenter.default.post(name: .toggleMiniRecorder, object: nil)
                    }
                }
                return
            }

            // Not recording - handle normally
            if self.inputMode == .prioritized {
                self.selectHighestPriorityAvailableDevice()
            } else if self.inputMode == .custom,
                      let currentID = self.selectedDeviceID,
                      !self.isDeviceAvailable(currentID) {
                self.fallbackToDefaultDevice()
            }
        }
    }
    
    private func getDeviceUID(deviceID: AudioDeviceID) -> String? {
        let uid: CFString? = getDeviceProperty(deviceID: deviceID,
                                             selector: kAudioDevicePropertyDeviceUID)
        return uid as String?
    }
    
    deinit {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        
        AudioObjectRemovePropertyListener(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            { (_, _, _, userData) -> OSStatus in
                return noErr
            },
            UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
        )
    }
    
    private func createPropertyAddress(selector: AudioObjectPropertySelector,
                                    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
                                    element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain) -> AudioObjectPropertyAddress {
        return AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: scope,
            mElement: element
        )
    }
    
    private func getDeviceProperty<T>(deviceID: AudioDeviceID,
                                    selector: AudioObjectPropertySelector,
                                    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> T? {
        guard deviceID != 0 else { return nil }
        
        var address = createPropertyAddress(selector: selector, scope: scope)
        var propertySize = UInt32(MemoryLayout<T>.size)
        var property: T? = nil
        
        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            &property
        )
        
        if status != noErr {
            logger.error("Failed to get device property \(selector) for device \(deviceID): \(status)")
            return nil
        }
        
        return property
    }
    
    private func notifyDeviceChange() {
        NotificationCenter.default.post(name: NSNotification.Name("AudioDeviceChanged"), object: nil)
    }
} 
