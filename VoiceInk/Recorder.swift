import Foundation
import AVFoundation
import CoreAudio
import os

private final class NotificationObserverToken: @unchecked Sendable {
    var value: NSObjectProtocol?
}

@MainActor
class Recorder: NSObject, ObservableObject {
    private var recorder: CoreAudioRecorder?
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "Recorder")
    private let deviceManager = AudioDeviceManager.shared
    private let deviceSwitchObserver = NotificationObserverToken()
    private var isReconfiguring = false
    private let mediaController = MediaController.shared
    private let playbackController = PlaybackController.shared
    @Published var audioMeter = AudioMeter(averagePower: 0, peakPower: 0)
    private var audioLevelCheckTask: Task<Void, Never>?
    private var audioMeterUpdateTask: Task<Void, Never>?
    /// Dedicated serial queue for hardware setup.
    private let audioSetupQueue = DispatchQueue(label: "com.fightingentropy.voiceink.audioSetup", qos: .userInitiated)
    private var audioRestorationTask: Task<Void, Never>?
    private var hasDetectedAudioInCurrentSession = false
    private let smoothedValuesLock = NSLock()
    private var smoothedAverage: Float = 0
    private var smoothedPeak: Float = 0
    /// Last meter snapshot actually published to `audioMeter`. Used to suppress
    /// no-op @Published writes that would otherwise trigger SwiftUI invalidation
    /// every tick even when the level barely changed.
    private var lastPublishedAverage: Double = 0
    private var lastPublishedPeak: Double = 0

    /// Audio chunk callback for streaming. Can be updated while recording;
    /// changes are forwarded to the live CoreAudioRecorder.
    var onAudioChunk: (@Sendable (_ data: Data) -> Void)? {
        didSet { recorder?.onAudioChunk = onAudioChunk }
    }
    
    enum RecorderError: Error {
        case couldNotStartRecording
    }
    
    override init() {
        super.init()
        setupDeviceSwitchObserver()
    }

    private func setupDeviceSwitchObserver() {
        deviceSwitchObserver.value = NotificationCenter.default.addObserver(
            forName: .audioDeviceSwitchRequired,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let newDeviceID = notification.userInfo?["newDeviceID"] as? AudioDeviceID else {
                return
            }
            Task { @MainActor [weak self] in
                await self?.handleDeviceSwitchRequired(newDeviceID)
            }
        }
    }

    private func handleDeviceSwitchRequired(_ newDeviceID: AudioDeviceID) async {
        guard !isReconfiguring else { return }
        guard let recorder = recorder else { return }

        // Prevent concurrent device switches and handleDeviceChange() interference
        isReconfiguring = true
        defer { isReconfiguring = false }

        logger.notice("🎙️ Device switch required: switching to device \(newDeviceID, privacy: .public)")

        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                audioSetupQueue.async {
                    do {
                        try recorder.switchDevice(to: newDeviceID)
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            // Notify user about the switch
            if let deviceName = deviceManager.availableDevices.first(where: { $0.id == newDeviceID })?.name {
                await MainActor.run {
                    NotificationManager.shared.showNotification(
                        title: "Switched to: \(deviceName)",
                        type: .info
                    )
                }
            }

            logger.notice("🎙️ Successfully switched recording to device \(newDeviceID, privacy: .public)")
        } catch {
            logger.error("❌ Failed to switch device: \(error.localizedDescription, privacy: .public)")

            // If switch fails, stop recording and notify user
            await handleRecordingError(error)
        }
    }

    func startRecording(toOutputFile url: URL) async throws {
        logger.notice("startRecording called – deviceID=\(self.deviceManager.getCurrentDevice(), privacy: .public), file=\(url.lastPathComponent, privacy: .public)")
        deviceManager.isRecordingActive = true
        
        let currentDeviceID = deviceManager.getCurrentDevice()
        let lastDeviceID = UserDefaults.standard.string(forKey: "lastUsedMicrophoneDeviceID")
        
        if String(currentDeviceID) != lastDeviceID {
            if let deviceName = deviceManager.availableDevices.first(where: { $0.id == currentDeviceID })?.name {
                await MainActor.run {
                    NotificationManager.shared.showNotification(
                        title: "Using: \(deviceName)",
                        type: .info
                    )
                }
            }
        }
        UserDefaults.standard.set(String(currentDeviceID), forKey: "lastUsedMicrophoneDeviceID")
        
        hasDetectedAudioInCurrentSession = false

        let deviceID = deviceManager.getCurrentDevice()

        do {
            let coreAudioRecorder = CoreAudioRecorder()
            coreAudioRecorder.onAudioChunk = onAudioChunk
            recorder = coreAudioRecorder

            // Offload initialization to background thread to avoid hotkey lag.
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                audioSetupQueue.async {
                    do {
                        try coreAudioRecorder.startRecording(toOutputFile: url, deviceID: deviceID)
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
            logger.notice("startRecording: CoreAudioRecorder started successfully")

            audioRestorationTask?.cancel()
            audioRestorationTask = nil

            Task { [weak self] in
                guard let self = self else { return }
                await self.playbackController.pauseMedia()
                _ = await self.mediaController.muteSystemAudio()
            }

            audioLevelCheckTask?.cancel()
            audioMeterUpdateTask?.cancel()

            startAudioMeterTimer()

            audioLevelCheckTask = Task {
                let notificationChecks: [TimeInterval] = [5.0, 12.0]

                for delay in notificationChecks {
                    try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))

                    if Task.isCancelled { return }

                    if self.hasDetectedAudioInCurrentSession {
                        return
                    }

                    await MainActor.run {
                        NotificationManager.shared.showNotification(
                            title: "No Audio Detected",
                            type: .warning
                        )
                    }
                }
            }

        } catch {
            logger.error("Failed to create audio recorder: \(error.localizedDescription, privacy: .public)")
            await stopRecording()
            throw RecorderError.couldNotStartRecording
        }
    }

    func stopRecording() async {
        logger.notice("stopRecording called")
        audioLevelCheckTask?.cancel()
        audioMeterUpdateTask?.cancel()
        audioMeterUpdateTask = nil

        // Capture current recorder and wait for the audio file to be finalized
        // before downstream code tries to read the freshly written WAV.
        let currentRecorder = self.recorder
        recorder = nil
        onAudioChunk = nil

        await withCheckedContinuation { continuation in
            audioSetupQueue.async {
                currentRecorder?.stopRecording()
                continuation.resume()
            }
        }

        smoothedValuesLock.withLock {
            smoothedAverage = 0
            smoothedPeak = 0
        }

        audioMeter = AudioMeter(averagePower: 0, peakPower: 0)
        lastPublishedAverage = 0
        lastPublishedPeak = 0

        audioRestorationTask = Task {
            await mediaController.unmuteSystemAudio()
            await playbackController.resumeMedia()
        }
        deviceManager.isRecordingActive = false
    }

    private func handleRecordingError(_ error: Error) async {
        logger.error("❌ Recording error occurred: \(error.localizedDescription, privacy: .public)")

        // Stop the recording
        await stopRecording()

        // Notify the user about the recording failure
        await MainActor.run {
            NotificationManager.shared.showNotification(
                title: "Recording Failed: \(error.localizedDescription)",
                type: .error
            )
        }
    }

    private func startAudioMeterTimer() {
        audioMeterUpdateTask?.cancel()
        audioMeterUpdateTask = Task { @MainActor [weak self] in
            guard let self = self else { return }

            while !Task.isCancelled {
                self.updateAudioMeter()
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
    }

    private func updateAudioMeter() {
        guard let recorder = recorder else { return }

        // Sample audio levels (thread-safe read)
        let averagePower = recorder.averagePower
        let peakPower = recorder.peakPower

        // Normalize values
        let minVisibleDb: Float = -60.0
        let maxVisibleDb: Float = 0.0

        let normalizedAverage: Float
        if averagePower < minVisibleDb {
            normalizedAverage = 0.0
        } else if averagePower >= maxVisibleDb {
            normalizedAverage = 1.0
        } else {
            normalizedAverage = (averagePower - minVisibleDb) / (maxVisibleDb - minVisibleDb)
        }

        let normalizedPeak: Float
        if peakPower < minVisibleDb {
            normalizedPeak = 0.0
        } else if peakPower >= maxVisibleDb {
            normalizedPeak = 1.0
        } else {
            normalizedPeak = (peakPower - minVisibleDb) / (maxVisibleDb - minVisibleDb)
        }

        // Apply EMA smoothing with thread-safe access
        smoothedValuesLock.lock()
        smoothedAverage = smoothedAverage * 0.6 + normalizedAverage * 0.4
        smoothedPeak = smoothedPeak * 0.6 + normalizedPeak * 0.4
        let newAudioMeter = AudioMeter(averagePower: Double(smoothedAverage), peakPower: Double(smoothedPeak))
        smoothedValuesLock.unlock()

        if !hasDetectedAudioInCurrentSession && newAudioMeter.averagePower > 0.01 {
            hasDetectedAudioInCurrentSession = true
        }

        // Skip @Published writes when the smoothed level hasn't moved enough to
        // be visible — SwiftUI would otherwise re-lay-out the waveform every
        // tick for sub-pixel deltas.
        let epsilon = 0.005
        if abs(newAudioMeter.averagePower - lastPublishedAverage) < epsilon,
           abs(newAudioMeter.peakPower - lastPublishedPeak) < epsilon {
            return
        }
        lastPublishedAverage = newAudioMeter.averagePower
        lastPublishedPeak = newAudioMeter.peakPower
        audioMeter = newAudioMeter
    }
    
    // MARK: - Cleanup

    deinit {
        audioLevelCheckTask?.cancel()
        audioMeterUpdateTask?.cancel()
        audioRestorationTask?.cancel()
        if let observer = deviceSwitchObserver.value {
            NotificationCenter.default.removeObserver(observer)
        }
    }
}

struct AudioMeter: Equatable {
    let averagePower: Double
    let peakPower: Double
}
