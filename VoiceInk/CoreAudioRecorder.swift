import Foundation
import CoreAudio
import AudioToolbox
@preconcurrency import AVFoundation
import Atomics
import os

/// Moves the worker's exclusively-owned ExtAudioFileRef across the Thread
/// boundary; only the drain worker touches it after startWorker hands it off.
private struct AudioFileRef: @unchecked Sendable {
    let ref: ExtAudioFileRef
}

// MARK: - SPSC Audio Packet Ring

/// Fixed-size single-producer/single-consumer ring. The realtime callback
/// publishes mono Float32 packets with no locks and no heap allocation; the
/// drain worker consumes them in order. @unchecked Sendable: the atomic
/// write/read indices are the only cross-thread coordination, and each slot
/// is owned exclusively by producer or consumer at any given time.
private final class AudioPacketRing: @unchecked Sendable {
    static let slotCount = 64
    static let slotCapacity = 4096 // frames of mono Float32 per slot

    struct Packet {
        let samples: UnsafePointer<Float32>
        let frameCount: Int
        let sampleRate: Double
        let hostTime: UInt64
    }

    private let sampleStorage: UnsafeMutablePointer<Float32>
    private let frameCounts: UnsafeMutablePointer<UInt32>
    private let sampleRates: UnsafeMutablePointer<Double>
    private let hostTimes: UnsafeMutablePointer<UInt64>
    private let writeIndex = ManagedAtomic<UInt64>(0)
    private let readIndex = ManagedAtomic<UInt64>(0)
    let droppedPackets = ManagedAtomic<UInt64>(0)

    init() {
        sampleStorage = .allocate(capacity: Self.slotCount * Self.slotCapacity)
        frameCounts = .allocate(capacity: Self.slotCount)
        sampleRates = .allocate(capacity: Self.slotCount)
        hostTimes = .allocate(capacity: Self.slotCount)
    }

    deinit {
        sampleStorage.deallocate()
        frameCounts.deallocate()
        sampleRates.deallocate()
        hostTimes.deallocate()
    }

    var isEmpty: Bool {
        readIndex.load(ordering: .relaxed) == writeIndex.load(ordering: .acquiring)
    }

    /// Realtime-safe producer: lock-free, allocation-free. Returns false when
    /// the packet was dropped (ring full or oversized packet).
    func publish(samples: UnsafePointer<Float32>, frameCount: Int, sampleRate: Double, hostTime: UInt64) -> Bool {
        guard frameCount > 0 else { return false }
        guard frameCount <= Self.slotCapacity else {
            droppedPackets.wrappingIncrement(ordering: .relaxed)
            return false
        }
        let write = writeIndex.load(ordering: .relaxed)
        let read = readIndex.load(ordering: .acquiring)
        guard write &- read < UInt64(Self.slotCount) else {
            droppedPackets.wrappingIncrement(ordering: .relaxed)
            return false
        }
        let slot = Int(write % UInt64(Self.slotCount))
        (sampleStorage + slot * Self.slotCapacity).update(from: samples, count: frameCount)
        frameCounts[slot] = UInt32(frameCount)
        sampleRates[slot] = sampleRate
        hostTimes[slot] = hostTime
        writeIndex.store(write &+ 1, ordering: .releasing)
        return true
    }

    /// Consumer side: the returned samples pointer stays valid until consume().
    func peek() -> Packet? {
        let read = readIndex.load(ordering: .relaxed)
        let write = writeIndex.load(ordering: .acquiring)
        guard read != write else { return nil }
        let slot = Int(read % UInt64(Self.slotCount))
        return Packet(
            samples: UnsafePointer(sampleStorage + slot * Self.slotCapacity),
            frameCount: Int(frameCounts[slot]),
            sampleRate: sampleRates[slot],
            hostTime: hostTimes[slot]
        )
    }

    func consume() {
        let read = readIndex.load(ordering: .relaxed)
        readIndex.store(read &+ 1, ordering: .releasing)
    }
}

// MARK: - Anti-aliased Resampler (worker-thread only)

/// Converts device-rate mono Float32 to 16kHz mono Int16 through
/// AVAudioConverter's polyphase sample-rate converter. Recreated in place when
/// the input sample rate changes (e.g. after a device switch), flushing the
/// buffered tail of the previous converter first.
/// @unchecked Sendable: confined to the drain worker thread; the converter's
/// @Sendable input block runs synchronously inside convert() on that thread.
private final class MonoResampler16k: @unchecked Sendable {
    static let outputSampleRate = 16000.0

    private let logger: Logger
    private var converter: AVAudioConverter?
    private var inputSampleRate: Double = 0
    private var inputBuffer: AVAudioPCMBuffer?
    private var outputBuffer: AVAudioPCMBuffer?
    /// One-shot handoff to the converter's input block; convert() consumes it
    /// synchronously, so it never outlives a process() call.
    private var pendingInput: AVAudioPCMBuffer?
    private(set) var tailFramesFlushed: UInt64 = 0

    init(logger: Logger) {
        self.logger = logger
    }

    func process(samples: UnsafePointer<Float32>, frameCount: Int, sampleRate: Double, sink: (AVAudioPCMBuffer) -> Void) {
        if converter == nil || sampleRate != inputSampleRate {
            flush(sink: sink)
            prepare(inputSampleRate: sampleRate)
        }
        guard let converter = converter, let inputBuffer = inputBuffer, let outputBuffer = outputBuffer else { return }
        guard frameCount > 0, frameCount <= Int(inputBuffer.frameCapacity) else { return }

        inputBuffer.floatChannelData![0].update(from: samples, count: frameCount)
        inputBuffer.frameLength = AVAudioFrameCount(frameCount)
        outputBuffer.frameLength = 0

        pendingInput = inputBuffer
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { [self] _, outStatus in
            guard let buffer = pendingInput else {
                outStatus.pointee = .noDataNow
                return nil
            }
            pendingInput = nil
            outStatus.pointee = .haveData
            return buffer
        }

        if status == .error {
            logger.error("🎙️ AVAudioConverter failed: \(conversionError?.localizedDescription ?? "unknown", privacy: .public)")
            return
        }
        if outputBuffer.frameLength > 0 {
            sink(outputBuffer)
        }
    }

    /// Drains the polyphase converter's buffered samples (the last ~100ms of
    /// dictation) by feeding end-of-stream, then discards the converter.
    func flush(sink: (AVAudioPCMBuffer) -> Void) {
        defer {
            converter = nil
            inputSampleRate = 0
        }
        guard let converter = converter, let outputBuffer = outputBuffer else { return }

        while true {
            outputBuffer.frameLength = 0
            var conversionError: NSError?
            let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
                outStatus.pointee = .endOfStream
                return nil
            }
            if status == .error {
                logger.error("🎙️ AVAudioConverter tail flush failed: \(conversionError?.localizedDescription ?? "unknown", privacy: .public)")
                return
            }
            guard outputBuffer.frameLength > 0 else { return }
            tailFramesFlushed &+= UInt64(outputBuffer.frameLength)
            sink(outputBuffer)
            if status == .endOfStream { return }
        }
    }

    private func prepare(inputSampleRate: Double) {
        guard inputSampleRate > 0,
              let inputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputSampleRate, channels: 1, interleaved: false),
              let outputFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Self.outputSampleRate, channels: 1, interleaved: true),
              let newConverter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            logger.error("🎙️ Failed to create AVAudioConverter for input rate \(inputSampleRate, privacy: .public)")
            return
        }
        newConverter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        let inputCapacity = AVAudioFrameCount(AudioPacketRing.slotCapacity)
        let outputCapacity = AVAudioFrameCount((Double(AudioPacketRing.slotCapacity) * Self.outputSampleRate / inputSampleRate).rounded(.up)) + 256
        guard let newInputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputCapacity),
              let newOutputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
            logger.error("🎙️ Failed to allocate conversion buffers for input rate \(inputSampleRate, privacy: .public)")
            return
        }

        converter = newConverter
        inputBuffer = newInputBuffer
        outputBuffer = newOutputBuffer
        self.inputSampleRate = inputSampleRate
        logger.notice("🎙️ Resampler ready: \(Int(inputSampleRate), privacy: .public)Hz → \(Int(Self.outputSampleRate), privacy: .public)Hz (anti-aliased, quality=max)")
    }
}

// MARK: - Core Audio Recorder (AUHAL-based, does not change system default device)
final class CoreAudioRecorder: @unchecked Sendable {

    // MARK: - Properties

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "CoreAudioRecorder")

    private var audioUnit: AudioUnit?
    private var audioFile: ExtAudioFileRef?

    private var isRecording = false
    private var currentDeviceID: AudioDeviceID = 0
    private var recordingURL: URL?

    // Device format (what the hardware provides)
    private var deviceFormat = AudioStreamBasicDescription()
    // Output format (16kHz mono PCM Int16 for transcription)
    private var outputFormat = AudioStreamBasicDescription()

    // Audio metering (thread-safe, updated by the drain worker)
    private let meterLock = NSLock()
    private var _averagePower: Float = -160.0
    private var _peakPower: Float = -160.0

    var averagePower: Float {
        meterLock.lock()
        defer { meterLock.unlock() }
        return _averagePower
    }

    var peakPower: Float {
        meterLock.lock()
        defer { meterLock.unlock() }
        return _peakPower
    }

    // Pre-allocated render buffer (to avoid malloc in real-time callback)
    private var renderBuffer: UnsafeMutablePointer<Float32>?
    private var renderBufferSize: UInt32 = 0

    // SPSC handoff between the realtime callback and the drain worker
    private var ring: AudioPacketRing?
    private var packetSemaphore = DispatchSemaphore(value: 0)
    private var workerDone = DispatchSemaphore(value: 0)
    private var workerThread: Thread?
    private let workerStopRequested = ManagedAtomic<Bool>(false)
    private let isCapturing = ManagedAtomic<Bool>(false)
    private let captureStartNanos = ManagedAtomic<UInt64>(0)

    /// Called on the drain worker thread (serialized, in order) with converted
    /// PCM data (16-bit, 16kHz, mono) for streaming.
    private let chunkHandlerLock = NSLock()
    private var _onAudioChunk: (@Sendable (_ data: Data) -> Void)?
    var onAudioChunk: (@Sendable (_ data: Data) -> Void)? {
        get { chunkHandlerLock.withLock { _onAudioChunk } }
        set { chunkHandlerLock.withLock { _onAudioChunk = newValue } }
    }

    // MARK: - Initialization

    init() {}

    deinit {
        stopRecording()
    }

    // MARK: - Public Interface

    /// Starts recording from the specified device to the given URL (WAV format)
    func startRecording(toOutputFile url: URL, deviceID: AudioDeviceID) throws {
        // Stop any existing recording
        stopRecording()

        if deviceID == 0 {
            logger.error("Cannot start recording - no valid audio device (deviceID is 0)")
            throw CoreAudioRecorderError.failedToSetDevice(status: 0)
        }

        // Validate device still exists before proceeding with setup
        guard isDeviceAvailable(deviceID) else {
            logger.error("Cannot start recording - device \(deviceID, privacy: .public) is no longer available")
            throw CoreAudioRecorderError.deviceNotAvailable
        }

        currentDeviceID = deviceID
        recordingURL = url

        logger.notice("🎙️ Starting recording from device \(deviceID, privacy: .public)")
        logDeviceDetails(deviceID: deviceID)

        // Step 1: Create and configure the AudioUnit (AUHAL)
        try createAudioUnit()

        // Step 2: Set the input device (does NOT change system default)
        try setInputDevice(deviceID)

        // Step 3: Configure formats
        try configureFormats()

        // Step 4: Set up the input callback
        try setupInputCallback()

        // Step 5: Create the output file
        try createOutputFile(at: url)

        // Step 6: Start the drain worker, then the AudioUnit
        startWorker()
        try startAudioUnit()

        isRecording = true
    }

    /// Stops the current recording. Blocks (bounded) until the drain worker has
    /// flushed the converter tail and finalized the WAV, because callers read
    /// the file immediately after.
    func stopRecording() {
        guard isRecording || audioUnit != nil else {
            logger.notice("stopRecording: skipped, not recording and no audio unit")
            return
        }
        logger.notice("stopRecording: stopping core audio recorder")

        // Stop realtime callbacks before anything else
        if let unit = audioUnit {
            AudioOutputUnitStop(unit)
        }
        isCapturing.store(false, ordering: .releasing)

        // Let the worker drain remaining packets, flush the converter tail, and
        // finalize the file. The worker owns the file's disposal.
        if workerThread != nil {
            workerStopRequested.store(true, ordering: .releasing)
            packetSemaphore.signal()
            if workerDone.wait(timeout: .now() + .seconds(2)) == .timedOut {
                logger.error("stopRecording: drain worker did not finish within 2s")
            }
            workerThread = nil
        } else if let file = audioFile {
            ExtAudioFileDispose(file)
        }
        audioFile = nil

        // Dispose AudioUnit
        if let unit = audioUnit {
            AudioComponentInstanceDispose(unit)
            audioUnit = nil
        }

        ring = nil

        // Free render buffer
        if let buffer = renderBuffer {
            buffer.deallocate()
            renderBuffer = nil
            renderBufferSize = 0
        }

        isRecording = false
        currentDeviceID = 0
        recordingURL = nil

        // Reset meters
        meterLock.lock()
        _averagePower = -160.0
        _peakPower = -160.0
        meterLock.unlock()
    }

    var isCurrentlyRecording: Bool { isRecording }
    var currentRecordingURL: URL? { recordingURL }
    var currentDevice: AudioDeviceID { currentDeviceID }

    /// Switches to a new input device mid-recording without stopping the file write
    func switchDevice(to newDeviceID: AudioDeviceID) throws {
        guard isRecording, let unit = audioUnit else {
            throw CoreAudioRecorderError.audioUnitNotInitialized
        }

        // Don't switch if it's the same device
        guard newDeviceID != currentDeviceID else { return }

        let oldDeviceID = currentDeviceID
        logger.notice("🎙️ Switching recording device from \(oldDeviceID, privacy: .public) to \(newDeviceID, privacy: .public)")

        // Step 1: Stop the AudioUnit (but keep file open)
        var status = AudioOutputUnitStop(unit)
        if status != noErr {
            logger.warning("🎙️ Warning: AudioOutputUnitStop returned \(status, privacy: .public)")
        }

        // Step 2: Wait for the worker to drain packets from the old device.
        // Slots are tagged with their sample rate, so the worker recreates the
        // converter on its own if the new device runs at a different rate.
        if let ring = ring {
            let deadline = DispatchTime.now() + .seconds(1)
            while !ring.isEmpty && DispatchTime.now() < deadline {
                usleep(2000)
            }
            if !ring.isEmpty {
                logger.warning("🎙️ switchDevice: ring not fully drained before reconfiguration")
            }
        }

        // Step 3: Uninitialize to allow reconfiguration
        status = AudioUnitUninitialize(unit)
        if status != noErr {
            logger.warning("🎙️ Warning: AudioUnitUninitialize returned \(status, privacy: .public)")
        }

        // Step 4: Set the new device
        var device = newDeviceID
        status = AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &device,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )

        if status != noErr {
            // Try to recover by restarting with old device
            logger.error("Failed to set new device: \(status, privacy: .public). Attempting recovery...")
            var recoveryDevice = oldDeviceID
            AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &recoveryDevice, UInt32(MemoryLayout<AudioDeviceID>.size))
            AudioUnitInitialize(unit)
            AudioOutputUnitStart(unit)
            throw CoreAudioRecorderError.failedToSetDevice(status: status)
        }

        // Step 5: Get new device format
        var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var newDeviceFormat = AudioStreamBasicDescription()
        status = AudioUnitGetProperty(
            unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            1,
            &newDeviceFormat,
            &formatSize
        )

        if status != noErr {
            throw CoreAudioRecorderError.failedToGetDeviceFormat(status: status)
        }

        // Step 6: Configure callback format for new device
        var callbackFormat = AudioStreamBasicDescription(
            mSampleRate: newDeviceFormat.mSampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(MemoryLayout<Float32>.size) * newDeviceFormat.mChannelsPerFrame,
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(MemoryLayout<Float32>.size) * newDeviceFormat.mChannelsPerFrame,
            mChannelsPerFrame: newDeviceFormat.mChannelsPerFrame,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        status = AudioUnitSetProperty(
            unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            1,
            &callbackFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )

        if status != noErr {
            throw CoreAudioRecorderError.failedToSetFormat(status: status)
        }

        // Step 7: Reallocate render buffer if needed
        let maxFrames = UInt32(AudioPacketRing.slotCapacity)
        let bufferSamples = maxFrames * newDeviceFormat.mChannelsPerFrame
        if bufferSamples > renderBufferSize {
            renderBuffer?.deallocate()
            renderBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: Int(bufferSamples))
            renderBufferSize = bufferSamples
        }

        // Update stored format
        deviceFormat = newDeviceFormat
        currentDeviceID = newDeviceID

        // Step 8: Reinitialize and restart
        status = AudioUnitInitialize(unit)
        if status != noErr {
            throw CoreAudioRecorderError.failedToInitialize(status: status)
        }

        status = AudioOutputUnitStart(unit)
        if status != noErr {
            throw CoreAudioRecorderError.failedToStart(status: status)
        }

        logger.notice("🎙️ Successfully switched to device \(newDeviceID, privacy: .public)")
    }

    // MARK: - AudioUnit Setup

    private func createAudioUnit() throws {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        guard let component = AudioComponentFindNext(nil, &desc) else {
            logger.error("AudioUnit not found - HAL Output component unavailable")
            throw CoreAudioRecorderError.audioUnitNotFound
        }

        var unit: AudioUnit?
        var status = AudioComponentInstanceNew(component, &unit)
        guard status == noErr, let audioUnit = unit else {
            logger.error("Failed to create AudioUnit instance: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToCreateAudioUnit(status: status)
        }

        self.audioUnit = audioUnit

        // Enable input on element 1 (input scope)
        var enableInput: UInt32 = 1
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1, // Element 1 = input
            &enableInput,
            UInt32(MemoryLayout<UInt32>.size)
        )

        if status != noErr {
            logger.error("Failed to enable audio input: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToEnableInput(status: status)
        }

        // Disable output on element 0 (output scope)
        var disableOutput: UInt32 = 0
        status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            0, // Element 0 = output
            &disableOutput,
            UInt32(MemoryLayout<UInt32>.size)
        )

        if status != noErr {
            logger.error("Failed to disable audio output: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToDisableOutput(status: status)
        }
    }

    private func setInputDevice(_ deviceID: AudioDeviceID) throws {
        guard let audioUnit = audioUnit else {
            throw CoreAudioRecorderError.audioUnitNotInitialized
        }

        var device = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &device,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )

        if status != noErr {
            logger.error("Failed to set input device \(deviceID, privacy: .public): \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToSetDevice(status: status)
        }
    }

    private func configureFormats() throws {
        guard let audioUnit = audioUnit else {
            throw CoreAudioRecorderError.audioUnitNotInitialized
        }

        // Get the device's native format (input scope, element 1)
        var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var status = AudioUnitGetProperty(
            audioUnit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            1,
            &deviceFormat,
            &formatSize
        )

        if status != noErr {
            logger.error("Failed to get device format: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToGetDeviceFormat(status: status)
        }

        // Configure output format: 16kHz, mono, PCM Int16
        outputFormat = AudioStreamBasicDescription(
            mSampleRate: 16000.0,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 16,
            mReserved: 0
        )

        // Set callback format (Float32 for processing, converted off the RT thread)
        var callbackFormat = AudioStreamBasicDescription(
            mSampleRate: deviceFormat.mSampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(MemoryLayout<Float32>.size) * deviceFormat.mChannelsPerFrame,
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(MemoryLayout<Float32>.size) * deviceFormat.mChannelsPerFrame,
            mChannelsPerFrame: deviceFormat.mChannelsPerFrame,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        status = AudioUnitSetProperty(
            audioUnit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            1,
            &callbackFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )

        if status != noErr {
            logger.error("Failed to set audio format: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToSetFormat(status: status)
        }

        // Log format details
        let devSampleRate = deviceFormat.mSampleRate
        let devChannels = deviceFormat.mChannelsPerFrame
        let devBits = deviceFormat.mBitsPerChannel
        let outSampleRate = outputFormat.mSampleRate
        let outChannels = outputFormat.mChannelsPerFrame
        let outBits = outputFormat.mBitsPerChannel
        logger.notice("🎙️ Device format: sampleRate=\(devSampleRate, privacy: .public), channels=\(devChannels, privacy: .public), bitsPerChannel=\(devBits, privacy: .public)")
        logger.notice("🎙️ Output format: sampleRate=\(outSampleRate, privacy: .public), channels=\(outChannels, privacy: .public), bitsPerChannel=\(outBits, privacy: .public)")
        if devSampleRate != outSampleRate {
            logger.notice("🎙️ Converting: \(Int(devSampleRate), privacy: .public)Hz → \(Int(outSampleRate), privacy: .public)Hz")
        }

        // Pre-allocate the render buffer for the real-time callback
        let maxFrames = UInt32(AudioPacketRing.slotCapacity)
        let bufferSamples = maxFrames * deviceFormat.mChannelsPerFrame
        renderBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: Int(bufferSamples))
        renderBufferSize = bufferSamples
    }

    private func setupInputCallback() throws {
        guard let audioUnit = audioUnit else {
            throw CoreAudioRecorderError.audioUnitNotInitialized
        }

        var callbackStruct = AURenderCallbackStruct(
            inputProc: inputCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )

        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global,
            0,
            &callbackStruct,
            UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        )

        if status != noErr {
            logger.error("Failed to set input callback: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToSetCallback(status: status)
        }
    }

    private func createOutputFile(at url: URL) throws {
        // Remove existing file if any
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }

        // Create ExtAudioFile for writing
        var fileRef: ExtAudioFileRef?
        var status = ExtAudioFileCreateWithURL(
            url as CFURL,
            kAudioFileWAVEType,
            &outputFormat,
            nil,
            AudioFileFlags.eraseFile.rawValue,
            &fileRef
        )

        if status != noErr {
            logger.error("Failed to create audio file at \(url.path, privacy: .public): \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToCreateFile(status: status)
        }

        audioFile = fileRef

        // Set client format (what we'll write)
        status = ExtAudioFileSetProperty(
            fileRef!,
            kExtAudioFileProperty_ClientDataFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size),
            &outputFormat
        )

        if status != noErr {
            logger.error("Failed to set file format: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToSetFileFormat(status: status)
        }
    }

    private func startAudioUnit() throws {
        guard let audioUnit = audioUnit else {
            throw CoreAudioRecorderError.audioUnitNotInitialized
        }

        var status = AudioUnitInitialize(audioUnit)
        if status != noErr {
            logger.error("Failed to initialize AudioUnit: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToInitialize(status: status)
        }

        isCapturing.store(true, ordering: .releasing)
        captureStartNanos.store(DispatchTime.now().uptimeNanoseconds, ordering: .releasing)
        status = AudioOutputUnitStart(audioUnit)
        if status != noErr {
            isCapturing.store(false, ordering: .releasing)
            logger.error("Failed to start AudioUnit: \(status, privacy: .public)")
            throw CoreAudioRecorderError.failedToStart(status: status)
        }
    }

    // MARK: - Input Callback (realtime thread: no disk I/O, no locks, no allocation)

    private let inputCallback: AURenderCallback = { (
        inRefCon,
        ioActionFlags,
        inTimeStamp,
        inBusNumber,
        inNumberFrames,
        ioData
    ) -> OSStatus in

        let recorder = Unmanaged<CoreAudioRecorder>.fromOpaque(inRefCon).takeUnretainedValue()
        return recorder.handleInputBuffer(
            ioActionFlags: ioActionFlags,
            inTimeStamp: inTimeStamp,
            inBusNumber: inBusNumber,
            inNumberFrames: inNumberFrames
        )
    }

    private func handleInputBuffer(
        ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
        inTimeStamp: UnsafePointer<AudioTimeStamp>,
        inBusNumber: UInt32,
        inNumberFrames: UInt32
    ) -> OSStatus {

        guard isCapturing.load(ordering: .acquiring),
              let audioUnit = audioUnit,
              let renderBuf = renderBuffer,
              let ring = ring,
              inNumberFrames > 0 else {
            return noErr
        }

        // Use pre-allocated buffer for input data
        let channelCount = deviceFormat.mChannelsPerFrame
        let requiredSamples = inNumberFrames * channelCount

        // Safety check - shouldn't happen with 4096 max frames
        guard requiredSamples <= renderBufferSize else {
            return noErr
        }

        let bytesPerFrame = UInt32(MemoryLayout<Float32>.size) * channelCount
        let bufferSize = inNumberFrames * bytesPerFrame

        var bufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: channelCount,
                mDataByteSize: bufferSize,
                mData: renderBuf
            )
        )

        // Render audio from the input
        let status = AudioUnitRender(
            audioUnit,
            ioActionFlags,
            inTimeStamp,
            inBusNumber,
            inNumberFrames,
            &bufferList
        )

        if status != noErr {
            return status
        }

        // Mix interleaved channels down to mono in place (bounded arithmetic)
        let frames = Int(inNumberFrames)
        let channels = Int(channelCount)
        if channels > 1 {
            let scale = 1.0 / Float32(channels)
            for frame in 0..<frames {
                var sum: Float32 = 0
                let base = frame * channels
                for channel in 0..<channels {
                    sum += renderBuf[base + channel]
                }
                renderBuf[frame] = sum * scale
            }
        }

        // Publish to the SPSC ring; the drain worker does metering,
        // resampling, file writes, and chunk delivery off this thread.
        let timeStamp = inTimeStamp.pointee
        let hostTime = timeStamp.mFlags.contains(.hostTimeValid) ? timeStamp.mHostTime : mach_absolute_time()
        if ring.publish(samples: renderBuf, frameCount: frames, sampleRate: deviceFormat.mSampleRate, hostTime: hostTime) {
            packetSemaphore.signal()
        }

        return noErr
    }

    // MARK: - Drain Worker

    private func startWorker() {
        guard let file = audioFile else { return }

        let ring = AudioPacketRing()
        self.ring = ring
        let semaphore = DispatchSemaphore(value: 0)
        packetSemaphore = semaphore
        let done = DispatchSemaphore(value: 0)
        workerDone = done
        workerStopRequested.store(false, ordering: .releasing)

        // The worker must not retain self: an abandoned recorder relies on
        // deinit -> stopRecording() to stop the AudioUnit and finalize the
        // file, and a strong capture would keep the mic hot forever.
        let logger = self.logger
        let stopRequested = workerStopRequested
        let captureStartNanos = self.captureStartNanos
        let meterSink: @Sendable (UnsafePointer<Float32>, Int) -> Void = { [weak self] samples, frameCount in
            self?.updateMeters(samples: samples, frameCount: frameCount)
        }
        let chunkHandler: @Sendable () -> (@Sendable (_ data: Data) -> Void)? = { [weak self] in
            self?.onAudioChunk
        }
        let fileRef = AudioFileRef(ref: file)
        let thread = Thread {
            Self.workerLoop(
                ring: ring,
                file: fileRef,
                packetSemaphore: semaphore,
                done: done,
                stopRequested: stopRequested,
                captureStartNanos: captureStartNanos,
                logger: logger,
                meterSink: meterSink,
                chunkHandler: chunkHandler
            )
        }
        thread.name = "com.fightingentropy.voiceink.coreaudio-drain"
        thread.qualityOfService = .userInteractive
        thread.start()
        workerThread = thread
    }

    private static func workerLoop(
        ring: AudioPacketRing,
        file fileRef: AudioFileRef,
        packetSemaphore: DispatchSemaphore,
        done: DispatchSemaphore,
        stopRequested: ManagedAtomic<Bool>,
        captureStartNanos: ManagedAtomic<UInt64>,
        logger: Logger,
        meterSink: @Sendable (UnsafePointer<Float32>, Int) -> Void,
        chunkHandler: @Sendable () -> (@Sendable (_ data: Data) -> Void)?
    ) {
        let file = fileRef.ref
        let resampler = MonoResampler16k(logger: logger)
        var framesCaptured: UInt64 = 0
        var framesWritten: UInt64 = 0
        var firstChunkDelivered = false

        func deliver(_ buffer: AVAudioPCMBuffer) {
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0, let channel = buffer.int16ChannelData?[0] else { return }

            var outputBufferList = AudioBufferList(
                mNumberBuffers: 1,
                mBuffers: AudioBuffer(
                    mNumberChannels: 1,
                    mDataByteSize: UInt32(frameCount * MemoryLayout<Int16>.size),
                    mData: UnsafeMutableRawPointer(channel)
                )
            )
            let writeStatus = ExtAudioFileWrite(file, UInt32(frameCount), &outputBufferList)
            if writeStatus != noErr {
                logger.error("🎙️ ExtAudioFileWrite failed with status: \(writeStatus, privacy: .public)")
            }
            framesWritten &+= UInt64(frameCount)

            if !firstChunkDelivered {
                firstChunkDelivered = true
                let elapsedNanos = DispatchTime.now().uptimeNanoseconds &- captureStartNanos.load(ordering: .acquiring)
                let elapsedMs = Double(elapsedNanos) / 1_000_000.0
                logger.notice("🎙️ First audio chunk delivered \(String(format: "%.1f", elapsedMs), privacy: .public)ms after AudioOutputUnitStart")
            }

            if let handler = chunkHandler() {
                handler(Data(bytes: channel, count: frameCount * MemoryLayout<Int16>.size))
            }
        }

        while true {
            while let packet = ring.peek() {
                framesCaptured &+= UInt64(packet.frameCount)
                meterSink(packet.samples, packet.frameCount)
                resampler.process(
                    samples: packet.samples,
                    frameCount: packet.frameCount,
                    sampleRate: packet.sampleRate,
                    sink: deliver
                )
                ring.consume()
            }

            if stopRequested.load(ordering: .acquiring), ring.isEmpty {
                break
            }
            _ = packetSemaphore.wait(timeout: .now() + .milliseconds(100))
        }

        resampler.flush(sink: deliver)
        ExtAudioFileDispose(file)

        let dropped = ring.droppedPackets.load(ordering: .relaxed)
        logger.notice("🎙️ Recording finished: capturedFrames=\(framesCaptured, privacy: .public), writtenFrames=\(framesWritten, privacy: .public), droppedPackets=\(dropped, privacy: .public), tailFramesFlushed=\(resampler.tailFramesFlushed, privacy: .public)")

        done.signal()
    }

    private func updateMeters(samples: UnsafePointer<Float32>, frameCount: Int) {
        guard frameCount > 0 else { return }

        var sum: Float = 0.0
        var peak: Float = 0.0

        for i in 0..<frameCount {
            let sample = abs(samples[i])
            sum += sample * sample
            if sample > peak {
                peak = sample
            }
        }

        let rms = sqrt(sum / Float(frameCount))
        let avgDb = 20.0 * log10(max(rms, 0.000001))
        let peakDb = 20.0 * log10(max(peak, 0.000001))

        meterLock.lock()
        _averagePower = avgDb
        _peakPower = peakDb
        meterLock.unlock()
    }

    // MARK: - Device Info Logging

    private func logDeviceDetails(deviceID: AudioDeviceID) {
        // Get device name
        let deviceName = getDeviceStringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceNameCFString) ?? "Unknown"

        // Get device UID
        let deviceUID = getDeviceStringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceUID) ?? "Unknown"

        // Get transport type
        let transportType = getTransportType(deviceID: deviceID)

        // Get manufacturer
        let manufacturer = getDeviceStringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceManufacturerCFString) ?? "Unknown"

        logger.notice("🎙️ Device info: name=\(deviceName, privacy: .public), uid=\(deviceUID, privacy: .public)")
        logger.notice("🎙️ Device details: transport=\(transportType, privacy: .public), manufacturer=\(manufacturer, privacy: .public)")

        // Get buffer frame size
        if let bufferSize = getBufferFrameSize(deviceID: deviceID) {
            let latencyMs = (Double(bufferSize) / 48000.0) * 1000.0 // Approximate latency assuming 48kHz
            logger.notice("🎙️ Buffer size: \(bufferSize, privacy: .public) frames, ~latency: \(String(format: "%.1f", latencyMs), privacy: .public)ms")
        }
    }

    private func getDeviceStringProperty(deviceID: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var propertySize = UInt32(MemoryLayout<CFString?>.size)
        var property: CFString?

        let status = withUnsafeMutablePointer(to: &property) { pointer in
            AudioObjectGetPropertyData(
                deviceID,
                &address,
                0,
                nil,
                &propertySize,
                UnsafeMutableRawPointer(pointer)
            )
        }

        if status == noErr, let cfString = property {
            return cfString as String
        }
        return nil
    }

    private func getTransportType(deviceID: AudioDeviceID) -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var transportType: UInt32 = 0
        var propertySize = UInt32(MemoryLayout<UInt32>.size)

        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            &transportType
        )

        if status != noErr {
            return "Unknown"
        }

        switch transportType {
        case kAudioDeviceTransportTypeBuiltIn:
            return "Built-in"
        case kAudioDeviceTransportTypeUSB:
            return "USB"
        case kAudioDeviceTransportTypeBluetooth:
            return "Bluetooth"
        case kAudioDeviceTransportTypeBluetoothLE:
            return "Bluetooth LE"
        case kAudioDeviceTransportTypeAggregate:
            return "Aggregate"
        case kAudioDeviceTransportTypeVirtual:
            return "Virtual"
        case kAudioDeviceTransportTypePCI:
            return "PCI"
        case kAudioDeviceTransportTypeFireWire:
            return "FireWire"
        case kAudioDeviceTransportTypeDisplayPort:
            return "DisplayPort"
        case kAudioDeviceTransportTypeHDMI:
            return "HDMI"
        case kAudioDeviceTransportTypeAVB:
            return "AVB"
        case kAudioDeviceTransportTypeThunderbolt:
            return "Thunderbolt"
        default:
            return "Other (\(transportType))"
        }
    }

    private func getBufferFrameSize(deviceID: AudioDeviceID) -> UInt32? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyBufferFrameSize,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var bufferSize: UInt32 = 0
        var propertySize = UInt32(MemoryLayout<UInt32>.size)

        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            &bufferSize
        )

        return status == noErr ? bufferSize : nil
    }

    /// Checks if a device is currently available using Apple's kAudioDevicePropertyDeviceIsAlive
    private func isDeviceAvailable(_ deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceIsAlive,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var isAlive: UInt32 = 0
        var propertySize = UInt32(MemoryLayout<UInt32>.size)

        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            &isAlive
        )

        return status == noErr && isAlive == 1
    }
}

// MARK: - Error Types

enum CoreAudioRecorderError: LocalizedError {
    case audioUnitNotFound
    case audioUnitNotInitialized
    case deviceNotAvailable
    case failedToCreateAudioUnit(status: OSStatus)
    case failedToEnableInput(status: OSStatus)
    case failedToDisableOutput(status: OSStatus)
    case failedToSetDevice(status: OSStatus)
    case failedToGetDeviceFormat(status: OSStatus)
    case failedToSetFormat(status: OSStatus)
    case failedToSetCallback(status: OSStatus)
    case failedToCreateFile(status: OSStatus)
    case failedToSetFileFormat(status: OSStatus)
    case failedToInitialize(status: OSStatus)
    case failedToStart(status: OSStatus)

    var errorDescription: String? {
        switch self {
        case .audioUnitNotFound:
            return "HAL Output AudioUnit not found"
        case .audioUnitNotInitialized:
            return "AudioUnit not initialized"
        case .deviceNotAvailable:
            return "Audio device is no longer available"
        case .failedToCreateAudioUnit(let status):
            return "Failed to create AudioUnit: \(status)"
        case .failedToEnableInput(let status):
            return "Failed to enable input: \(status)"
        case .failedToDisableOutput(let status):
            return "Failed to disable output: \(status)"
        case .failedToSetDevice(let status):
            return "Failed to set input device: \(status)"
        case .failedToGetDeviceFormat(let status):
            return "Failed to get device format: \(status)"
        case .failedToSetFormat(let status):
            return "Failed to set audio format: \(status)"
        case .failedToSetCallback(let status):
            return "Failed to set input callback: \(status)"
        case .failedToCreateFile(let status):
            return "Failed to create audio file: \(status)"
        case .failedToSetFileFormat(let status):
            return "Failed to set file format: \(status)"
        case .failedToInitialize(let status):
            return "Failed to initialize AudioUnit: \(status)"
        case .failedToStart(let status):
            return "Failed to start AudioUnit: \(status)"
        }
    }
}
