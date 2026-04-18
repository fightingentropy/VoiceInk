import Foundation
import Testing
@testable import VoiceInk

struct LocalTranscriptionHotPathTests {
    @Test
    func boundedPCMChunkBufferTrimsOldestAudioAndPreservesOrder() {
        let buffer = BoundedPCMChunkBuffer(capacityBytes: 10, sampleAlignmentBytes: 2)

        buffer.append(Data([0, 1, 2, 3]))
        buffer.append(Data([4, 5, 6, 7]))
        buffer.append(Data([8, 9, 10, 11]))

        #expect(buffer.bufferedByteCount == 10)
        #expect(buffer.hasTrimmedAudio)

        let drainedBytes = buffer.drain().flatMap { Array($0) }
        #expect(drainedBytes == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    }

    @Test
    func boundedPCMChunkBufferNeverExceedsCapacityForLargeSingleChunk() {
        let buffer = BoundedPCMChunkBuffer(capacityBytes: 8, sampleAlignmentBytes: 2)

        buffer.append(Data(Array<UInt8>(0 ..< 20)))

        #expect(buffer.bufferedByteCount == 8)
        #expect(buffer.hasTrimmedAudio)

        let drainedBytes = buffer.drain().flatMap { Array($0) }
        #expect(drainedBytes == [12, 13, 14, 15, 16, 17, 18, 19])
    }

    @Test
    func bufferedForwarderReplaysBoundedStartupAudioBeforeGoingLive() async throws {
        let forwarder = BufferedPCMChunkForwarder(capacityBytes: 8)
        let collector = ByteCollector()

        forwarder.send(Data([0, 1, 2, 3]))
        forwarder.send(Data([4, 5, 6, 7]))
        forwarder.send(Data([8, 9, 10, 11]))

        let installTask = Task {
            try await Task.sleep(for: .milliseconds(50))
            forwarder.installConsumer { data in
                collector.append(data)
            }
        }

        try await installTask.value
        forwarder.send(Data([12, 13]))

        #expect(forwarder.hasTrimmedAudio)
        #expect(collector.snapshot() == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    }

    @Test
    func coherePCMDecoderNormalizesRecorderBytes() throws {
        let pcm = Data([
            0x00, 0x00,
            0xFF, 0x7F,
            0x00, 0x80,
        ])

        let decoded = try CohereNativeFeatureExtractor.decodePCM16Mono(pcm)

        #expect(decoded.count == 3)
        #expect(decoded[0] == 0)
        #expect(decoded[1] == 1)
        #expect(decoded[2] == -1)
    }

    @Test
    func localWhisperPCMDecoderNormalizesRecorderBytes() {
        let pcm = Data([
            0x00, 0x00,
            0xFF, 0x7F,
            0x00, 0x80,
        ])

        let decoded = LocalTranscriptionService.decodePCM16Mono(pcm)

        #expect(decoded.count == 3)
        #expect(decoded[0] == 0)
        #expect(decoded[1] == 1)
        #expect(decoded[2] == -1)
    }

    /// Original `stride.map`-based decoder kept around only so the
    /// benchmark below can prove the replacement is actually faster.
    private static func decodePCM16MonoLegacy(_ pcm: Data) -> [Float] {
        guard !pcm.isEmpty else { return [] }
        return stride(from: 0, to: pcm.count - 1, by: 2).map { offset in
            pcm[offset ..< offset + 2].withUnsafeBytes { bytes in
                let short = Int16(littleEndian: bytes.load(as: Int16.self))
                return max(-1.0, min(Float(short) / 32767.0, 1.0))
            }
        }
    }

    /// Micro-benchmark: decode 30 s of 16 kHz mono PCM16 with the current
    /// implementation AND the legacy `stride.map` version, then print a
    /// side-by-side summary with the speedup ratio. Not an assertion.
    @Test
    func localWhisperPCMDecoderThroughputBenchmark() {
        let sampleCount = 16_000 * 30 // 30 seconds of 16 kHz mono audio.
        var bytes = [UInt8](repeating: 0, count: sampleCount * 2)
        for i in 0..<sampleCount {
            let value = Int16(truncatingIfNeeded: (i * 7919) & 0x7FFF) &- 16_384
            let le = value.littleEndian
            bytes[i * 2]     = UInt8(truncatingIfNeeded: le)
            bytes[i * 2 + 1] = UInt8(truncatingIfNeeded: le >> 8)
        }
        let pcm = Data(bytes)

        _ = LocalTranscriptionService.decodePCM16Mono(pcm) // warmup
        _ = Self.decodePCM16MonoLegacy(pcm)

        let iterations = 20
        let clock = ContinuousClock()

        func measure(_ block: () -> [Float]) -> (median: UInt64, min: UInt64) {
            var timingsNs: [UInt64] = []
            timingsNs.reserveCapacity(iterations)
            for _ in 0..<iterations {
                let start = clock.now
                let decoded = block()
                let elapsed = clock.now - start
                #expect(decoded.count == sampleCount)
                let comp = elapsed.components
                let ns = UInt64(comp.seconds) * 1_000_000_000 + UInt64(comp.attoseconds / 1_000_000_000)
                timingsNs.append(ns)
            }
            timingsNs.sort()
            return (timingsNs[timingsNs.count / 2], timingsNs.first!)
        }

        let current = measure { LocalTranscriptionService.decodePCM16Mono(pcm) }
        let legacy = measure { Self.decodePCM16MonoLegacy(pcm) }

        let speedup = Double(legacy.median) / Double(current.median)
        let summary = String(format:
            """
            [BENCH] decodePCM16Mono (%d samples, %d iterations)
              current: median=%.2fms  min=%.2fms  (%.1f Msamples/s)
              legacy:  median=%.2fms  min=%.2fms  (%.1f Msamples/s)
              speedup: %.2fx
            """,
            sampleCount, iterations,
            Double(current.median) / 1_000_000, Double(current.min) / 1_000_000,
            1_000.0 / (Double(current.median) / Double(sampleCount)),
            Double(legacy.median) / 1_000_000, Double(legacy.min) / 1_000_000,
            1_000.0 / (Double(legacy.median) / Double(sampleCount)),
            speedup
        )
        print(summary)
        // Stdout from test runners is buffered into the xcresult bundle and not
        // relayed by `xcodebuild test`, so also drop a file the caller can read.
        try? summary.write(
            toFile: "/tmp/voiceink-bench.txt",
            atomically: true,
            encoding: .utf8
        )
    }

    @Test
    func predefinedModelsExposeWhisperTurboPreset() {
        let whisperTurbo = PredefinedModels.models.first { $0.name == "whisper-large-v3-turbo" }

        #expect(whisperTurbo != nil)
        #expect(whisperTurbo?.provider == .local)
        #expect(whisperTurbo?.displayName == "Whisper Large v3 Turbo")
        #expect((whisperTurbo as? LocalModel)?.whisperKitVariant == "openai_whisper-large-v3_turbo")
    }

    @Test
    @MainActor
    func fileTranscriptionSessionUsesPCMFastPathWhenServiceSupportsIt() async throws {
        let service = MockPCMBufferTranscriptionService()
        let session = FileTranscriptionSession(service: service)
        let model = LocalCohereTranscribeModel(
            name: "cohere-transcribe-test",
            displayName: "Cohere Test",
            size: "0 GB",
            description: "Test model",
            speed: 1,
            accuracy: 1,
            isMultilingualModel: true,
            supportedLanguages: ["en": "English"]
        )

        let callback = try await session.prepare(model: model)
        #expect(callback != nil)

        let pcm = Data([0x34, 0x12, 0x78, 0x56])
        callback?(pcm)

        let result = try await session.transcribe(audioURL: URL(fileURLWithPath: "/tmp/unused.wav"))

        #expect(result == "pcm-fast-path")
        #expect(service.pcmTranscribeCallCount == 1)
        #expect(service.lastPCMBuffer == pcm)
        #expect(service.lastSampleRate == 16_000)
    }

    @Test
    @MainActor
    func registryRejectsRecorderOnlyModelsForAudioFileTranscription() async throws {
        let registry = TranscriptionServiceRegistry(
            modelProvider: MockLocalModelProvider()
        )
        let model = LocalCohereTranscribeModel(
            name: "cohere-transcribe-test",
            displayName: "Cohere Test",
            size: "0 GB",
            description: "Test model",
            speed: 1,
            accuracy: 1,
            isMultilingualModel: true,
            supportedLanguages: ["en": "English"]
        )

        do {
            _ = try await registry.transcribe(audioURL: URL(fileURLWithPath: "/tmp/unused.wav"), model: model)
            Issue.record("Recorder-only model unexpectedly accepted file-based transcription.")
        } catch let error as TranscriptionCapabilityError {
            guard case let .audioFileInputUnsupported(modelName) = error else {
                Issue.record("Unexpected capability error: \(error.localizedDescription)")
                return
            }
            #expect(modelName == model.displayName)
        }
    }
}

@Suite(.serialized, .timeLimit(.minutes(10)))
struct VoxtralRuntimeLifecycleTests {
    @Test
    func preparedStateCanUnloadAfterLeaseRelease() async throws {
        let runtime = VoxtralNativeRuntime.shared
        _ = await runtime.unloadAllUnusedPreparedStates()

        _ = try await runtime.warmupModel(
            modelReference: LocalVoxtralConfiguration.modelName,
            autoDownload: true
        )

        let warmed = await runtime.hasPreparedState(LocalVoxtralConfiguration.modelName)
        #expect(warmed)

        let lease = try await runtime.acquirePreparedState(
            modelReference: LocalVoxtralConfiguration.modelName,
            autoDownload: true
        )

        let blockedUnload = await runtime.unloadPreparedState(LocalVoxtralConfiguration.modelName)
        #expect(!blockedUnload)

        await lease.release()

        let unloaded = await runtime.unloadPreparedState(LocalVoxtralConfiguration.modelName)
        let stillPrepared = await runtime.hasPreparedState(LocalVoxtralConfiguration.modelName)
        #expect(unloaded)
        #expect(!stillPrepared)
    }
}

private final class ByteCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes: [UInt8] = []

    func append(_ data: Data) {
        lock.lock()
        bytes.append(contentsOf: data)
        lock.unlock()
    }

    func snapshot() -> [UInt8] {
        lock.lock()
        defer { lock.unlock() }
        return bytes
    }
}

private final class MockPCMBufferTranscriptionService: PCMBufferTranscriptionService, @unchecked Sendable {
    private(set) var pcmTranscribeCallCount = 0
    private(set) var lastPCMBuffer = Data()
    private(set) var lastSampleRate: Int?

    func transcribe(recordedPCMBuffer: Data, sampleRate: Int, model: any TranscriptionModel) async throws -> String {
        _ = model
        pcmTranscribeCallCount += 1
        lastPCMBuffer = recordedPCMBuffer
        lastSampleRate = sampleRate
        return "pcm-fast-path"
    }
}

@MainActor
private final class MockLocalModelProvider: LocalModelProvider {
    let isModelLoaded = false
    let whisperKitRuntime: WhisperKitRuntime? = nil
    let loadedLocalModel: WhisperModel? = nil
    let availableModels: [WhisperModel] = []
}
