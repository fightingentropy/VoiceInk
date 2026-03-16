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
