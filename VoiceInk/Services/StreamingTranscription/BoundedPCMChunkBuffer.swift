import Foundation
import os

/// Thread-safe bounded buffer for PCM16 mono audio chunks.
final class BoundedPCMChunkBuffer: @unchecked Sendable {
    static let defaultCapacityBytes = 128_000

    private let capacityBytes: Int
    private let sampleAlignmentBytes: Int
    private let logger: Logger?
    private let label: String
    private let lock = NSLock()

    private var chunks: [Data] = []
    private var headIndex = 0
    private var totalBytes = 0
    private var isClosed = false
    private var didTrimAudio = false

    init(
        capacityBytes: Int = BoundedPCMChunkBuffer.defaultCapacityBytes,
        sampleAlignmentBytes: Int = MemoryLayout<Int16>.size,
        logger: Logger? = nil,
        label: String = "PCM audio"
    ) {
        self.capacityBytes = capacityBytes
        self.sampleAlignmentBytes = max(1, sampleAlignmentBytes)
        self.logger = logger
        self.label = label
    }

    var bufferedByteCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return totalBytes
    }

    var hasTrimmedAudio: Bool {
        lock.lock()
        defer { lock.unlock() }
        return didTrimAudio
    }

    var isEmpty: Bool {
        lock.lock()
        defer { lock.unlock() }
        return totalBytes == 0
    }

    func append(_ data: Data) {
        guard !data.isEmpty else { return }

        lock.lock()
        defer { lock.unlock() }

        guard !isClosed else { return }

        chunks.append(data)
        totalBytes += data.count
        trimIfNeededLocked()
    }

    func drain() -> [Data] {
        lock.lock()
        defer { lock.unlock() }

        let drained = headIndex == 0 ? chunks : Array(chunks[headIndex..<chunks.count])
        chunks.removeAll(keepingCapacity: true)
        headIndex = 0
        totalBytes = 0
        return drained
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }

        chunks.removeAll(keepingCapacity: false)
        headIndex = 0
        totalBytes = 0
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        isClosed = true
    }

    private func trimIfNeededLocked() {
        guard totalBytes > capacityBytes else { return }

        if !didTrimAudio {
            didTrimAudio = true
            logger?.warning("\(self.label, privacy: .public) buffer exceeded \(self.capacityBytes, privacy: .public) bytes; dropping oldest audio.")
        }

        var excessBytes = totalBytes - capacityBytes
        if sampleAlignmentBytes > 1 {
            let remainder = excessBytes % sampleAlignmentBytes
            if remainder != 0 {
                excessBytes += sampleAlignmentBytes - remainder
            }
        }

        while excessBytes > 0, headIndex < chunks.count {
            let chunk = chunks[headIndex]
            if chunk.count <= excessBytes {
                headIndex += 1
                totalBytes -= chunk.count
                excessBytes -= chunk.count
                continue
            }

            let trimmedChunk = chunk.dropFirst(excessBytes)
            chunks[headIndex] = Data(trimmedChunk)
            totalBytes -= excessBytes
            excessBytes = 0
        }

        compactConsumedChunksIfNeeded()

        if totalBytes > capacityBytes {
            totalBytes = capacityBytes
        }
    }

    private func compactConsumedChunksIfNeeded() {
        guard headIndex > 0 else { return }

        if headIndex >= chunks.count {
            chunks.removeAll(keepingCapacity: true)
            headIndex = 0
            return
        }

        guard headIndex > 32, headIndex * 2 >= chunks.count else { return }
        chunks.removeFirst(headIndex)
        headIndex = 0
    }
}

/// Buffers PCM chunks until a real-time consumer is installed, then replays buffered audio in order.
final class BufferedPCMChunkForwarder: @unchecked Sendable {
    private enum State {
        case buffering
        case flushing
        case live(@Sendable (Data) -> Void)
        case finished
    }

    private let lock = NSLock()
    private let buffer: BoundedPCMChunkBuffer
    private var state: State = .buffering

    init(
        capacityBytes: Int = BoundedPCMChunkBuffer.defaultCapacityBytes,
        logger: Logger? = nil,
        label: String = "Streaming startup audio"
    ) {
        self.buffer = BoundedPCMChunkBuffer(
            capacityBytes: capacityBytes,
            logger: logger,
            label: label
        )
    }

    var hasTrimmedAudio: Bool {
        buffer.hasTrimmedAudio
    }

    func send(_ data: Data) {
        guard !data.isEmpty else { return }

        var liveConsumer: (@Sendable (Data) -> Void)?

        lock.lock()
        switch state {
        case .buffering, .flushing:
            buffer.append(data)
        case .live(let consumer):
            liveConsumer = consumer
        case .finished:
            break
        }
        lock.unlock()

        liveConsumer?(data)
    }

    func installConsumer(_ consumer: @escaping @Sendable (Data) -> Void) {
        while true {
            let batch: [Data]

            lock.lock()
            switch state {
            case .finished:
                lock.unlock()
                return
            case .live:
                lock.unlock()
                return
            case .buffering:
                state = .flushing
                batch = buffer.drain()
            case .flushing:
                batch = buffer.drain()
            }

            if batch.isEmpty {
                state = .live(consumer)
                lock.unlock()
                return
            }
            lock.unlock()

            for chunk in batch {
                consumer(chunk)
            }
        }
    }

    func finish() {
        lock.lock()
        state = .finished
        lock.unlock()
        buffer.close()
        buffer.clear()
    }
}
