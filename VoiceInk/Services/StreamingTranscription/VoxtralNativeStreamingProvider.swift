import Foundation
import os

final class VoxtralNativeStreamingProvider: StreamingTranscriptionProvider, @unchecked Sendable {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "VoxtralNativeStreaming")
    private var engine: VoxtralNativeStreamingEngine?
    private var nativeLease: VoxtralNativePreparedLease?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?

    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        if let nativeLease {
            Task {
                await nativeLease.release()
            }
        }
        eventsContinuation?.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        _ = language
        let modelReference = model.name == "voxtral-mini-realtime-local"
            ? LocalVoxtralConfiguration.modelName
            : model.name

        let lease = try await VoxtralNativeRuntime.shared.acquirePreparedState(
            modelReference: modelReference,
            autoDownload: true
        )
        let engine = VoxtralNativeStreamingEngine(
            preparedState: lease.preparedState,
            continuation: eventsContinuation!
        )

        nativeLease = lease
        self.engine = engine
        eventsContinuation?.yield(.sessionStarted)
        logger.notice("Using native Voxtral MLX provider for \(modelReference, privacy: .public)")
    }

    func sendAudioChunk(_ data: Data) async throws {
        guard let engine else {
            throw StreamingTranscriptionError.notConnected
        }

        do {
            try await engine.ingestPCM16(data)
        } catch {
            eventsContinuation?.yield(.error(error))
            throw error
        }
    }

    func commit() async throws {
        guard let engine else {
            throw StreamingTranscriptionError.notConnected
        }

        do {
            try await engine.finalize()
        } catch {
            eventsContinuation?.yield(.error(error))
            throw error
        }
    }

    func disconnect() async {
        if let engine {
            await engine.cancel()
        }

        if let nativeLease {
            await nativeLease.release()
            self.nativeLease = nil
        }

        engine = nil
        eventsContinuation?.finish()
    }
}
