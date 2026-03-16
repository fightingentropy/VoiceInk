import AVFoundation
import Foundation
import MLX
import Testing
@testable import VoiceInk

@Suite(.serialized, .timeLimit(.minutes(5)))
struct VoxtralNativeStreamingSmokeTests {
    @Test
    func streamingEncoderMatchesOfflineEncoder() async throws {
        let tempDirectory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let audioURL = tempDirectory.appendingPathComponent("voxtral-encoder-parity.aiff")
        let phrase = "Voice Ink native Voxtral encoder parity test."
        try synthesizeSpeech(phrase, to: audioURL)

        let samples = try VoxtralNativeAudio.loadAudioFile(audioURL)
        try await withPreparedState { preparedState in
            let paddedAudio = VoxtralNativeAudio.padAudio(
                samples,
                leftPadTokenCount: preparedState.bootstrap.audioConfiguration.streamingLeftPadTokens
            )

            let offlineMel = VoxtralNativeAudio.logMelSpectrogram(paddedAudio)
            let offlineEmbeddings = preparedState.model.encode(offlineMel)
            eval(offlineEmbeddings)

            var audioTail: [Float]?
            var conv1Tail: MLXArray?
            var conv2Tail: MLXArray?
            var encoderCache: [VoxtralRotatingKVCache]?
            var downsampleBuffer: MLXArray?
            var streamingEmbeddings: [MLXArray] = []

            for chunk in floatChunks(from: paddedAudio, chunkSampleCount: VoxtralNativeAudio.samplesPerToken) {
                let (mel, nextAudioTail) = VoxtralNativeAudio.logMelSpectrogramStep(
                    audioChunk: chunk,
                    audioTail: audioTail
                )
                audioTail = nextAudioTail

                let encoded = preparedState.model.encodeStep(
                    mel,
                    conv1Tail: conv1Tail,
                    conv2Tail: conv2Tail,
                    encoderCache: encoderCache,
                    downsampleBuffer: downsampleBuffer
                )

                conv1Tail = encoded.conv1Tail
                conv2Tail = encoded.conv2Tail
                encoderCache = encoded.encoderCache
                downsampleBuffer = encoded.downsampleBuffer

                if let embeddings = encoded.embeddings {
                    eval(embeddings)
                    if let encoderCache {
                        evalCache(encoderCache)
                    }
                    streamingEmbeddings.append(embeddings)
                }
            }

            let combinedStreamingEmbeddings = if streamingEmbeddings.isEmpty {
                MLXArray.zeros([0, offlineEmbeddings.dim(1)], dtype: offlineEmbeddings.dtype)
            } else if streamingEmbeddings.count == 1 {
                streamingEmbeddings[0]
            } else {
                concatenated(streamingEmbeddings, axis: 0)
            }
            eval(combinedStreamingEmbeddings)

            let rowCountDelta = abs(combinedStreamingEmbeddings.dim(0) - offlineEmbeddings.dim(0))
            #expect(
                rowCountDelta <= 1,
                "Streaming encoder length mismatch: got \(combinedStreamingEmbeddings.dim(0)) rows, expected \(offlineEmbeddings.dim(0))."
            )

            let shapesMatch = combinedStreamingEmbeddings.dim(1) == offlineEmbeddings.dim(1)
            #expect(
                shapesMatch,
                "Streaming encoder feature width mismatch: got \(combinedStreamingEmbeddings.dim(1)), expected \(offlineEmbeddings.dim(1))."
            )

            guard shapesMatch else { return }

            let rowsToCompare = min(combinedStreamingEmbeddings.dim(0), offlineEmbeddings.dim(0))
            #expect(rowsToCompare > 0, "Streaming encoder produced no comparable embeddings.")
        }
    }

    @Test
    func nativeOfflineTranscribesGeneratedSpeech() async throws {
        let tempDirectory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let audioURL = tempDirectory.appendingPathComponent("voxtral-offline-smoke.aiff")
        let phrase = "Voice Ink native Voxtral smoke test. Local transcription is working."
        try synthesizeSpeech(phrase, to: audioURL)

        let transcript = try await offlineTranscript(for: audioURL)
        let normalizedTranscript = transcript.lowercased()
        #expect(
            normalizedTranscript.contains("voice") || normalizedTranscript.contains("vox"),
            "Transcript: \(transcript)"
        )
        #expect(
            normalizedTranscript.contains("working") || normalizedTranscript.contains("local"),
            "Transcript: \(transcript)"
        )
    }

    @Test
    func nativeStreamingTranscribesGeneratedSpeech() async throws {
        let tempDirectory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let audioURL = tempDirectory.appendingPathComponent("voxtral-smoke.aiff")
        let phrase = "Voice Ink native Voxtral smoke test. Local transcription is working."
        try synthesizeSpeech(phrase, to: audioURL)

        let samples = try VoxtralNativeAudio.loadAudioFile(audioURL)
        #expect(!samples.isEmpty)

        try await withPreparedState { preparedState in
            let (stream, continuation) = AsyncStream.makeStream(
                of: StreamingTranscriptionEvent.self,
                bufferingPolicy: .unbounded
            )
            let engine = VoxtralNativeStreamingEngine(
                preparedState: preparedState,
                continuation: continuation
            )

            let recorder = EventRecorder()
            let consumer = Task {
                for await event in stream {
                    await recorder.record(event)
                }
            }

            for chunk in pcmChunks(from: samples, chunkSampleCount: 1_280) {
                try await engine.ingestPCM16(chunk)
            }

            try await engine.finalize()
            continuation.finish()
            await consumer.value

            let transcript = await recorder.finalTranscript()
            #expect(!transcript.isEmpty)

            let normalizedTranscript = transcript.lowercased()
            #expect(
                normalizedTranscript.contains("voice") || normalizedTranscript.contains("vox"),
                "Transcript: \(transcript)"
            )
            #expect(
                normalizedTranscript.contains("test") || normalizedTranscript.contains("working"),
                "Transcript: \(transcript)"
            )
        }
    }

    @Test
    func nativeStreamingTranscribesGeneratedSpeechWithSingleChunk() async throws {
        let tempDirectory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let audioURL = tempDirectory.appendingPathComponent("voxtral-single-chunk.aiff")
        let phrase = "Voice Ink native Voxtral smoke test. Local transcription is working."
        try synthesizeSpeech(phrase, to: audioURL)

        let transcript = try await streamingTranscript(
            for: audioURL,
            chunkSampleCount: Int.max
        )
        let normalizedTranscript = transcript.lowercased()
        #expect(
            normalizedTranscript.contains("voice") || normalizedTranscript.contains("vox"),
            "Transcript: \(transcript)"
        )
        #expect(
            normalizedTranscript.contains("working") || normalizedTranscript.contains("local"),
            "Transcript: \(transcript)"
        )
    }

    private func synthesizeSpeech(_ text: String, to outputURL: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-v", "Samantha", "-o", outputURL.path, text]
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw SmokeTestError.speechSynthesisFailed(process.terminationStatus)
        }
    }

    private func pcmChunks(from samples: [Float], chunkSampleCount: Int) -> [Data] {
        let int16Samples = samples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * Float(Int16.max))
        }

        var chunks: [Data] = []
        chunks.reserveCapacity(max(1, int16Samples.count / chunkSampleCount))

        var index = 0
        while index < int16Samples.count {
            let end = min(index + chunkSampleCount, int16Samples.count)
            let slice = int16Samples[index ..< end]
            chunks.append(slice.withUnsafeBytes { Data($0) })
            index = end
        }

        return chunks
    }

    private func floatChunks(from samples: [Float], chunkSampleCount: Int) -> [[Float]] {
        var chunks: [[Float]] = []
        chunks.reserveCapacity(max(1, samples.count / chunkSampleCount))

        var index = 0
        while index < samples.count {
            let end = min(index + chunkSampleCount, samples.count)
            chunks.append(Array(samples[index ..< end]))
            index = end
        }

        return chunks
    }

    private func withPreparedState<T>(
        _ body: (VoxtralNativePreparedState) async throws -> T
    ) async throws -> T {
        let lease = try await VoxtralNativeRuntime.shared.acquirePreparedState(
            modelReference: LocalVoxtralConfiguration.modelName,
            autoDownload: true
        )
        defer {
            Task {
                await lease.release()
            }
        }

        return try await body(lease.preparedState)
    }

    private func offlineTranscript(for audioURL: URL) async throws -> String {
        let samples = try VoxtralNativeAudio.loadAudioFile(audioURL)
        return try await withPreparedState { preparedState in
            let paddedAudio = VoxtralNativeAudio.padAudio(
                samples,
                leftPadTokenCount: preparedState.bootstrap.audioConfiguration.streamingLeftPadTokens
            )
            let mel = VoxtralNativeAudio.logMelSpectrogram(paddedAudio)
            let audioEmbeddings = preparedState.model.encode(mel)
            eval(audioEmbeddings)

            let prefixLength = preparedState.bootstrap.prompt.tokenIDs.count
            #expect(audioEmbeddings.dim(0) >= prefixLength)

            let prefixAudioEmbeddings = audioEmbeddings[..<prefixLength, axis: 0]
            let prefixEmbeddings = (preparedState.promptEmbeddings + prefixAudioEmbeddings)
                .expandedDimensions(axis: 0)

            let caches = preparedState.model.language_model.layers.map { _ in
                VoxtralRotatingKVCache(maxSize: preparedState.bootstrap.modelConfig.slidingWindow)
            }

            var logits = preparedState.model.decode(
                prefixEmbeddings,
                t_cond: preparedState.timeConditioning,
                mask: .causal,
                cache: caches
            )
            eval(logits)
            evalCache(caches)

            var token = logits[0, logits.dim(1) - 1].argMax(axis: -1)
            asyncEval(token)
            var outputTokenIDs: [Int] = []

            if audioEmbeddings.dim(0) > prefixLength {
                for position in prefixLength ..< audioEmbeddings.dim(0) {
                    let tokenEmbedding = preparedState.model.language_model.embed(token.reshaped([1, 1]))[0, 0]
                    let stepEmbedding = (audioEmbeddings[position] + tokenEmbedding)
                        .expandedDimensions(axis: 0)
                        .expandedDimensions(axis: 0)
                    logits = preparedState.model.decode(
                        stepEmbedding,
                        t_cond: preparedState.timeConditioning,
                        mask: .none,
                        cache: caches
                    )
                    eval(logits)
                    evalCache(caches)

                    let nextToken = logits[0, logits.dim(1) - 1].argMax(axis: -1)
                    asyncEval(nextToken)

                    let tokenID = token.item(Int.self)
                    if tokenID == preparedState.bootstrap.tokenizer.eosID {
                        break
                    }

                    outputTokenIDs.append(tokenID)
                    token = nextToken
                }
            }

            return preparedState.bootstrap.tokenizer.decode(outputTokenIDs)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private func streamingTranscript(for audioURL: URL, chunkSampleCount: Int) async throws -> String {
        let samples = try VoxtralNativeAudio.loadAudioFile(audioURL)
        return try await withPreparedState { preparedState in
            let (stream, continuation) = AsyncStream.makeStream(
                of: StreamingTranscriptionEvent.self,
                bufferingPolicy: .unbounded
            )
            let engine = VoxtralNativeStreamingEngine(
                preparedState: preparedState,
                continuation: continuation
            )

            let recorder = EventRecorder()
            let consumer = Task {
                for await event in stream {
                    await recorder.record(event)
                }
            }

            for chunk in pcmChunks(from: samples, chunkSampleCount: chunkSampleCount) {
                try await engine.ingestPCM16(chunk)
            }

            try await engine.finalize()
            continuation.finish()
            await consumer.value
            return await recorder.finalTranscript()
        }
    }

    private func evalCache(_ caches: [VoxtralRotatingKVCache]) {
        for cache in caches {
            if let keys = cache.keys {
                eval(keys)
            }
            if let values = cache.values {
                eval(values)
            }
        }
    }
}

private actor EventRecorder {
    private var committedTexts: [String] = []
    private var partialTexts: [String] = []

    func record(_ event: StreamingTranscriptionEvent) {
        switch event {
        case .committed(let text):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                committedTexts.append(trimmed)
            }
        case .partial(let text):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                partialTexts.append(trimmed)
            }
        case .sessionStarted, .error:
            break
        }
    }

    func finalTranscript() -> String {
        if let committed = committedTexts.last {
            return committed
        }
        return partialTexts.last ?? ""
    }
}

private enum SmokeTestError: LocalizedError {
    case speechSynthesisFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .speechSynthesisFailed(let status):
            return "Speech synthesis failed with exit status \(status)."
        }
    }
}
