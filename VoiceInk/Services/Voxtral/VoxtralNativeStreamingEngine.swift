import Foundation
import MLX

actor VoxtralNativeStreamingEngine {
    private let preparedState: VoxtralNativePreparedState
    private let continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation
    private let eosTokenID: Int
    private let leftPadTokenCount: Int
    private let prefixLength: Int
    private let temperature: Float
    private let minimumProcessingSamples: Int

    private var audioTail: [Float]?
    private var conv1Tail: MLXArray?
    private var conv2Tail: MLXArray?
    private var encoderCache: [VoxtralRotatingKVCache]?
    private var downsampleBuffer: MLXArray?
    private var pendingAudio: [Float] = []
    private var audioEmbeddings: MLXArray?
    private var decoderCache: [VoxtralRotatingKVCache]?
    private var nextToken: MLXArray?
    private var totalAudioSamplesFed = 0
    private var totalPositionsDecoded = 0
    private var firstCycle = true
    private var prefilled = false
    private var decodedTokenIDs: [Int] = []
    private var lastEmittedPartial = ""

    init(
        preparedState: VoxtralNativePreparedState,
        continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation,
        temperature: Float = 0
    ) {
        self.preparedState = preparedState
        self.continuation = continuation
        self.eosTokenID = preparedState.bootstrap.tokenizer.eosID
        self.leftPadTokenCount = preparedState.bootstrap.audioConfiguration.streamingLeftPadTokens
        self.prefixLength = preparedState.bootstrap.prompt.tokenIDs.count
        self.temperature = temperature
        self.minimumProcessingSamples = Self.minimumProcessingSamples(
            audioConfiguration: preparedState.bootstrap.audioConfiguration
        )
    }

    func ingestPCM16(_ data: Data) throws {
        guard !data.isEmpty else { return }
        pendingAudio.append(contentsOf: VoxtralNativeAudio.decodePCM16(data))
        try processPendingAudio()
        try decodeAvailablePositions(limitBySafeTotal: true)
    }

    func finalize() throws {
        try feedFinalAudio()
        try decodeAvailablePositions(limitBySafeTotal: false)
        appendPendingTokenIfNeeded()
        continuation.yield(.committed(text: currentTranscript()))
        resetState()
        GPU.clearCache()
    }

    func cancel() {
        resetState()
    }

    private func processPendingAudio() throws {
        if firstCycle {
            guard pendingAudio.count >= minimumProcessingSamples else { return }
            let sampleCount = (pendingAudio.count / VoxtralNativeAudio.samplesPerToken) * VoxtralNativeAudio.samplesPerToken
            let realAudio = Array(pendingAudio.prefix(sampleCount))
            pendingAudio.removeFirst(sampleCount)
            totalAudioSamplesFed += sampleCount
            let padded = VoxtralNativeAudio.zeroSamples(
                count: leftPadTokenCount * VoxtralNativeAudio.samplesPerToken
            ) + realAudio
            firstCycle = false
            try appendAudioChunk(padded)
        }

        while !firstCycle && pendingAudio.count >= minimumProcessingSamples {
            let sampleCount = (pendingAudio.count / VoxtralNativeAudio.samplesPerToken) * VoxtralNativeAudio.samplesPerToken
            let chunk = Array(pendingAudio.prefix(sampleCount))
            pendingAudio.removeFirst(sampleCount)
            totalAudioSamplesFed += sampleCount
            try appendAudioChunk(chunk)
        }
    }

    private func feedFinalAudio() throws {
        let rightPadding = VoxtralNativeAudio.zeroSamples(
            count: VoxtralNativeAudio.rightPadTokenCount * VoxtralNativeAudio.samplesPerToken
        )
        let realAudio = pendingAudio
        pendingAudio.removeAll(keepingCapacity: true)
        totalAudioSamplesFed += realAudio.count

        var chunk = realAudio + rightPadding
        if firstCycle {
            chunk = VoxtralNativeAudio.zeroSamples(
                count: leftPadTokenCount * VoxtralNativeAudio.samplesPerToken
            ) + chunk
            firstCycle = false
        }

        guard !chunk.isEmpty else { return }
        try appendAudioChunk(chunk)
    }

    private func appendAudioChunk(_ chunk: [Float]) throws {
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

        if let newEmbeddings = encoded.embeddings {
            appendEmbeddings(newEmbeddings)
            try checkedEval(newEmbeddings, cacheArrays(from: encoderCache))
        }
    }

    private func appendEmbeddings(_ newEmbeddings: MLXArray) {
        if let audioEmbeddings {
            self.audioEmbeddings = concatenated([audioEmbeddings, newEmbeddings], axis: 0)
        } else {
            self.audioEmbeddings = newEmbeddings
        }
    }

    private func decodeAvailablePositions(limitBySafeTotal: Bool) throws {
        while true {
            if !prefilled {
                guard try prefillIfPossible() else { break }
                continue
            }

            guard let audioEmbeddings else { break }

            let decodableCount: Int
            if limitBySafeTotal {
                let safeTotal = leftPadTokenCount + (totalAudioSamplesFed / VoxtralNativeAudio.samplesPerToken)
                decodableCount = min(audioEmbeddings.dim(0), safeTotal - totalPositionsDecoded)
            } else {
                decodableCount = audioEmbeddings.dim(0)
            }

            guard decodableCount > 0 else { break }

            var consumed = 0
            while consumed < decodableCount {
                guard let nextToken else { break }

                let tokenEmbedding = preparedState.model.language_model.embed(nextToken.reshaped([1, 1]))[0, 0]
                let stepEmbedding = (audioEmbeddings[consumed] + tokenEmbedding)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                let logits = preparedState.model.decode(
                    stepEmbedding,
                    t_cond: preparedState.timeConditioning,
                    mask: .none,
                    cache: decoderCache
                )
                try checkedEval(logits, cacheArrays(from: decoderCache))

                let sampledToken = sample(logits)
                asyncEval(sampledToken)

                let tokenID = nextToken.item(Int.self)
                self.nextToken = sampledToken

                if tokenID == eosTokenID {
                    let committedText = currentTranscript()
                    continuation.yield(.committed(text: committedText))
                    resetState()
                    return
                }

                decodedTokenIDs.append(tokenID)
                emitPartialIfNeeded()

                consumed += 1
                if consumed % 256 == 0 {
                    GPU.clearCache()
                }
            }

            trimAudioEmbeddings(consumed)
            totalPositionsDecoded += consumed

            if consumed == 0 {
                break
            }
        }
    }

    private func prefillIfPossible() throws -> Bool {
        guard !prefilled, let audioEmbeddings, audioEmbeddings.dim(0) >= prefixLength else {
            return false
        }

        decoderCache = preparedState.model.language_model.layers.map { _ in
            VoxtralRotatingKVCache(maxSize: preparedState.bootstrap.modelConfig.slidingWindow)
        }

        let prefixAudioEmbeddings = audioEmbeddings[..<prefixLength, axis: 0]
        let prefixEmbeddings = (preparedState.promptEmbeddings + prefixAudioEmbeddings)
            .expandedDimensions(axis: 0)

        let logits = preparedState.model.decode(
            prefixEmbeddings,
            t_cond: preparedState.timeConditioning,
            mask: .causal,
            cache: decoderCache
        )
        try checkedEval(logits, cacheArrays(from: decoderCache))

        let sampledToken = sample(logits)
        asyncEval(sampledToken)
        self.nextToken = sampledToken
        self.prefilled = true
        self.totalPositionsDecoded = prefixLength
        trimAudioEmbeddings(prefixLength)
        return true
    }

    private func trimAudioEmbeddings(_ consumed: Int) {
        guard let audioEmbeddings else { return }
        let remaining = audioEmbeddings.dim(0) - consumed
        self.audioEmbeddings = remaining > 0 ? audioEmbeddings[consumed..., axis: 0] : nil
    }

    private func appendPendingTokenIfNeeded() {
        guard let nextToken else { return }
        let tokenID = nextToken.item(Int.self)
        guard tokenID != eosTokenID else { return }
        decodedTokenIDs.append(tokenID)
    }

    private func currentTranscript() -> String {
        preparedState.bootstrap.tokenizer.decode(decodedTokenIDs)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func emitPartialIfNeeded() {
        let transcript = currentTranscript()
        guard !transcript.isEmpty, transcript != lastEmittedPartial else { return }
        lastEmittedPartial = transcript
        continuation.yield(.partial(text: transcript))
    }

    private func sample(_ logits: MLXArray) -> MLXArray {
        let lastLogits = logits[0, logits.dim(1) - 1]
        if temperature <= 0 {
            return lastLogits.argMax(axis: -1)
        }

        return categorical(lastLogits / temperature, key: MLXRandom.key(0))
    }

    private func cacheArrays(from caches: [VoxtralRotatingKVCache]?) -> [MLXArray] {
        guard let caches else { return [] }
        return caches.flatMap { cache in
            [cache.keys, cache.values].compactMap { $0 }
        }
    }

    private func resetState() {
        audioTail = nil
        conv1Tail = nil
        conv2Tail = nil
        encoderCache = nil
        downsampleBuffer = nil
        pendingAudio.removeAll(keepingCapacity: true)
        audioEmbeddings = nil
        decoderCache = nil
        nextToken = nil
        totalAudioSamplesFed = 0
        totalPositionsDecoded = 0
        firstCycle = true
        prefilled = false
        decodedTokenIDs.removeAll(keepingCapacity: true)
        lastEmittedPartial = ""
    }

    private static func minimumProcessingSamples(audioConfiguration: VoxtralTekkenSpec.Audio) -> Int {
        let requestedSamples = Int(
            (Double(VoxtralNativeAudio.sampleRate) * Double(audioConfiguration.transcriptionDelayMS)) / 1000.0
        )
        let roundedSamples = ((requestedSamples + VoxtralNativeAudio.samplesPerToken - 1) / VoxtralNativeAudio.samplesPerToken)
            * VoxtralNativeAudio.samplesPerToken
        return max(VoxtralNativeAudio.samplesPerToken, roundedSamples)
    }
}
