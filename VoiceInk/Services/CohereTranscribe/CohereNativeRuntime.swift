import Foundation
import MLX

struct CohereNativeBootstrap: Sendable {
    let modelDirectory: URL
    let config: CohereNativeModelConfig
    let tokenizer: CohereNativeTokenizer
    let promptText: String
    let promptTokenIDs: [Int]
    let promptTokenIDs32: [Int32]
    let featureExtractor: CohereNativeFeatureExtractor
}

struct CohereNativeWarmupSummary: Sendable {
    let modelDirectory: URL
    let promptTokenCount: Int
    let vocabularySize: Int
    let featureShape: [Int]
    let encoderOutputShape: [Int]
    let decoderLogitsShape: [Int]
    let encodedLengths: [Int]
    let parameterCount: Int
}

enum CohereNativeGenerationStopReason: Sendable {
    case endOfText
    case noSpeech
    case maxNewTokens
}

struct CohereNativeGenerationResult: Sendable {
    let text: String
    let generatedTokenIDs: [Int]
    let stopTokenID: Int?
    let stopReason: CohereNativeGenerationStopReason
    let encoderOutputShape: [Int]
    let lastLogitsShape: [Int]
    let audioWasTruncated: Bool
}

actor CohereNativeRuntime {
    static let shared = CohereNativeRuntime()
    private static let minimumDefaultNewTokens = 48
    private static let maximumDefaultNewTokens = 192
    private static let estimatedTokensPerSecond = 5.0
    private static let generationSlackTokens = 16

    private var bootstraps: [String: CohereNativeBootstrap] = [:]
    private var preparedStates: [String: CohereNativePreparedState] = [:]

    func prepareBootstrap(
        modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository,
        language: String = LocalCohereTranscribeConfiguration.fallbackLanguage,
        punctuation: Bool = true,
        autoDownload: Bool = false
    ) async throws -> CohereNativeBootstrap {
        let cacheKey = "\(modelReference)|\(language.lowercased())|\(punctuation)"
        if let bootstrap = bootstraps[cacheKey] {
            return bootstrap
        }

        let modelDirectory = try await CohereNativeModelManager.shared.preparedModelDirectory(
            for: modelReference,
            autoDownload: autoDownload
        )

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(CohereNativeModelConfig.self, from: Data(contentsOf: configURL))
        let tokenizer = try CohereNativeTokenizer(
            modelURL: modelDirectory.appendingPathComponent("tokenizer.model")
        )

        let resolvedLanguage = config.supportedLanguages.contains(language) ? language : LocalCohereTranscribeConfiguration.fallbackLanguage
        let promptText = CohereNativePromptBuilder.buildPrompt(language: resolvedLanguage, punctuation: punctuation)
        let promptTokenIDs = tokenizer.encode(promptText)
        let promptTokenIDs32 = promptTokenIDs.map(Int32.init)
        let featureExtractor = CohereNativeFeatureExtractor(configuration: config.audioConfiguration)

        let bootstrap = CohereNativeBootstrap(
            modelDirectory: modelDirectory,
            config: config,
            tokenizer: tokenizer,
            promptText: promptText,
            promptTokenIDs: promptTokenIDs,
            promptTokenIDs32: promptTokenIDs32,
            featureExtractor: featureExtractor
        )
        bootstraps[cacheKey] = bootstrap
        return bootstrap
    }

    func warmupModel(
        modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository,
        language: String = LocalCohereTranscribeConfiguration.fallbackLanguage,
        punctuation: Bool = true,
        autoDownload: Bool = false
    ) async throws -> CohereNativeWarmupSummary {
        let bootstrap = try await prepareBootstrap(
            modelReference: modelReference,
            language: language,
            punctuation: punctuation,
            autoDownload: autoDownload
        )
        let preparedState = try await preparedState(
            for: modelReference,
            language: language,
            punctuation: punctuation,
            autoDownload: autoDownload
        )

        let silenceDuration = 1
        let silence = Array(
            repeating: Float.zero,
            count: bootstrap.config.audioConfiguration.sampleRate * silenceDuration
        )
        let features = bootstrap.featureExtractor.extractLogMelFeatures(from: silence)

        return CohereNativeWarmupSummary(
            modelDirectory: bootstrap.modelDirectory,
            promptTokenCount: bootstrap.promptTokenIDs.count,
            vocabularySize: bootstrap.tokenizer.vocabularySize,
            featureShape: features.shape,
            encoderOutputShape: preparedState.summary.outputShape,
            decoderLogitsShape: preparedState.summary.decoderLogitsShape,
            encodedLengths: preparedState.summary.encodedLengths,
            parameterCount: preparedState.summary.parameterCount
        )
    }

    func transcribe(
        audioSamples: [Float],
        sampleRate: Int,
        modelReference: String = LocalCohereTranscribeConfiguration.nativeModelRepository,
        language: String = LocalCohereTranscribeConfiguration.fallbackLanguage,
        punctuation: Bool = true,
        autoDownload: Bool = false,
        maxNewTokens requestedMaxNewTokens: Int? = nil
    ) async throws -> CohereNativeGenerationResult {
        let bootstrap = try await prepareBootstrap(
            modelReference: modelReference,
            language: language,
            punctuation: punctuation,
            autoDownload: autoDownload
        )
        let preparedState = try await preparedState(
            for: modelReference,
            language: language,
            punctuation: punctuation,
            autoDownload: autoDownload
        )

        let eosTokenID = bootstrap.tokenizer.tokenID(for: CohereNativePromptBuilder.endOfTextToken)
            ?? bootstrap.tokenizer.tokenID(for: "</s>")
        guard let eosTokenID else {
            throw CohereNativeModelError.missingModelAssets
        }
        let eosTokenID32 = Int32(eosTokenID)
        let noSpeechTokenID32 = bootstrap.tokenizer.tokenID(for: CohereNativePromptBuilder.noSpeechToken).map(Int32.init)

        return try transcribe(
            audioSamples: audioSamples,
            sampleRate: sampleRate,
            bootstrap: bootstrap,
            preparedState: preparedState,
            eosTokenID32: eosTokenID32,
            noSpeechTokenID32: noSpeechTokenID32,
            requestedMaxNewTokens: requestedMaxNewTokens
        )
    }

    private func transcribe(
        audioSamples: [Float],
        sampleRate: Int,
        bootstrap: CohereNativeBootstrap,
        preparedState: CohereNativePreparedState,
        eosTokenID32: Int32,
        noSpeechTokenID32: Int32?,
        requestedMaxNewTokens: Int?
    ) throws -> CohereNativeGenerationResult {
        guard sampleRate == bootstrap.config.audioConfiguration.sampleRate else {
            throw CohereNativeAudioError.unsupportedFormat
        }

        let maximumSampleCount = Int(
            Double(bootstrap.config.audioConfiguration.sampleRate) * bootstrap.config.audioConfiguration.maxClipDuration
        )
        let audioWasTruncated = audioSamples.count > maximumSampleCount
        let clippedAudio = audioWasTruncated ? Array(audioSamples.prefix(maximumSampleCount)) : audioSamples

        let inputFeatures = bootstrap.featureExtractor
            .extractLogMelFeatures(from: clippedAudio)
            .expandedDimensions(axis: 0)
            .asType(preparedState.model.encoder.subsampling.computeDType)
        let (encoderHiddenStates, encodedLengths) = preparedState.model.encode(
            inputFeatures: inputFeatures,
            lengths: [inputFeatures.dim(2)]
        )
        let decoderContext = preparedState.model.prepareDecoderContext(
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )
        let decoderContextEvalArrays =
            [encoderHiddenStates, decoderContext.encoderHiddenStates, decoderContext.crossAttentionMask]
        eval(decoderContextEvalArrays)

        let promptTokenIDs = bootstrap.promptTokenIDs32
        var decodedTokenIDs = promptTokenIDs
        let maxSequenceLength = bootstrap.config.decoder.config.maxSequenceLength
        let availableNewTokens = max(1, maxSequenceLength - decodedTokenIDs.count)
        let defaultMaxNewTokens = Self.recommendedMaxNewTokens(
            audioSampleCount: clippedAudio.count,
            sampleRate: bootstrap.config.audioConfiguration.sampleRate,
            promptTokenCount: decodedTokenIDs.count,
            maxSequenceLength: maxSequenceLength
        )
        let maxNewTokens = min(requestedMaxNewTokens ?? defaultMaxNewTokens, availableNewTokens)
        decodedTokenIDs.reserveCapacity(promptTokenIDs.count + maxNewTokens)

        var stopReason: CohereNativeGenerationStopReason = .maxNewTokens
        var stopTokenID: Int?
        var lastLogitsShape = preparedState.summary.decoderLogitsShape
        let decoderCache = preparedState.decoderCache
        decoderCache.resetForTranscription()
        var currentLogits: MLXArray?

        currentLogits = preparedState.model.prefill(
            inputIDs: preparedState.promptInputIDs,
            positions: preparedState.promptPositions,
            decoderContext: decoderContext,
            cache: decoderCache,
            applyLogSoftmax: false
        )
        if let currentLogits {
            eval(currentLogits, decoderCache.arraysForEval())
            lastLogitsShape = currentLogits.shape
        }

        guard var currentLogits else {
            throw CohereNativeModelError.missingModelAssets
        }

        for generationIndex in 0 ..< maxNewTokens {
            let lastLogits = currentLogits[0, currentLogits.dim(1) - 1]
            let nextTokenID = lastLogits.argMax(axis: -1).item(Int32.self)

            if nextTokenID == eosTokenID32 {
                stopReason = .endOfText
                stopTokenID = Int(nextTokenID)
                break
            }

            if let noSpeechTokenID32, decodedTokenIDs.count == promptTokenIDs.count, nextTokenID == noSpeechTokenID32 {
                stopReason = .noSpeech
                stopTokenID = Int(nextTokenID)
                break
            }

            decodedTokenIDs.append(nextTokenID)

            if generationIndex < maxNewTokens - 1 {
                currentLogits = preparedState.model.decodeStep(
                    inputIDs: MLXArray([nextTokenID], [1, 1]),
                    positions: preparedState.singleStepPositions[decodedTokenIDs.count - 1],
                    decoderContext: decoderContext,
                    cache: decoderCache,
                    applyLogSoftmax: false
                )
                eval(currentLogits, decoderCache.arraysForEval())
                lastLogitsShape = currentLogits.shape
            }
        }

        let generatedTokenIDs = decodedTokenIDs.dropFirst(promptTokenIDs.count).map(Int.init)
        let text = bootstrap.tokenizer.decode(generatedTokenIDs)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return CohereNativeGenerationResult(
            text: text,
            generatedTokenIDs: generatedTokenIDs,
            stopTokenID: stopTokenID,
            stopReason: stopReason,
            encoderOutputShape: encoderHiddenStates.shape,
            lastLogitsShape: lastLogitsShape,
            audioWasTruncated: audioWasTruncated
        )
    }

    func clearPreparedBootstraps() {
        bootstraps.removeAll()
        preparedStates.removeAll()
        GPU.clearCache()
    }

    private func preparedState(
        for modelReference: String,
        language: String,
        punctuation: Bool,
        autoDownload: Bool
    ) async throws -> CohereNativePreparedState {
        let cacheKey = "\(modelReference)|\(language.lowercased())|\(punctuation)"
        if let cachedPreparedState = preparedStates[cacheKey] {
            return cachedPreparedState
        }

        let bootstrap = try await prepareBootstrap(
            modelReference: modelReference,
            language: language,
            punctuation: punctuation,
            autoDownload: autoDownload
        )
        let loadedPreparedState = try CohereNativeEncoderLoader.loadPreparedState(from: bootstrap)
        preparedStates[cacheKey] = loadedPreparedState
        return loadedPreparedState
    }

    static func recommendedMaxNewTokens(
        audioSampleCount: Int,
        sampleRate: Int,
        promptTokenCount: Int,
        maxSequenceLength: Int
    ) -> Int {
        let availableNewTokens = max(1, maxSequenceLength - promptTokenCount)
        guard sampleRate > 0 else {
            return min(minimumDefaultNewTokens, availableNewTokens)
        }

        let durationSeconds = Double(audioSampleCount) / Double(sampleRate)
        let estimatedTokens = Int(ceil(durationSeconds * estimatedTokensPerSecond)) + generationSlackTokens
        let boundedEstimate = min(maximumDefaultNewTokens, estimatedTokens)
        let defaultBudget = max(minimumDefaultNewTokens, boundedEstimate)
        return min(defaultBudget, availableNewTokens)
    }
}
