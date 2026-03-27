import Foundation
import MLX

struct CohereNativeBootstrap: Sendable {
    let modelDirectory: URL
    let config: CohereNativeModelConfig
    let tokenizer: CohereNativeTokenizer
    let promptText: String
    let promptTokenIDs: [Int]
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
        let featureExtractor = CohereNativeFeatureExtractor(configuration: config.audioConfiguration)

        let bootstrap = CohereNativeBootstrap(
            modelDirectory: modelDirectory,
            config: config,
            tokenizer: tokenizer,
            promptText: promptText,
            promptTokenIDs: promptTokenIDs,
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
        audioURL: URL,
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
        let noSpeechTokenID = bootstrap.tokenizer.tokenID(for: CohereNativePromptBuilder.noSpeechToken)

        let loadedAudio = try CohereNativeFeatureExtractor.loadAudioFile(
            audioURL,
            sampleRate: bootstrap.config.audioConfiguration.sampleRate
        )
        let maximumSampleCount = Int(
            Double(bootstrap.config.audioConfiguration.sampleRate) * bootstrap.config.audioConfiguration.maxClipDuration
        )
        let audioWasTruncated = loadedAudio.count > maximumSampleCount
        let clippedAudio = audioWasTruncated ? Array(loadedAudio.prefix(maximumSampleCount)) : loadedAudio

        let inputFeatures = bootstrap.featureExtractor
            .extractLogMelFeatures(from: clippedAudio)
            .expandedDimensions(axis: 0)
            .asType(preparedState.model.encoder.subsampling.out.weight.dtype)
        let lengths = [inputFeatures.dim(2)]
        let (encoderHiddenStates, encodedLengths) = preparedState.model.encode(
            inputFeatures: inputFeatures,
            lengths: lengths
        )
        eval(encoderHiddenStates)

        var decodedTokenIDs = bootstrap.promptTokenIDs
        let maxSequenceLength = bootstrap.config.decoder.config.maxSequenceLength
        let availableNewTokens = max(1, maxSequenceLength - decodedTokenIDs.count)
        let maxNewTokens = min(requestedMaxNewTokens ?? availableNewTokens, availableNewTokens)

        var stopReason: CohereNativeGenerationStopReason = .maxNewTokens
        var stopTokenID: Int?
        var lastLogitsShape = preparedState.summary.decoderLogitsShape

        for _ in 0 ..< maxNewTokens {
            let inputIDs = MLXArray(decodedTokenIDs, [1, decodedTokenIDs.count]).asType(.int32)
            let logits = preparedState.model.decode(
                inputIDs: inputIDs,
                encoderHiddenStates: encoderHiddenStates,
                encodedLengths: encodedLengths
            )
            eval(logits)
            lastLogitsShape = logits.shape

            let lastLogits = logits[0, logits.dim(1) - 1]
            let nextToken = lastLogits.argMax(axis: -1)
            eval(nextToken)
            let nextTokenID = nextToken.item(Int.self)

            if nextTokenID == eosTokenID {
                stopReason = .endOfText
                stopTokenID = nextTokenID
                break
            }

            if let noSpeechTokenID, decodedTokenIDs.count == bootstrap.promptTokenIDs.count, nextTokenID == noSpeechTokenID {
                stopReason = .noSpeech
                stopTokenID = nextTokenID
                break
            }

            decodedTokenIDs.append(nextTokenID)
        }

        let generatedTokenIDs = Array(decodedTokenIDs.dropFirst(bootstrap.promptTokenIDs.count))
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
}
