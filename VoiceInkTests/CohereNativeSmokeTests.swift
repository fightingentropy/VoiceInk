import Foundation
import MLX
import MLXNN
import Testing
@testable import VoiceInk

@Suite(.serialized, .timeLimit(.minutes(45)))
struct CohereNativeSmokeTests {
    @Test
    func nativeRuntimeTranscribesShortBenchmarkSample() async throws {
        guard Self.shouldRunSmoke else {
            return
        }

        let sample = try benchmarkSample()
        let firstStep = try await firstStepDiagnostic(for: sample)
        let result = try await CohereNativeRuntime.shared.transcribe(
            audioURL: sample.audioURL,
            autoDownload: true,
            maxNewTokens: 96
        )

        let metrics = canonicalMetrics(reference: sample.referenceText, hypothesis: result.text)

        print("COHERE_NATIVE_SMOKE_AUDIO \(sample.audioURL.path)")
        print("COHERE_NATIVE_SMOKE_REFERENCE \(sample.referenceText)")
        print("COHERE_NATIVE_SMOKE_PROMPT_IDS \(firstStep.promptTokenIDs)")
        print("COHERE_NATIVE_SMOKE_FEATURE_SHAPE \(firstStep.featureShape)")
        print("COHERE_NATIVE_SMOKE_FEATURE_SUMMARY \(firstStep.featureSummary)")
        print("COHERE_NATIVE_SMOKE_SUBSAMPLED_SUMMARY \(firstStep.subsampledSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_INPUT_SUMMARY \(firstStep.firstLayer.inputSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_FF1_NORM_SUMMARY \(firstStep.firstLayer.feedForwardNormSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_FF1_OUTPUT_SUMMARY \(firstStep.firstLayer.feedForwardOutputSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_FF1_RESIDUAL_SUMMARY \(firstStep.firstLayer.feedForwardResidualSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_ATTN_NORM_SUMMARY \(firstStep.firstLayer.attentionNormSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_ATTENTION_SUMMARY \(firstStep.firstLayer.attentionOutputSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_ATTENTION_SCORES_SUMMARY \(firstStep.firstLayer.attentionScoreSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_CONV_NORM_SUMMARY \(firstStep.firstLayer.convolutionNormSummary)")
        print("COHERE_NATIVE_SMOKE_LAYER0_CONV_SUMMARY \(firstStep.firstLayer.convolutionOutputSummary)")
        print("COHERE_NATIVE_SMOKE_ENCODER_SUMMARY \(firstStep.encoderSummary)")
        print("COHERE_NATIVE_SMOKE_DECODER_HIDDEN_SUMMARY \(firstStep.decoderHiddenSummary)")
        print("COHERE_NATIVE_SMOKE_RAW_LOGITS_SUMMARY \(firstStep.rawLogitsSummary)")
        print("COHERE_NATIVE_SMOKE_LOGITS_SUMMARY \(firstStep.logitsSummary)")
        print("COHERE_NATIVE_SMOKE_FIRST_STEP_TOP_IDS \(firstStep.topCandidates.map(\.tokenID))")
        print("COHERE_NATIVE_SMOKE_FIRST_STEP_TOP_TOKENS \(firstStep.topCandidates.map(\.tokenText))")
        print("COHERE_NATIVE_SMOKE_FIRST_STEP_TOP_SCORES \(firstStep.topCandidates.map(\.score))")
        print("COHERE_NATIVE_SMOKE_TRANSCRIPT \(result.text)")
        print("COHERE_NATIVE_SMOKE_STOP_REASON \(String(describing: result.stopReason))")
        print("COHERE_NATIVE_SMOKE_CANONICAL_WER \(metrics.wordErrorRate)")

        #expect(!result.text.isEmpty, "Native Cohere MLX returned an empty transcript.")
        #expect(
            metrics.wordErrorRate <= 0.5,
            "Native Cohere MLX transcript diverged too far from the benchmark reference."
        )
    }

    private static var shouldRunSmoke: Bool {
        if ProcessInfo.processInfo.environment["VOICEINK_RUN_COHERE_NATIVE_SMOKE"] == "1" {
            return true
        }

        return FileManager.default.fileExists(atPath: "/tmp/voiceink-run-cohere-native-smoke")
    }

    private func benchmarkSample() throws -> BenchmarkSample {
        let manifestURL = AppStoragePaths.recentBenchmarkManifestURL
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(BenchmarkManifest.self, from: data)

        guard let selectedItem = manifest.items
            .filter({
                $0.durationSeconds > 0 &&
                    $0.durationSeconds <= 3 &&
                    FileManager.default.fileExists(atPath: $0.audioPath)
            })
            .sorted(by: { $0.durationSeconds < $1.durationSeconds })
            .first else {
            throw CohereNativeSmokeError.missingBenchmarkSample
        }

        return BenchmarkSample(
            referenceText: selectedItem.referenceText,
            audioURL: URL(fileURLWithPath: selectedItem.audioPath)
        )
    }

    private func firstStepDiagnostic(for sample: BenchmarkSample) async throws -> FirstStepDiagnostic {
        let bootstrap = try await CohereNativeRuntime.shared.prepareBootstrap(autoDownload: true)
        let preparedState = try CohereNativeEncoderLoader.loadPreparedState(from: bootstrap)

        let loadedAudio = try CohereNativeFeatureExtractor.loadAudioFile(
            sample.audioURL,
            sampleRate: bootstrap.config.audioConfiguration.sampleRate
        )
        let maximumSampleCount = Int(
            Double(bootstrap.config.audioConfiguration.sampleRate) * bootstrap.config.audioConfiguration.maxClipDuration
        )
        let clippedAudio = Array(loadedAudio.prefix(maximumSampleCount))
        let inputFeatures = bootstrap.featureExtractor
            .extractLogMelFeatures(from: clippedAudio)
            .expandedDimensions(axis: 0)
            .asType(preparedState.model.encoder.subsampling.out.weight.dtype)
        let lengths = [inputFeatures.dim(2)]
        let (subsampledStates, _) = preparedState.model.encoder.subsampling(
            inputFeatures,
            lengths: lengths
        )
        eval(subsampledStates)
        let positionalEncoding = CohereNativeRelPositionalEncoding(modelDimensions: preparedState.model.encoder.d_model)
        let (positionedStates, positionalEmbeddings) = positionalEncoding(subsampledStates)
        eval(positionedStates, positionalEmbeddings)
        let firstLayer = firstLayerDiagnostic(
            hiddenStates: positionedStates,
            positionalEmbeddings: positionalEmbeddings,
            lengths: [positionedStates.dim(1)],
            layer: preparedState.model.encoder.layers[0]
        )
        let (encoderHiddenStates, encodedLengths) = preparedState.model.encode(
            inputFeatures: inputFeatures,
            lengths: lengths
        )
        eval(encoderHiddenStates)

        let promptIDs = MLXArray(bootstrap.promptTokenIDs, [1, bootstrap.promptTokenIDs.count]).asType(.int32)
        let decoderHiddenStates = decoderHiddenState(
            model: preparedState.model,
            inputIDs: promptIDs,
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )
        eval(decoderHiddenStates)
        let rawLogits = preparedState.model.lm_head(decoderHiddenStates)
        eval(rawLogits)
        let logits =
            if preparedState.model.usesLogSoftmax {
                logSoftmax(rawLogits, axis: -1)
            } else {
                rawLogits
            }
        eval(logits)

        let lastLogits = logits[0, logits.dim(1) - 1]
        let topCandidates = topCandidates(
            from: lastLogits,
            tokenizer: bootstrap.tokenizer,
            count: 10
        )

        return FirstStepDiagnostic(
            promptTokenIDs: bootstrap.promptTokenIDs,
            featureShape: inputFeatures.shape,
            featureSummary: tensorSummary(inputFeatures),
            subsampledSummary: tensorSummary(subsampledStates),
            firstLayer: firstLayer,
            encoderSummary: tensorSummary(encoderHiddenStates),
            decoderHiddenSummary: tensorSummary(decoderHiddenStates),
            rawLogitsSummary: tensorSummary(rawLogits[0, rawLogits.dim(1) - 1]),
            logitsSummary: tensorSummary(lastLogits),
            topCandidates: topCandidates
        )
    }

    private func decoderHiddenState(
        model: CohereNativeConditionalGenerationModel,
        inputIDs: MLXArray,
        encoderHiddenStates: MLXArray,
        encodedLengths: [Int]
    ) -> MLXArray {
        let decoderComputeDType: DType = .float32
        let sequenceLength = inputIDs.dim(1)
        let positions = MLXArray(Array(0 ..< sequenceLength))
            .asType(.int32)
            .expandedDimensions(axis: 0)
        let selfAttentionMask = makeSelfAttentionMask(
            batchSize: inputIDs.dim(0),
            targetLength: sequenceLength,
            dtype: decoderComputeDType
        )
        let crossAttentionMask = makeCrossAttentionMask(
            lengths: encodedLengths,
            sourceLength: encoderHiddenStates.dim(1),
            dtype: decoderComputeDType
        )
        return model.decoder(
            inputIDs,
            positions: positions,
            encoderHiddenStates: encoderHiddenStates.asType(decoderComputeDType),
            selfAttentionMask: selfAttentionMask,
            crossAttentionMask: crossAttentionMask
        )
    }

    private func makeSelfAttentionMask(batchSize: Int, targetLength: Int, dtype: DType) -> MLXArray {
        let causalInvalid = MLXArray(Float(1), dtype: dtype) - tri(targetLength, m: targetLength, dtype: dtype)
        let mask = causalInvalid
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
            * -1_000_000_000

        if batchSize == 1 {
            return mask
        }

        return repeated(mask, count: batchSize, axis: 0)
    }

    private func makeCrossAttentionMask(lengths: [Int], sourceLength: Int, dtype: DType) -> MLXArray {
        var values = Array(repeating: Float.zero, count: lengths.count * sourceLength)
        for (batchIndex, length) in lengths.enumerated() where length < sourceLength {
            let rowStart = batchIndex * sourceLength
            for sourceIndex in length ..< sourceLength {
                values[rowStart + sourceIndex] = -1_000_000_000
            }
        }

        return MLXArray(values, [lengths.count, 1, 1, sourceLength]).asType(dtype)
    }

    private func firstLayerDiagnostic(
        hiddenStates: MLXArray,
        positionalEmbeddings: MLXArray,
        lengths: [Int],
        layer: CohereNativeConformerLayer
    ) -> FirstLayerDiagnostic {
        let maxAudioLength = hiddenStates.dim(1)
        let padMask = CohereNativeConformerEncoder.makePadMaskForTesting(
            lengths: lengths,
            maxAudioLength: maxAudioLength
        )
        let attentionMask = maximum(
            padMask.expandedDimensions(axis: 1),
            padMask.expandedDimensions(axis: 2)
        )

        let ff1Norm = layer.norm_feed_forward1(hiddenStates)
        let ff1Output = layer.feed_forward1(ff1Norm)
        let ff1Residual = hiddenStates + (ff1Output * 0.5)

        let attentionNorm = layer.norm_self_att(ff1Residual)
        let attentionOutput = layer.self_attn(attentionNorm, posEmb: positionalEmbeddings, mask: attentionMask)
        let attentionScores = attentionScoreSummary(
            attentionInput: attentionNorm,
            positionalEmbeddings: positionalEmbeddings,
            mask: attentionMask,
            attention: layer.self_attn
        )
        let attentionResidual = ff1Residual + attentionOutput

        let convolutionNorm = layer.norm_conv(attentionResidual)
        let convolutionOutput = layer.conv(convolutionNorm, padMask: padMask)

        eval(ff1Norm, ff1Output, ff1Residual, attentionNorm, attentionOutput, convolutionNorm, convolutionOutput)

        return FirstLayerDiagnostic(
            inputSummary: tensorSummary(hiddenStates),
            feedForwardNormSummary: tensorSummary(ff1Norm),
            feedForwardOutputSummary: tensorSummary(ff1Output),
            feedForwardResidualSummary: tensorSummary(ff1Residual),
            attentionNormSummary: tensorSummary(attentionNorm),
            attentionOutputSummary: tensorSummary(attentionOutput),
            attentionScoreSummary: attentionScores,
            convolutionNormSummary: tensorSummary(convolutionNorm),
            convolutionOutputSummary: tensorSummary(convolutionOutput)
        )
    }

    private func attentionScoreSummary(
        attentionInput: MLXArray,
        positionalEmbeddings: MLXArray,
        mask: MLXArray,
        attention: CohereNativeRelPositionMultiHeadAttention
    ) -> TensorSummary {
        let attentionDType: DType = .float32
        let batch = attentionInput.dim(0)
        let sequenceLength = attentionInput.dim(1)
        let modelDimensions = attention.headCount * attention.headDimension

        let qkv = attention.qkv_proj(attentionInput)
        let pieces = qkv.split(indices: [modelDimensions, modelDimensions * 2], axis: -1)
        let q = pieces[0]
            .asType(attentionDType)
            .reshaped([batch, sequenceLength, attention.headCount, attention.headDimension])
            .transposed(0, 2, 1, 3)
        let k = pieces[1]
            .asType(attentionDType)
            .reshaped([batch, sequenceLength, attention.headCount, attention.headDimension])
            .transposed(0, 2, 1, 3)
        let repeatedPositionalEmbeddings =
            if positionalEmbeddings.dim(0) == 1, batch > 1 {
                repeated(positionalEmbeddings, count: batch, axis: 0)
            } else {
                positionalEmbeddings
            }
        let p = attention.pos_proj(repeatedPositionalEmbeddings)
            .asType(attentionDType)
            .reshaped([batch, repeatedPositionalEmbeddings.dim(1), attention.headCount, attention.headDimension])
            .transposed(0, 2, 1, 3)
        let biasU = attention.pos_bias_u.asType(attentionDType).expandedDimensions(axes: [0, 2])
        let biasV = attention.pos_bias_v.asType(attentionDType).expandedDimensions(axes: [0, 2])
        let matrixAC = matmul(q + biasU, k.transposed(0, 1, 3, 2))
        var matrixBD = matmul(q + biasV, p.transposed(0, 1, 3, 2))
        matrixBD = CohereNativeRelPositionMultiHeadAttention.relShift(matrixBD)
        matrixBD = matrixBD[..<matrixAC.dim(3), axis: 3]
        let scores = (matrixAC + matrixBD) * attention.scale
        let maskedScores = scores + mask.asType(attentionDType).expandedDimensions(axis: 1) * -1_000_000_000

        eval(matrixAC, matrixBD, scores, maskedScores)
        return tensorSummary(maskedScores)
    }

    private func topCandidates(
        from logits: MLXArray,
        tokenizer: CohereNativeTokenizer,
        count: Int
    ) -> [FirstStepCandidate] {
        var candidates: [FirstStepCandidate] = []
        candidates.reserveCapacity(count)

        for tokenID in 0 ..< logits.dim(0) {
            let score = logits[tokenID].item(Float.self)
            let tokenText = tokenizer.decode([tokenID])
            let candidate = FirstStepCandidate(
                tokenID: tokenID,
                tokenText: tokenText,
                score: score
            )

            if candidates.count < count {
                candidates.append(candidate)
                candidates.sort(by: { $0.score > $1.score })
                continue
            }

            guard let lastCandidate = candidates.last, score > lastCandidate.score else {
                continue
            }

            candidates.removeLast()
            candidates.append(candidate)
            candidates.sort(by: { $0.score > $1.score })
        }

        return candidates
    }

    private func tensorSummary(_ array: MLXArray) -> TensorSummary {
        let flat = array.asType(.float32).reshaped([-1])
        var finiteCount = 0
        var nonFiniteCount = 0
        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude

        for index in 0 ..< flat.dim(0) {
            let value = flat[index].item(Float.self)
            guard value.isFinite else {
                nonFiniteCount += 1
                continue
            }

            finiteCount += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        }

        return TensorSummary(
            finiteCount: finiteCount,
            nonFiniteCount: nonFiniteCount,
            minimum: finiteCount > 0 ? minimum : nil,
            maximum: finiteCount > 0 ? maximum : nil
        )
    }

    private func canonicalMetrics(reference: String, hypothesis: String) -> CanonicalMetrics {
        let referenceWords = canonicalWords(from: reference)
        let hypothesisWords = canonicalWords(from: hypothesis)
        guard !referenceWords.isEmpty else {
            return CanonicalMetrics(wordErrorRate: hypothesisWords.isEmpty ? 0 : 1)
        }

        let distance = levenshteinDistance(referenceWords, hypothesisWords)
        return CanonicalMetrics(wordErrorRate: Double(distance) / Double(referenceWords.count))
    }

    private func canonicalWords(from text: String) -> [String] {
        text
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9 ]", with: " ", options: .regularExpression)
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
    }

    private func levenshteinDistance(_ lhs: [String], _ rhs: [String]) -> Int {
        if lhs.isEmpty { return rhs.count }
        if rhs.isEmpty { return lhs.count }

        var previous = Array(0 ... rhs.count)
        for (lhsIndex, lhsWord) in lhs.enumerated() {
            var current = [lhsIndex + 1]
            current.reserveCapacity(rhs.count + 1)

            for (rhsIndex, rhsWord) in rhs.enumerated() {
                let insertion = current[rhsIndex] + 1
                let deletion = previous[rhsIndex + 1] + 1
                let substitution = previous[rhsIndex] + (lhsWord == rhsWord ? 0 : 1)
                current.append(min(insertion, deletion, substitution))
            }

            previous = current
        }

        return previous[rhs.count]
    }
}

private struct BenchmarkManifest: Decodable {
    let items: [BenchmarkManifestItem]
}

private struct BenchmarkManifestItem: Decodable {
    let audioPath: String
    let durationSeconds: TimeInterval
    let referenceText: String
}

private struct BenchmarkSample {
    let referenceText: String
    let audioURL: URL
}

private struct FirstStepDiagnostic {
    let promptTokenIDs: [Int]
    let featureShape: [Int]
    let featureSummary: TensorSummary
    let subsampledSummary: TensorSummary
    let firstLayer: FirstLayerDiagnostic
    let encoderSummary: TensorSummary
    let decoderHiddenSummary: TensorSummary
    let rawLogitsSummary: TensorSummary
    let logitsSummary: TensorSummary
    let topCandidates: [FirstStepCandidate]
}

private struct FirstLayerDiagnostic {
    let inputSummary: TensorSummary
    let feedForwardNormSummary: TensorSummary
    let feedForwardOutputSummary: TensorSummary
    let feedForwardResidualSummary: TensorSummary
    let attentionNormSummary: TensorSummary
    let attentionOutputSummary: TensorSummary
    let attentionScoreSummary: TensorSummary
    let convolutionNormSummary: TensorSummary
    let convolutionOutputSummary: TensorSummary
}

private struct FirstStepCandidate {
    let tokenID: Int
    let tokenText: String
    let score: Float
}

private struct TensorSummary: CustomStringConvertible {
    let finiteCount: Int
    let nonFiniteCount: Int
    let minimum: Float?
    let maximum: Float?

    var description: String {
        let minimumDescription = minimum.map { String($0) } ?? "nil"
        let maximumDescription = maximum.map { String($0) } ?? "nil"
        return "finite=\(finiteCount) nonFinite=\(nonFiniteCount) min=\(minimumDescription) max=\(maximumDescription)"
    }
}

private struct CanonicalMetrics {
    let wordErrorRate: Double
}

private enum CohereNativeSmokeError: LocalizedError {
    case missingBenchmarkSample

    var errorDescription: String? {
        switch self {
        case .missingBenchmarkSample:
            return "No short benchmark sample was available for the Cohere native smoke test."
        }
    }
}
