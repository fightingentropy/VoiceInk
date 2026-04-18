import Foundation
import MLX
import Testing
@testable import VoiceInk

@Suite(.serialized)
struct CohereNativeFoundationTests {
    @Test
    func promptBuilderMatchesUpstreamFormat() {
        let prompt = CohereNativePromptBuilder.buildPrompt(language: "en", punctuation: true)
        #expect(
            prompt == "<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"
        )
    }

    @Test
    func nativeConfigDecodesAudioShapeInputs() throws {
        let configData = Data(
            """
            {
              "vocab_size": 16384,
              "encoder": {
                "d_model": 1280,
                "ff_expansion_factor": 4,
                "n_heads": 8,
                "n_layers": 48,
                "conv_kernel_size": 9,
                "dropout": 0,
                "subsampling_conv_channels": 256,
                "subsampling_factor": 8,
                "feat_in": 128,
                "feat_out": -1,
                "pos_emb_max_len": 5000
              },
              "transf_decoder": {
                "config_dict": {
                  "hidden_size": 1024,
                  "inner_size": 4096,
                  "num_attention_heads": 8,
                  "num_layers": 8,
                  "hidden_act": "relu",
                  "max_sequence_length": 1024
                }
              },
              "head": {
                "hidden_size": 1024,
                "num_classes": 16384,
                "log_softmax": true
              },
              "preprocessor": {
                "dither": 1e-5,
                "features": 128,
                "n_fft": 512,
                "normalize": "per_feature",
                "sample_rate": 16000,
                "window": "hann",
                "window_size": 0.025,
                "window_stride": 0.01
              },
              "max_audio_clip_s": 35,
              "overlap_chunk_second": 5,
              "supported_languages": ["en", "fr", "de"]
            }
            """.utf8
        )

        let config = try JSONDecoder().decode(CohereNativeModelConfig.self, from: configData)

        #expect(config.encoder.dModel == 1280)
        #expect(config.decoder.config.hiddenSize == 1024)
        #expect(config.audioConfiguration.windowSize == 400)
        #expect(config.audioConfiguration.hopLength == 160)
        #expect(config.audioConfiguration.melBinCount == 128)
        #expect(config.supportedLanguages == ["en", "fr", "de"])
    }

    @Test
    func nativeFeatureExtractorProducesExpectedMelShape() {
        let configuration = CohereNativeAudioConfiguration(
            sampleRate: 16_000,
            fftSize: 512,
            windowSize: 400,
            hopLength: 160,
            melBinCount: 128,
            normalizePerFeature: true,
            dither: 0,
            maxClipDuration: 35,
            overlapDuration: 5
        )
        let extractor = CohereNativeFeatureExtractor(configuration: configuration)
        let silence = Array(repeating: Float.zero, count: configuration.sampleRate)

        let features = extractor.extractLogMelFeatures(from: silence)

        #expect(features.shape == [128, 101])
    }

    @Test
    func nativeEncoderPreprocessHandlesConvolutionWeightLayouts() {
        let pytorchConv1D = MLXArray(0 ..< 12, [4, 1, 3]).asType(.float32)
        let mlxConv1D = MLXArray(0 ..< 12, [4, 3, 1]).asType(.float32)
        let pytorchConv2D = MLXArray(0 ..< 60, [4, 1, 3, 5]).asType(.float32)
        let mlxConv2D = MLXArray(0 ..< 60, [4, 3, 5, 1]).asType(.float32)

        let processed = CohereNativeEncoderLoader.preprocessCheckpointWeights([
            "encoder.layers.0.conv.depthwise_conv.weight": pytorchConv1D,
            "encoder.layers.1.conv.depthwise_conv.weight": mlxConv1D,
            "encoder.subsampling.conv.0.weight": pytorchConv2D,
            "encoder.subsampling.conv0.weight": mlxConv2D,
        ])

        let processedPyTorchConv1D = processed["encoder.layers.0.conv.depthwise_conv.weight"]!
        let processedMLXConv1D = processed["encoder.layers.1.conv.depthwise_conv.weight"]!
        let processedPyTorchConv2D = processed["encoder.subsampling.conv.0.weight"]!
        let processedMLXConv2D = processed["encoder.subsampling.conv0.weight"]!

        #expect(processedPyTorchConv1D.shape == [4, 3, 1])
        #expect(processedPyTorchConv1D[0, 0, 0].item(Float.self) == 0)
        #expect(processedPyTorchConv1D[0, 2, 0].item(Float.self) == 2)

        #expect(processedMLXConv1D.shape == [4, 3, 1])
        #expect(processedMLXConv1D[0, 2, 0].item(Float.self) == 2)

        #expect(processedPyTorchConv2D.shape == [4, 3, 5, 1])
        #expect(processedPyTorchConv2D[0, 0, 0, 0].item(Float.self) == 0)
        #expect(processedPyTorchConv2D[0, 2, 4, 0].item(Float.self) == 14)

        #expect(processedMLXConv2D.shape == [4, 3, 5, 1])
        #expect(processedMLXConv2D[0, 2, 4, 0].item(Float.self) == 14)
    }

    @Test
    func nativeEncoderRelativeShiftMatchesUpstreamLayout() {
        let input = MLXArray([1, 2, 3, 4, 5, 6], [1, 1, 2, 3]).asType(.float32)

        let shifted = CohereNativeRelPositionMultiHeadAttention.relShift(input)

        #expect(shifted.shape == [1, 1, 2, 3])
        #expect(shifted[0, 0, 0, 0].item(Float.self) == 2)
        #expect(shifted[0, 0, 0, 1].item(Float.self) == 3)
        #expect(shifted[0, 0, 0, 2].item(Float.self) == 0)
        #expect(shifted[0, 0, 1, 0].item(Float.self) == 4)
        #expect(shifted[0, 0, 1, 1].item(Float.self) == 5)
        #expect(shifted[0, 0, 1, 2].item(Float.self) == 6)
    }

    @Test
    func nativeDecoderStepMatchesFullSequenceDecode() {
        let configData = Data(
            """
            {
              "vocab_size": 32,
              "encoder": {
                "d_model": 8,
                "ff_expansion_factor": 2,
                "n_heads": 2,
                "n_layers": 2,
                "conv_kernel_size": 5,
                "dropout": 0,
                "subsampling_conv_channels": 8,
                "subsampling_factor": 8,
                "feat_in": 128,
                "feat_out": -1,
                "pos_emb_max_len": 128
              },
              "transf_decoder": {
                "config_dict": {
                  "hidden_size": 16,
                  "inner_size": 32,
                  "num_attention_heads": 4,
                  "num_layers": 2,
                  "hidden_act": "relu",
                  "max_sequence_length": 64
                }
              },
              "head": {
                "hidden_size": 16,
                "num_classes": 32,
                "log_softmax": true
              },
              "preprocessor": {
                "dither": 0,
                "features": 128,
                "n_fft": 512,
                "normalize": "per_feature",
                "sample_rate": 16000,
                "window": "hann",
                "window_size": 0.025,
                "window_stride": 0.01
              },
              "max_audio_clip_s": 35,
              "overlap_chunk_second": 5,
              "supported_languages": ["en"]
            }
            """.utf8
        )

        let config = try! JSONDecoder().decode(CohereNativeModelConfig.self, from: configData)
        let model = CohereNativeConditionalGenerationModel(config: config)
        model.train(false)
        eval(model)

        let promptTokenIDs = [3, 5, 7, 9]
        let inputIDs = MLXArray(promptTokenIDs, [1, promptTokenIDs.count]).asType(.int32)
        let encoderHiddenStates = (MLXArray(0 ..< 96, [1, 6, 16]).asType(.float32) * 0.01)
        let encodedLengths = [5]

        let fullLogits = model.decode(
            inputIDs: inputIDs,
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )

        let decoderContext = model.prepareDecoderContext(
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )
        let decoderCache = model.makeDecoderCache()
        let prefillCache = model.makeDecoderCache()
        var stepLogits: MLXArray?
        let prefillLogits = model.prefill(
            inputIDs: inputIDs,
            positions: MLXArray(Array(0 ..< promptTokenIDs.count), [1, promptTokenIDs.count]).asType(.int32),
            decoderContext: decoderContext,
            cache: prefillCache
        )
        let greedyDecoderCache = model.makeDecoderCache()
        var greedyStepLogits: MLXArray?

        for (position, tokenID) in promptTokenIDs.enumerated() {
            stepLogits = model.decodeStep(
                inputIDs: MLXArray([tokenID], [1, 1]).asType(.int32),
                positions: MLXArray([position], [1, 1]).asType(.int32),
                decoderContext: decoderContext,
                cache: decoderCache
            )
            greedyStepLogits = model.decodeStep(
                inputIDs: MLXArray([tokenID], [1, 1]).asType(.int32),
                positions: MLXArray([position], [1, 1]).asType(.int32),
                decoderContext: decoderContext,
                cache: greedyDecoderCache,
                applyLogSoftmax: false
            )
        }

        guard let stepLogits, let greedyStepLogits else {
            Issue.record("Cached decoder did not produce logits.")
            return
        }

        let fullLastLogits = fullLogits[0, fullLogits.dim(1) - 1].asType(.float32).reshaped([-1])
        let stepLastLogits = stepLogits[0, 0].asType(.float32).reshaped([-1])
        let prefillLastLogits = prefillLogits[0, prefillLogits.dim(1) - 1].asType(.float32).reshaped([-1])
        let greedyLastLogits = greedyStepLogits[0, 0].asType(.float32).reshaped([-1])
        eval(
            fullLastLogits,
            stepLastLogits,
            prefillLastLogits,
            greedyLastLogits,
            decoderCache.arraysForEval(),
            prefillCache.arraysForEval(),
            greedyDecoderCache.arraysForEval()
        )

        var maxDifference: Float = 0
        for index in 0 ..< fullLastLogits.dim(0) {
            let difference = abs(fullLastLogits[index].item(Float.self) - stepLastLogits[index].item(Float.self))
            maxDifference = max(maxDifference, difference)
        }

        var maxPrefillDifference: Float = 0
        for index in 0 ..< fullLastLogits.dim(0) {
            let difference = abs(fullLastLogits[index].item(Float.self) - prefillLastLogits[index].item(Float.self))
            maxPrefillDifference = max(maxPrefillDifference, difference)
        }

        let continuedTokenID = Int32(11)
        let continuedFullLogits = model.decode(
            inputIDs: MLXArray(promptTokenIDs + [Int(continuedTokenID)], [1, promptTokenIDs.count + 1]).asType(.int32),
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )
        let continuedPrefillLogits = model.decodeStep(
            inputIDs: MLXArray([continuedTokenID], [1, 1]).asType(.int32),
            positions: MLXArray([promptTokenIDs.count], [1, 1]).asType(.int32),
            decoderContext: decoderContext,
            cache: prefillCache
        )
        let continuedFullLastLogits = continuedFullLogits[0, continuedFullLogits.dim(1) - 1].asType(.float32).reshaped([-1])
        let continuedPrefillLastLogits = continuedPrefillLogits[0, 0].asType(.float32).reshaped([-1])
        eval(continuedFullLastLogits, continuedPrefillLastLogits, prefillCache.arraysForEval())

        var maxContinuationDifference: Float = 0
        for index in 0 ..< continuedFullLastLogits.dim(0) {
            let difference = abs(continuedFullLastLogits[index].item(Float.self) - continuedPrefillLastLogits[index].item(Float.self))
            maxContinuationDifference = max(maxContinuationDifference, difference)
        }

        #expect(maxDifference < 1e-4)
        #expect(maxPrefillDifference < 1e-4)
        #expect(maxContinuationDifference < 1e-4)
        #expect(fullLastLogits.argMax(axis: -1).item(Int.self) == greedyLastLogits.argMax(axis: -1).item(Int.self))
    }

    @Test
    func nativeRuntimeRecommendedTokenBudgetUsesAudioDuration() {
        let shortClipBudget = CohereNativeRuntime.recommendedMaxNewTokens(
            audioSampleCount: 2 * 16_000,
            sampleRate: 16_000,
            promptTokenCount: 10,
            maxSequenceLength: 1_024
        )
        let mediumClipBudget = CohereNativeRuntime.recommendedMaxNewTokens(
            audioSampleCount: 20 * 16_000,
            sampleRate: 16_000,
            promptTokenCount: 10,
            maxSequenceLength: 1_024
        )
        let longClipBudget = CohereNativeRuntime.recommendedMaxNewTokens(
            audioSampleCount: 40 * 16_000,
            sampleRate: 16_000,
            promptTokenCount: 10,
            maxSequenceLength: 1_024
        )

        #expect(shortClipBudget == 48)
        #expect(mediumClipBudget == 116)
        #expect(longClipBudget == 192)
    }

    @Test
    func nativeRuntimeRecommendedTokenBudgetRespectsSequenceLimit() {
        let budget = CohereNativeRuntime.recommendedMaxNewTokens(
            audioSampleCount: 40 * 16_000,
            sampleRate: 16_000,
            promptTokenCount: 980,
            maxSequenceLength: 1_024
        )

        #expect(budget == 44)
    }

    @Test
    func decoderKVCacheResetPreservesSelfAttentionStorage() {
        let cache = CohereNativeDecoderKVCache(maxSequenceLength: 8)
        let key = MLXArray.zeros([1, 2, 3, 4], dtype: .float32)
        let value = MLXArray.zeros([1, 2, 3, 4], dtype: .float32)

        _ = cache.prepare(key: key, value: value)

        #expect(cache.hasAllocatedStorage())
        #expect(cache.currentVisibleLength() == 3)
        #expect(cache.currentStorageShape() == [1, 2, 8, 4])

        cache.reset(keepStorage: true)

        #expect(cache.hasAllocatedStorage())
        #expect(cache.currentVisibleLength() == 0)
        #expect(cache.currentStorageShape() == [1, 2, 8, 4])

        _ = cache.prepare(key: key, value: value)
        #expect(cache.currentVisibleLength() == 3)

        cache.reset(keepStorage: false)
        #expect(!cache.hasAllocatedStorage())
        #expect(cache.currentVisibleLength() == 0)
    }
}
