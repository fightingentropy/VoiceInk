import Foundation
import MLX
import Testing
@testable import VoiceInk

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
    func nativeEncoderPreprocessTransposesConvolutionWeights() {
        let conv1D = MLXArray(0 ..< 24, [4, 2, 3]).asType(.float32)
        let conv2D = MLXArray(0 ..< 108, [4, 3, 3, 3]).asType(.float32)

        let processed = CohereNativeEncoderLoader.preprocessCheckpointWeights([
            "encoder.layers.0.conv.depthwise_conv.weight": conv1D,
            "encoder.subsampling.conv.0.weight": conv2D,
        ])

        let processedConv1D = processed["encoder.layers.0.conv.depthwise_conv.weight"]!
        let processedConv2D = processed["encoder.subsampling.conv.0.weight"]!

        #expect(processedConv1D.shape == [4, 3, 2])
        #expect(processedConv1D[0, 0, 0].item(Float.self) == 0)
        #expect(processedConv1D[0, 2, 1].item(Float.self) == 5)

        #expect(processedConv2D.shape == [4, 3, 3, 3])
        #expect(processedConv2D[0, 0, 0, 0].item(Float.self) == 0)
        #expect(processedConv2D[0, 2, 2, 2].item(Float.self) == 26)
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
}
