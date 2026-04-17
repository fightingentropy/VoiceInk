import Foundation

struct CohereNativeAudioConfiguration: Sendable, Equatable {
    let sampleRate: Int
    let fftSize: Int
    let windowSize: Int
    let hopLength: Int
    let melBinCount: Int
    let normalizePerFeature: Bool
    let dither: Float
    let maxClipDuration: Double
    let overlapDuration: Double
}

struct CohereNativeModelConfig: Decodable, Sendable {
    struct Quantization: Decodable, Sendable {
        let bits: Int
        let groupSize: Int

        private enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    struct EncoderConfig: Decodable, Sendable {
        let dModel: Int
        let ffExpansionFactor: Int
        let headCount: Int
        let layerCount: Int
        let convKernelSize: Int
        let dropout: Float
        let subsamplingConvChannels: Int
        let subsamplingFactor: Int
        let featureCount: Int
        let projectedFeatureCount: Int
        let positionalEmbeddingMaxLength: Int

        private enum CodingKeys: String, CodingKey {
            case dModel = "d_model"
            case ffExpansionFactor = "ff_expansion_factor"
            case headCount = "n_heads"
            case layerCount = "n_layers"
            case convKernelSize = "conv_kernel_size"
            case dropout
            case subsamplingConvChannels = "subsampling_conv_channels"
            case subsamplingFactor = "subsampling_factor"
            case featureCount = "feat_in"
            case projectedFeatureCount = "feat_out"
            case positionalEmbeddingMaxLength = "pos_emb_max_len"
        }
    }

    struct DecoderConfig: Decodable, Sendable {
        struct CoreConfig: Decodable, Sendable {
            let hiddenSize: Int
            let innerSize: Int
            let headCount: Int
            let layerCount: Int
            let hiddenActivation: String
            let maxSequenceLength: Int

            private enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case innerSize = "inner_size"
                case headCount = "num_attention_heads"
                case layerCount = "num_layers"
                case hiddenActivation = "hidden_act"
                case maxSequenceLength = "max_sequence_length"
            }
        }

        let config: CoreConfig

        private enum CodingKeys: String, CodingKey {
            case config = "config_dict"
        }
    }

    struct HeadConfig: Decodable, Sendable {
        let hiddenSize: Int
        let classCount: Int
        let logSoftmax: Bool

        private enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case classCount = "num_classes"
            case logSoftmax = "log_softmax"
        }
    }

    struct PreprocessorConfig: Decodable, Sendable {
        let dither: Float
        let featureCount: Int
        let fftSize: Int
        let normalize: String
        let sampleRate: Int
        let window: String
        let windowSizeSeconds: Double
        let windowStrideSeconds: Double

        private enum CodingKeys: String, CodingKey {
            case dither
            case featureCount = "features"
            case fftSize = "n_fft"
            case normalize
            case sampleRate = "sample_rate"
            case window
            case windowSizeSeconds = "window_size"
            case windowStrideSeconds = "window_stride"
        }
    }

    let vocabularySize: Int
    let encoder: EncoderConfig
    let decoder: DecoderConfig
    let head: HeadConfig
    let preprocessor: PreprocessorConfig
    let maxAudioClipDuration: Double
    let overlapChunkDuration: Double
    let supportedLanguages: [String]
    // Present when the MLX checkpoint was shipped pre-quantized
    // (e.g. the 4-bit or 6-bit builds). nil for the fp16 build.
    let quantization: Quantization?

    private enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case encoder
        case decoder = "transf_decoder"
        case head
        case preprocessor
        case maxAudioClipDuration = "max_audio_clip_s"
        case overlapChunkDuration = "overlap_chunk_second"
        case supportedLanguages = "supported_languages"
        case quantization
    }

    var audioConfiguration: CohereNativeAudioConfiguration {
        CohereNativeAudioConfiguration(
            sampleRate: preprocessor.sampleRate,
            fftSize: preprocessor.fftSize,
            windowSize: Int(round(preprocessor.windowSizeSeconds * Double(preprocessor.sampleRate))),
            hopLength: Int(round(preprocessor.windowStrideSeconds * Double(preprocessor.sampleRate))),
            melBinCount: preprocessor.featureCount,
            normalizePerFeature: preprocessor.normalize == "per_feature",
            dither: preprocessor.dither,
            maxClipDuration: maxAudioClipDuration,
            overlapDuration: overlapChunkDuration
        )
    }
}
