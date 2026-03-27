import Foundation
import MLX
import MLXNN

struct CohereNativeEncoderWarmupSummary: Sendable {
    let outputShape: [Int]
    let encodedLengths: [Int]
    let decoderLogitsShape: [Int]
    let parameterCount: Int
}

enum CohereNativeEncoderLoaderError: LocalizedError {
    case missingWeights(URL)

    var errorDescription: String? {
        switch self {
        case .missingWeights(let directory):
            return "No Cohere native MLX weight file was found in \(directory.path)."
        }
    }
}

final class CohereNativePreparedState: @unchecked Sendable {
    let bootstrap: CohereNativeBootstrap
    let model: CohereNativeConditionalGenerationModel
    let summary: CohereNativeEncoderWarmupSummary

    init(
        bootstrap: CohereNativeBootstrap,
        model: CohereNativeConditionalGenerationModel,
        summary: CohereNativeEncoderWarmupSummary
    ) {
        self.bootstrap = bootstrap
        self.model = model
        self.summary = summary
    }
}

final class CohereNativeConformerFeedForward: Module {
    let linear1: Linear
    let linear2: Linear

    init(dModel: Int, hiddenDimensions: Int) {
        self.linear1 = Linear(dModel, hiddenDimensions)
        self.linear2 = Linear(hiddenDimensions, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

final class CohereNativeConformerConvolution: Module {
    let pointwise_conv1: Conv1d
    let depthwise_conv: Conv1d
    let batch_norm: BatchNorm
    let pointwise_conv2: Conv1d

    init(dModel: Int, kernelSize: Int) {
        self.pointwise_conv1 = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel * 2,
            kernelSize: 1
        )
        self.depthwise_conv = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: kernelSize,
            padding: (kernelSize - 1) / 2,
            groups: dModel
        )
        self.batch_norm = BatchNorm(featureCount: dModel)
        self.pointwise_conv2 = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: 1
        )
    }

    func callAsFunction(_ x: MLXArray, padMask: MLXArray?) -> MLXArray {
        var x = glu(pointwise_conv1(x), axis: -1)

        if let padMask {
            let validMask = MLXArray(Float(1), dtype: x.dtype) - padMask.asType(x.dtype).expandedDimensions(axis: 2)
            x = x * validMask
        }

        x = depthwise_conv(x)
        x = batch_norm(x)
        x = silu(x)
        return pointwise_conv2(x)
    }
}

final class CohereNativeRelPositionMultiHeadAttention: Module {
    let qkv_proj: Linear
    let pos_proj: Linear
    let out_proj: Linear
    let pos_bias_u: MLXArray
    let pos_bias_v: MLXArray

    let headCount: Int
    let headDimension: Int
    let scale: Float
    private let attentionDType: DType = .float32

    init(headCount: Int, modelDimensions: Int) {
        let headDimension = modelDimensions / headCount
        self.qkv_proj = Linear(modelDimensions, modelDimensions * 3)
        self.pos_proj = Linear(modelDimensions, modelDimensions, bias: false)
        self.out_proj = Linear(modelDimensions, modelDimensions)
        self.pos_bias_u = MLXArray.zeros([headCount, headDimension], dtype: .float32)
        self.pos_bias_v = MLXArray.zeros([headCount, headDimension], dtype: .float32)
        self.headCount = headCount
        self.headDimension = headDimension
        self.scale = Foundation.pow(Float(headDimension), -0.5)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let modelDimensions = headCount * headDimension
        let outputDType = x.dtype

        let qkv = qkv_proj(x)
        let pieces = qkv.split(indices: [modelDimensions, modelDimensions * 2], axis: -1)
        let q = pieces[0]
            .asType(attentionDType)
            .reshaped([batch, sequenceLength, headCount, headDimension])
            .transposed(0, 2, 1, 3)
        let k = pieces[1]
            .asType(attentionDType)
            .reshaped([batch, sequenceLength, headCount, headDimension])
            .transposed(0, 2, 1, 3)
        let v = pieces[2]
            .asType(attentionDType)
            .reshaped([batch, sequenceLength, headCount, headDimension])
            .transposed(0, 2, 1, 3)

        let repeatedPosEmb =
            if posEmb.dim(0) == 1, batch > 1 {
                repeated(posEmb, count: batch, axis: 0)
            } else {
                posEmb
            }

        let p = pos_proj(repeatedPosEmb)
            .asType(attentionDType)
            .reshaped([batch, repeatedPosEmb.dim(1), headCount, headDimension])
            .transposed(0, 2, 1, 3)

        let biasU = pos_bias_u.asType(attentionDType).expandedDimensions(axes: [0, 2])
        let biasV = pos_bias_v.asType(attentionDType).expandedDimensions(axes: [0, 2])

        let matrixAC = matmul(q + biasU, k.transposed(0, 1, 3, 2))
        var matrixBD = matmul(q + biasV, p.transposed(0, 1, 3, 2))
        matrixBD = Self.relShift(matrixBD)
        matrixBD = matrixBD[..<matrixAC.dim(3), axis: 3]

        var scores = (matrixAC + matrixBD) * scale

        let expandedMask: MLXArray?
        if let mask {
            let resolvedMask = mask.asType(attentionDType).expandedDimensions(axis: 1)
            scores = scores + resolvedMask * -1_000_000_000
            expandedMask = resolvedMask
        } else {
            expandedMask = nil
        }

        var attention = softmax(scores, axis: -1, precise: true)
        if let expandedMask {
            attention = attention * (MLXArray(Float(1), dtype: attention.dtype) - expandedMask)
        }

        let attended = matmul(attention, v)
        return out_proj(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped([batch, sequenceLength, modelDimensions])
                .asType(outputDType)
        )
    }

    static func relShift(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let heads = x.dim(1)
        let queryLength = x.dim(2)
        let positionalLength = x.dim(3)

        let paddedInput = padded(
            x,
            widths: [0, 0, 0, [1, 0]],
            value: MLXArray(0, dtype: x.dtype)
        )
        let reshaped = paddedInput.reshaped([batch, heads, positionalLength + 1, queryLength])
        let shifted = reshaped[1..., axis: 2]
        return shifted.reshaped([batch, heads, queryLength, positionalLength])
    }
}

final class CohereNativeConformerLayer: Module {
    let norm_feed_forward1: LayerNorm
    let feed_forward1: CohereNativeConformerFeedForward
    let norm_self_att: LayerNorm
    let self_attn: CohereNativeRelPositionMultiHeadAttention
    let norm_conv: LayerNorm
    let conv: CohereNativeConformerConvolution
    let norm_feed_forward2: LayerNorm
    let feed_forward2: CohereNativeConformerFeedForward
    let norm_out: LayerNorm

    init(dModel: Int, hiddenDimensions: Int, headCount: Int, convKernelSize: Int) {
        self.norm_feed_forward1 = LayerNorm(dimensions: dModel)
        self.feed_forward1 = CohereNativeConformerFeedForward(dModel: dModel, hiddenDimensions: hiddenDimensions)
        self.norm_self_att = LayerNorm(dimensions: dModel)
        self.self_attn = CohereNativeRelPositionMultiHeadAttention(
            headCount: headCount,
            modelDimensions: dModel
        )
        self.norm_conv = LayerNorm(dimensions: dModel)
        self.conv = CohereNativeConformerConvolution(dModel: dModel, kernelSize: convKernelSize)
        self.norm_feed_forward2 = LayerNorm(dimensions: dModel)
        self.feed_forward2 = CohereNativeConformerFeedForward(dModel: dModel, hiddenDimensions: hiddenDimensions)
        self.norm_out = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?, padMask: MLXArray?) -> MLXArray {
        var hiddenStates = x

        let feedForward1Residual = hiddenStates
        hiddenStates = feedForward1Residual + (feed_forward1(norm_feed_forward1(hiddenStates)) * 0.5)

        let attentionResidual = hiddenStates
        hiddenStates = attentionResidual + self_attn(norm_self_att(hiddenStates), posEmb: posEmb, mask: mask)

        let convolutionResidual = hiddenStates
        hiddenStates = convolutionResidual + conv(norm_conv(hiddenStates), padMask: padMask)

        let feedForward2Residual = hiddenStates
        hiddenStates = feedForward2Residual + (feed_forward2(norm_feed_forward2(hiddenStates)) * 0.5)

        return norm_out(hiddenStates)
    }
}

final class CohereNativeConvSubsampling: Module {
    let conv: [Module]
    let out: Linear

    init(config: CohereNativeModelConfig.EncoderConfig) {
        let convChannels = config.subsamplingConvChannels
        let outputDimensions = config.projectedFeatureCount > 0 ? config.projectedFeatureCount : config.dModel

        self.conv = [
            Conv2d(inputChannels: 1, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1),
            ReLU(),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                groups: convChannels
            ),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 1),
            ReLU(),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                groups: convChannels
            ),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 1),
            ReLU(),
        ]

        self.out = Linear(
            convChannels * (config.featureCount / config.subsamplingFactor),
            outputDimensions
        )
    }

    func callAsFunction(_ x: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        var hiddenStates = x.transposed(0, 2, 1).expandedDimensions(axis: 3)
        var lengths = lengths

        for layer in conv {
            let mask = Self.makeSubsamplingMask(lengths: lengths, time: hiddenStates.dim(1), dtype: hiddenStates.dtype)
            hiddenStates = hiddenStates * mask

            switch layer {
            case let convolution as Conv2d:
                hiddenStates = convolution(hiddenStates)
                let kernelSize = convolution.weight.dim(1)
                let stride = convolution.stride.0
                let padding = convolution.padding.0
                if stride > 1 {
                    lengths = lengths.map { max(1, ($0 + (2 * padding) - kernelSize) / stride + 1) }
                }
            case let activation as ReLU:
                hiddenStates = activation(hiddenStates)
            default:
                fatalError("Unsupported Cohere native subsampling layer: \(type(of: layer))")
            }
        }

        let finalMask = Self.makeSubsamplingMask(lengths: lengths, time: hiddenStates.dim(1), dtype: hiddenStates.dtype)
        hiddenStates = hiddenStates * finalMask

        let batch = hiddenStates.dim(0)
        let time = hiddenStates.dim(1)
        let features = hiddenStates.dim(2)
        let channels = hiddenStates.dim(3)

        let flattened = hiddenStates
            .swappedAxes(2, 3)
            .reshaped([batch, time, channels * features])
        return (out(flattened), lengths)
    }

    private static func makeSubsamplingMask(lengths: [Int], time: Int, dtype: DType) -> MLXArray {
        let batch = lengths.count
        var values = Array(repeating: Float(1), count: batch * time)
        for (batchIndex, length) in lengths.enumerated() where length < time {
            let rowStart = batchIndex * time
            for timeIndex in length ..< time {
                values[rowStart + timeIndex] = 0
            }
        }

        return MLXArray(values, [batch, time, 1, 1]).asType(dtype)
    }
}

final class CohereNativeRelPositionalEncoding {
    let modelDimensions: Int

    init(modelDimensions: Int) {
        self.modelDimensions = modelDimensions
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let inputLength = x.dim(1)
        let positions = Array(stride(from: inputLength - 1, through: -(inputLength - 1), by: -1)).map(Float.init)
        let divTerm = (0 ..< modelDimensions / 2).map { index -> Float in
            let exponent = Float(index * 2) / Float(modelDimensions)
            return Float(Foundation.exp(-Foundation.log(10_000) * Double(exponent)))
        }

        var values = Array(repeating: Float.zero, count: positions.count * modelDimensions)
        for (rowIndex, position) in positions.enumerated() {
            for (columnIndex, divisor) in divTerm.enumerated() {
                let angle = position * divisor
                values[(rowIndex * modelDimensions) + (columnIndex * 2)] = Float(Foundation.sin(Double(angle)))
                values[(rowIndex * modelDimensions) + (columnIndex * 2) + 1] = Float(Foundation.cos(Double(angle)))
            }
        }

        let positionalEmbeddings = MLXArray(values, [1, positions.count, modelDimensions]).asType(x.dtype)
        return (x, positionalEmbeddings)
    }
}

final class CohereNativeConformerEncoder: Module {
    let d_model: Int
    let subsampling: CohereNativeConvSubsampling
    let layers: [CohereNativeConformerLayer]

    private let positionalEncoding: CohereNativeRelPositionalEncoding

    init(config: CohereNativeModelConfig.EncoderConfig) {
        let hiddenDimensions = config.dModel * config.ffExpansionFactor
        self.d_model = config.dModel
        self.subsampling = CohereNativeConvSubsampling(config: config)
        self.positionalEncoding = CohereNativeRelPositionalEncoding(modelDimensions: config.dModel)
        self.layers = (0 ..< config.layerCount).map { _ in
            CohereNativeConformerLayer(
                dModel: config.dModel,
                hiddenDimensions: hiddenDimensions,
                headCount: config.headCount,
                convKernelSize: config.convKernelSize
            )
        }
    }

    func callAsFunction(_ inputFeatures: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        let expectedDType = subsampling.out.weight.dtype
        let inputFeatures = inputFeatures.dtype == expectedDType ? inputFeatures : inputFeatures.asType(expectedDType)

        var (hiddenStates, lengths) = subsampling(inputFeatures, lengths: lengths)
        let maxAudioLength = hiddenStates.dim(1)
        let (positionalHiddenStates, positionalEmbeddings) = positionalEncoding(hiddenStates)
        hiddenStates = positionalHiddenStates

        let padMask = Self.makePadMask(lengths: lengths, maxAudioLength: maxAudioLength)
        let attentionMask = maximum(
            padMask.expandedDimensions(axis: 1),
            padMask.expandedDimensions(axis: 2)
        )

        for layer in layers {
            hiddenStates = layer(
                hiddenStates,
                posEmb: positionalEmbeddings,
                mask: attentionMask,
                padMask: padMask
            )
        }

        return (hiddenStates, lengths)
    }

    private static func makePadMask(lengths: [Int], maxAudioLength: Int) -> MLXArray {
        var values = Array(repeating: Float.zero, count: lengths.count * maxAudioLength)
        for (batchIndex, length) in lengths.enumerated() where length < maxAudioLength {
            let rowStart = batchIndex * maxAudioLength
            for timeIndex in length ..< maxAudioLength {
                values[rowStart + timeIndex] = 1
            }
        }
        return MLXArray(values, [lengths.count, maxAudioLength])
    }

    static func makePadMaskForTesting(lengths: [Int], maxAudioLength: Int) -> MLXArray {
        makePadMask(lengths: lengths, maxAudioLength: maxAudioLength)
    }
}

final class CohereNativeFixedPositionalEncoding: Module {
    let pos_enc: MLXArray

    init(hiddenSize: Int, maxSequenceLength: Int) {
        let divisorTerm = (0 ..< hiddenSize / 2).map { index -> Float in
            return Float(Foundation.exp((-Foundation.log(10_000) / Double(hiddenSize)) * Double(index * 2)))
        }

        var values = Array(repeating: Float.zero, count: maxSequenceLength * hiddenSize)
        let scale = Float(Foundation.sqrt(Double(hiddenSize)))
        for position in 0 ..< maxSequenceLength {
            for (index, divisor) in divisorTerm.enumerated() {
                let angle = Float(position) * divisor
                values[(position * hiddenSize) + (index * 2)] = Float(Foundation.sin(Double(angle))) / scale
                values[(position * hiddenSize) + (index * 2) + 1] = Float(Foundation.cos(Double(angle))) / scale
            }
        }

        self.pos_enc = MLXArray(values, [maxSequenceLength, hiddenSize])
    }

    func callAsFunction(_ positions: MLXArray) -> MLXArray {
        pos_enc[positions.asType(.int32)]
    }
}

final class CohereNativeDecoderAttention: Module {
    let qkv_proj: Linear
    let out_proj: Linear

    let hiddenSize: Int
    let headCount: Int
    let headDimension: Int
    let scale: Float
    private let attentionDType: DType = .float32

    init(hiddenSize: Int, headCount: Int) {
        let headDimension = hiddenSize / headCount
        self.qkv_proj = Linear(hiddenSize, hiddenSize * 3)
        self.out_proj = Linear(hiddenSize, hiddenSize)
        self.hiddenSize = hiddenSize
        self.headCount = headCount
        self.headDimension = headDimension
        self.scale = Foundation.pow(Float(headDimension), -0.5)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        contextStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let targetLength = hiddenStates.dim(1)
        let outputDType = hiddenStates.dtype
        let hiddenQKV = qkv_proj(hiddenStates)
        let hiddenPieces = hiddenQKV.split(indices: [hiddenSize, hiddenSize * 2], axis: -1)

        let source = contextStates ?? hiddenStates
        let sourcePieces =
            if contextStates == nil {
                hiddenPieces
            } else {
                qkv_proj(source).split(indices: [hiddenSize, hiddenSize * 2], axis: -1)
            }

        let query = hiddenPieces[0]
            .asType(attentionDType)
            .reshaped([batch, targetLength, headCount, headDimension])
            .transposed(0, 2, 1, 3)
        let key = sourcePieces[1]
            .asType(attentionDType)
            .reshaped([batch, source.dim(1), headCount, headDimension])
            .transposed(0, 2, 1, 3)
        let value = sourcePieces[2]
            .asType(attentionDType)
            .reshaped([batch, source.dim(1), headCount, headDimension])
            .transposed(0, 2, 1, 3)

        var scores = matmul(query, key.transposed(0, 1, 3, 2)) * scale
        if let attentionMask {
            scores = scores + attentionMask.asType(attentionDType)
        }

        let attention = softmax(scores, axis: -1, precise: true)
        let attended = matmul(attention, value)
        return out_proj(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped([batch, targetLength, hiddenSize])
                .asType(outputDType)
        )
    }
}

final class CohereNativeDecoderFeedForward: Module {
    let dense_in: Linear
    let dense_out: Linear
    let activationName: String

    init(hiddenSize: Int, innerSize: Int, activationName: String) {
        self.dense_in = Linear(hiddenSize, innerSize)
        self.dense_out = Linear(innerSize, hiddenSize)
        self.activationName = activationName.lowercased().replacingOccurrences(of: "swish", with: "silu")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = dense_in(x)
        let activated: MLXArray
        switch activationName {
        case "silu":
            activated = silu(projected)
        default:
            activated = relu(projected)
        }
        return dense_out(activated)
    }
}

final class CohereNativeTransformerDecoderLayer: Module {
    let layer_norm_1: LayerNorm
    let first_sub_layer: CohereNativeDecoderAttention
    let layer_norm_2: LayerNorm
    let second_sub_layer: CohereNativeDecoderAttention
    let layer_norm_3: LayerNorm
    let third_sub_layer: CohereNativeDecoderFeedForward

    init(hiddenSize: Int, innerSize: Int, headCount: Int, activationName: String) {
        self.layer_norm_1 = LayerNorm(dimensions: hiddenSize)
        self.first_sub_layer = CohereNativeDecoderAttention(hiddenSize: hiddenSize, headCount: headCount)
        self.layer_norm_2 = LayerNorm(dimensions: hiddenSize)
        self.second_sub_layer = CohereNativeDecoderAttention(hiddenSize: hiddenSize, headCount: headCount)
        self.layer_norm_3 = LayerNorm(dimensions: hiddenSize)
        self.third_sub_layer = CohereNativeDecoderFeedForward(
            hiddenSize: hiddenSize,
            innerSize: innerSize,
            activationName: activationName
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?
    ) -> MLXArray {
        let selfResidual = hiddenStates
        let selfOutput = first_sub_layer(
            layer_norm_1(hiddenStates),
            contextStates: nil,
            attentionMask: selfAttentionMask
        )
        let hiddenStates = selfResidual + selfOutput

        let crossResidual = hiddenStates
        let crossOutput = second_sub_layer(
            layer_norm_2(hiddenStates),
            contextStates: encoderHiddenStates,
            attentionMask: crossAttentionMask
        )
        let crossHiddenStates = crossResidual + crossOutput

        let feedForwardResidual = crossHiddenStates
        return feedForwardResidual + third_sub_layer(layer_norm_3(crossHiddenStates))
    }
}

final class CohereNativeTransformerDecoderEmbedding: Module {
    let token_embedding: Embedding
    let position_embedding: CohereNativeFixedPositionalEncoding
    let layer_norm: LayerNorm
    private let computeDType: DType = .float32

    init(vocabularySize: Int, hiddenSize: Int, maxSequenceLength: Int) {
        self.token_embedding = Embedding(embeddingCount: vocabularySize, dimensions: hiddenSize)
        self.position_embedding = CohereNativeFixedPositionalEncoding(
            hiddenSize: hiddenSize,
            maxSequenceLength: maxSequenceLength
        )
        self.layer_norm = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(_ inputIDs: MLXArray, positions: MLXArray) -> MLXArray {
        let embeddedTokens = token_embedding(inputIDs.asType(.int32)).asType(computeDType)
        let embeddedPositions = position_embedding(positions).asType(computeDType)
        return layer_norm(embeddedTokens + embeddedPositions).asType(computeDType)
    }
}

final class CohereNativeTransformerDecoderCore: Module {
    let layers: [CohereNativeTransformerDecoderLayer]
    let final_layer_norm: LayerNorm

    init(hiddenSize: Int, innerSize: Int, headCount: Int, layerCount: Int, activationName: String) {
        self.layers = (0 ..< layerCount).map { _ in
            CohereNativeTransformerDecoderLayer(
                hiddenSize: hiddenSize,
                innerSize: innerSize,
                headCount: headCount,
                activationName: activationName
            )
        }
        self.final_layer_norm = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?
    ) -> MLXArray {
        var hiddenStates = hiddenStates
        for layer in layers {
            hiddenStates = layer(
                hiddenStates,
                encoderHiddenStates: encoderHiddenStates,
                selfAttentionMask: selfAttentionMask,
                crossAttentionMask: crossAttentionMask
            )
        }
        return final_layer_norm(hiddenStates)
    }
}

final class CohereNativeTransformerDecoder: Module {
    let embedding: CohereNativeTransformerDecoderEmbedding
    let core: CohereNativeTransformerDecoderCore

    init(config: CohereNativeModelConfig.DecoderConfig.CoreConfig, vocabularySize: Int) {
        self.embedding = CohereNativeTransformerDecoderEmbedding(
            vocabularySize: vocabularySize,
            hiddenSize: config.hiddenSize,
            maxSequenceLength: config.maxSequenceLength
        )
        self.core = CohereNativeTransformerDecoderCore(
            hiddenSize: config.hiddenSize,
            innerSize: config.innerSize,
            headCount: config.headCount,
            layerCount: config.layerCount,
            activationName: config.hiddenActivation
        )
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        positions: MLXArray,
        encoderHiddenStates: MLXArray,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?
    ) -> MLXArray {
        let embedded = embedding(inputIDs, positions: positions)
        return core(
            embedded,
            encoderHiddenStates: encoderHiddenStates,
            selfAttentionMask: selfAttentionMask,
            crossAttentionMask: crossAttentionMask
        )
    }
}

final class CohereNativeConditionalGenerationModel: Module {
    let encoder: CohereNativeConformerEncoder
    let bridge_proj: Linear
    let decoder: CohereNativeTransformerDecoder
    let lm_head: Linear
    let usesLogSoftmax: Bool

    init(config: CohereNativeModelConfig) {
        self.encoder = CohereNativeConformerEncoder(config: config.encoder)
        self.bridge_proj = Linear(config.encoder.dModel, config.decoder.config.hiddenSize)
        self.decoder = CohereNativeTransformerDecoder(
            config: config.decoder.config,
            vocabularySize: config.head.classCount
        )
        self.lm_head = Linear(config.head.hiddenSize, config.head.classCount)
        self.usesLogSoftmax = config.head.logSoftmax
    }

    func callAsFunction(
        inputFeatures: MLXArray,
        lengths: [Int],
        inputIDs: MLXArray,
        positions: MLXArray? = nil
    ) -> (logits: MLXArray, encoderHiddenStates: MLXArray, encodedLengths: [Int]) {
        let (encoderHiddenStates, encodedLengths) = encode(
            inputFeatures: inputFeatures,
            lengths: lengths
        )
        let logits = decode(
            inputIDs: inputIDs,
            positions: positions,
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )
        return (logits, encoderHiddenStates, encodedLengths)
    }

    func encode(
        inputFeatures: MLXArray,
        lengths: [Int]
    ) -> (encoderHiddenStates: MLXArray, encodedLengths: [Int]) {
        let (encoderHiddenStates, encodedLengths) = encoder(inputFeatures, lengths: lengths)
        let bridgedEncoderStates = bridge_proj(encoderHiddenStates)
        return (bridgedEncoderStates, encodedLengths)
    }

    func decode(
        inputIDs: MLXArray,
        positions: MLXArray? = nil,
        encoderHiddenStates: MLXArray,
        encodedLengths: [Int]
    ) -> MLXArray {
        let normalizedInputIDs = inputIDs.asType(.int32)
        let decoderComputeDType: DType = .float32
        let resolvedPositions = Self.resolvePositions(
            positions,
            batchSize: normalizedInputIDs.dim(0),
            sequenceLength: normalizedInputIDs.dim(1)
        )
        let selfAttentionMask = Self.makeSelfAttentionMask(
            batchSize: normalizedInputIDs.dim(0),
            targetLength: normalizedInputIDs.dim(1),
            dtype: decoderComputeDType
        )
        let crossAttentionMask = Self.makeCrossAttentionMask(
            lengths: encodedLengths,
            sourceLength: encoderHiddenStates.dim(1),
            dtype: decoderComputeDType
        )
        let decoderEncoderHiddenStates = encoderHiddenStates.asType(decoderComputeDType)

        let decoderHiddenStates = decoder(
            normalizedInputIDs,
            positions: resolvedPositions,
            encoderHiddenStates: decoderEncoderHiddenStates,
            selfAttentionMask: selfAttentionMask,
            crossAttentionMask: crossAttentionMask
        )

        let logits = Self.applyLinear(
            decoderHiddenStates.asType(decoderComputeDType),
            weight: lm_head.weight.asType(decoderComputeDType),
            bias: lm_head.bias?.asType(decoderComputeDType)
        )
        return usesLogSoftmax ? logSoftmax(logits, axis: -1) : logits
    }

    private static func applyLinear(_ x: MLXArray, weight: MLXArray, bias: MLXArray?) -> MLXArray {
        if let bias {
            return addMM(bias, x, weight.T)
        }

        return matmul(x, weight.T)
    }

    private static func makeSelfAttentionMask(batchSize: Int, targetLength: Int, dtype: DType) -> MLXArray {
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

    private static func makeCrossAttentionMask(lengths: [Int], sourceLength: Int, dtype: DType) -> MLXArray {
        var values = Array(repeating: Float.zero, count: lengths.count * sourceLength)
        for (batchIndex, length) in lengths.enumerated() where length < sourceLength {
            let rowStart = batchIndex * sourceLength
            for sourceIndex in length ..< sourceLength {
                values[rowStart + sourceIndex] = -1_000_000_000
            }
        }

        return MLXArray(values, [lengths.count, 1, 1, sourceLength]).asType(dtype)
    }

    private static func resolvePositions(
        _ positions: MLXArray?,
        batchSize: Int,
        sequenceLength: Int
    ) -> MLXArray {
        if let positions {
            return positions.asType(.int32)
        }

        let basePositions = MLXArray(Array(0 ..< sequenceLength))
            .asType(.int32)
            .expandedDimensions(axis: 0)
        if batchSize == 1 {
            return basePositions
        }

        return repeated(basePositions, count: batchSize, axis: 0)
    }
}

enum CohereNativeEncoderLoader {
    private static let mlxCacheLimit = 8 * 1024 * 1024 * 1024

    static func loadPreparedState(from bootstrap: CohereNativeBootstrap) throws -> CohereNativePreparedState {
        GPU.set(cacheLimit: mlxCacheLimit)

        let model = CohereNativeConditionalGenerationModel(config: bootstrap.config)
        let rawWeights = try loadWeights(from: bootstrap.modelDirectory)
        let weights = filterWeights(preprocessCheckpointWeights(rawWeights), for: model)

        model.update(parameters: ModuleParameters.unflattened(weights))
        model.train(false)
        eval(model)

        let silenceDuration = 1
        let silence = Array(
            repeating: Float.zero,
            count: bootstrap.config.audioConfiguration.sampleRate * silenceDuration
        )
        let features = bootstrap.featureExtractor
            .extractLogMelFeatures(from: silence)
            .expandedDimensions(axis: 0)
            .asType(model.encoder.subsampling.out.weight.dtype)
        let lengths = [features.dim(2)]
        let promptIDs = MLXArray(bootstrap.promptTokenIDs, [1, bootstrap.promptTokenIDs.count]).asType(.int32)
        let (logits, encoderHiddenStates, encodedLengths) = model(
            inputFeatures: features,
            lengths: lengths,
            inputIDs: promptIDs
        )
        eval(logits, encoderHiddenStates)

        let summary = CohereNativeEncoderWarmupSummary(
            outputShape: encoderHiddenStates.shape,
            encodedLengths: encodedLengths,
            decoderLogitsShape: logits.shape,
            parameterCount: weights.count
        )

        return CohereNativePreparedState(
            bootstrap: bootstrap,
            model: model,
            summary: summary
        )
    }

    static func preprocessCheckpointWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var processed: [String: MLXArray] = [:]
        processed.reserveCapacity(weights.count)

        for (name, value) in weights {
            if name.hasSuffix("num_batches_tracked") {
                continue
            }

            if name.hasSuffix(".weight"), value.ndim == 3 {
                processed[name] = contiguous(value.transposed(0, 2, 1))
            } else if name.hasSuffix(".weight"), value.ndim == 4 {
                processed[name] = contiguous(value.transposed(0, 2, 3, 1))
            } else {
                processed[name] = value
            }
        }

        return processed
    }

    private static func filterWeights(
        _ weights: [String: MLXArray],
        for model: CohereNativeConditionalGenerationModel
    ) -> [String: MLXArray] {
        let allowedKeys = Set(model.parameters().flattened().map(\.0))
        return weights.reduce(into: [:]) { partialResult, entry in
            if allowedKeys.contains(entry.key) {
                partialResult[entry.key] = entry.value
            }
        }
    }

    private static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let singleFileURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: singleFileURL.path) else {
            throw CohereNativeEncoderLoaderError.missingWeights(directory)
        }

        return try loadArrays(url: singleFileURL)
    }
}
