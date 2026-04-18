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
    let promptInputIDs: MLXArray
    let promptPositions: MLXArray
    let singleStepPositions: [MLXArray]
    let decoderCache: CohereNativeDecoderCache

    init(
        bootstrap: CohereNativeBootstrap,
        model: CohereNativeConditionalGenerationModel,
        summary: CohereNativeEncoderWarmupSummary,
        promptInputIDs: MLXArray,
        promptPositions: MLXArray,
        singleStepPositions: [MLXArray],
        decoderCache: CohereNativeDecoderCache
    ) {
        self.bootstrap = bootstrap
        self.model = model
        self.summary = summary
        self.promptInputIDs = promptInputIDs
        self.promptPositions = promptPositions
        self.singleStepPositions = singleStepPositions
        self.decoderCache = decoderCache
    }
}

struct CohereNativeDecoderContext: @unchecked Sendable {
    let encoderHiddenStates: MLXArray
    let crossAttentionMask: MLXArray
}

final class CohereNativeDecoderKVCache: @unchecked Sendable {
    fileprivate var key: MLXArray? {
        guard visibleLength > 0 else {
            return nil
        }
        return visiblePrefix(from: keyStorage)
    }

    fileprivate var value: MLXArray? {
        guard visibleLength > 0 else {
            return nil
        }
        return visiblePrefix(from: valueStorage)
    }

    private let maxSequenceLength: Int?
    private var keyStorage: MLXArray?
    private var valueStorage: MLXArray?
    private var visibleLength = 0

    init(maxSequenceLength: Int? = nil) {
        self.maxSequenceLength = maxSequenceLength
    }

    func append(key newKey: MLXArray, value newValue: MLXArray) -> (MLXArray, MLXArray) {
        if let maxSequenceLength {
            let chunkLength = newKey.dim(2)
            let nextLength = visibleLength + chunkLength
            precondition(
                nextLength <= maxSequenceLength,
                "Cohere native decoder KV cache exceeded its allocated max sequence length."
            )

            if keyStorage == nil || valueStorage == nil {
                keyStorage = MLXArray.zeros(
                    [newKey.dim(0), newKey.dim(1), maxSequenceLength, newKey.dim(3)],
                    dtype: newKey.dtype
                )
                valueStorage = MLXArray.zeros(
                    [newValue.dim(0), newValue.dim(1), maxSequenceLength, newValue.dim(3)],
                    dtype: newValue.dtype
                )
            }

            keyStorage![0..., 0..., visibleLength ..< nextLength, 0...] = newKey
            valueStorage![0..., 0..., visibleLength ..< nextLength, 0...] = newValue
            visibleLength = nextLength
            return (key!, value!)
        }

        if let keyStorage, let valueStorage {
            let appendedKey = concatenated([keyStorage, newKey], axis: 2)
            let appendedValue = concatenated([valueStorage, newValue], axis: 2)
            self.keyStorage = appendedKey
            self.valueStorage = appendedValue
            visibleLength = appendedKey.dim(2)
            return (appendedKey, appendedValue)
        }

        self.keyStorage = newKey
        self.valueStorage = newValue
        visibleLength = newKey.dim(2)
        return (newKey, newValue)
    }

    func prepare(key newKey: MLXArray, value newValue: MLXArray) -> (MLXArray, MLXArray) {
        if let key, let value {
            return (key, value)
        }

        if let maxSequenceLength {
            let visibleLength = newKey.dim(2)
            precondition(
                visibleLength <= maxSequenceLength,
                "Cohere native decoder KV cache prefix exceeded its allocated max sequence length."
            )
            let keyStorageShape = [newKey.dim(0), newKey.dim(1), maxSequenceLength, newKey.dim(3)]
            let valueStorageShape = [newValue.dim(0), newValue.dim(1), maxSequenceLength, newValue.dim(3)]

            if keyStorage?.shape != keyStorageShape || valueStorage?.shape != valueStorageShape {
                keyStorage = MLXArray.zeros(
                    keyStorageShape,
                    dtype: newKey.dtype
                )
                valueStorage = MLXArray.zeros(
                    valueStorageShape,
                    dtype: newValue.dtype
                )
            }

            keyStorage![0..., 0..., 0 ..< visibleLength, 0...] = newKey
            valueStorage![0..., 0..., 0 ..< visibleLength, 0...] = newValue
            self.visibleLength = visibleLength
            return (self.key!, self.value!)
        }

        self.keyStorage = newKey
        self.valueStorage = newValue
        visibleLength = newKey.dim(2)
        return (newKey, newValue)
    }

    func arraysForEval() -> [MLXArray] {
        [keyStorage, valueStorage].compactMap { $0 }
    }

    func clear() {
        reset(keepStorage: false)
    }

    func reset(keepStorage: Bool) {
        if !keepStorage {
            keyStorage = nil
            valueStorage = nil
        }
        
        visibleLength = 0
    }

    func hasAllocatedStorage() -> Bool {
        keyStorage != nil && valueStorage != nil
    }

    func currentVisibleLength() -> Int {
        visibleLength
    }

    func currentStorageShape() -> [Int]? {
        keyStorage?.shape
    }

    func currentValueStorageShape() -> [Int]? {
        valueStorage?.shape
    }

    private func visiblePrefix(from storage: MLXArray?) -> MLXArray? {
        guard let storage else {
            return nil
        }

        if let maxSequenceLength, visibleLength < maxSequenceLength {
            return storage[0..., 0..., 0 ..< visibleLength, 0...]
        }

        return storage
    }
}

final class CohereNativeDecoderLayerCache: @unchecked Sendable {
    let selfAttention: CohereNativeDecoderKVCache
    let crossAttention = CohereNativeDecoderKVCache()

    init(maxSequenceLength: Int) {
        self.selfAttention = CohereNativeDecoderKVCache(maxSequenceLength: maxSequenceLength)
    }

    func arraysForEval() -> [MLXArray] {
        selfAttention.arraysForEval() + crossAttention.arraysForEval()
    }

    func clear() {
        selfAttention.clear()
        crossAttention.clear()
    }

    func resetForTranscription() {
        selfAttention.reset(keepStorage: true)
        crossAttention.reset(keepStorage: false)
    }
}

final class CohereNativeDecoderCache: @unchecked Sendable {
    let layers: [CohereNativeDecoderLayerCache]

    init(layerCount: Int, maxSequenceLength: Int) {
        self.layers = (0 ..< layerCount).map { _ in
            CohereNativeDecoderLayerCache(maxSequenceLength: maxSequenceLength)
        }
    }

    func arraysForEval() -> [MLXArray] {
        layers.flatMap { $0.arraysForEval() }
    }

    func clear() {
        layers.forEach { $0.clear() }
    }

    func resetForTranscription() {
        layers.forEach { $0.resetForTranscription() }
    }
}

final class CohereNativeConformerFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

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
    @ModuleInfo var qkv_proj: Linear
    @ModuleInfo var pos_proj: Linear
    @ModuleInfo var out_proj: Linear
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
    let conv0: Conv2d
    let conv2: Conv2d
    let conv3: Conv2d
    let conv5: Conv2d
    let conv6: Conv2d
    @ModuleInfo var out: Linear

    init(config: CohereNativeModelConfig.EncoderConfig) {
        let convChannels = config.subsamplingConvChannels
        let outputDimensions = config.projectedFeatureCount > 0 ? config.projectedFeatureCount : config.dModel

        self.conv0 = Conv2d(
            inputChannels: 1,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self.conv2 = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            groups: convChannels
        )
        self.conv3 = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 1
        )
        self.conv5 = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            groups: convChannels
        )
        self.conv6 = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 1
        )

        self.out = Linear(
            convChannels * (config.featureCount / config.subsamplingFactor),
            outputDimensions
        )
    }

    var computeDType: DType {
        conv0.weight.dtype
    }

    func callAsFunction(_ x: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        var hiddenStates = x.transposed(0, 2, 1).expandedDimensions(axis: 3)
        var lengths = lengths

        hiddenStates = applyStridedConv(conv0, to: hiddenStates, lengths: &lengths)
        hiddenStates = ReLU()(hiddenStates)
        hiddenStates = applyStridedConv(conv2, to: hiddenStates, lengths: &lengths)
        hiddenStates = applyStridedConv(conv3, to: hiddenStates, lengths: &lengths)
        hiddenStates = ReLU()(hiddenStates)
        hiddenStates = applyStridedConv(conv5, to: hiddenStates, lengths: &lengths)
        hiddenStates = applyStridedConv(conv6, to: hiddenStates, lengths: &lengths)
        hiddenStates = ReLU()(hiddenStates)

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

    private func applyStridedConv(_ convolution: Conv2d, to input: MLXArray, lengths: inout [Int]) -> MLXArray {
        let mask = Self.makeSubsamplingMask(lengths: lengths, time: input.dim(1), dtype: input.dtype)
        let masked = input * mask
        let output = convolution(masked)
        let kernelSize = convolution.weight.dim(1)
        let stride = convolution.stride.0
        let padding = convolution.padding.0
        if stride > 1 {
            lengths = lengths.map { max(1, ($0 + (2 * padding) - kernelSize) / stride + 1) }
        }
        return output
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
        let expectedDType = subsampling.computeDType
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
    @ModuleInfo var qkv_proj: Linear
    @ModuleInfo var out_proj: Linear

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

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            if let attentionMask {
                .array(attentionMask.asType(attentionDType))
            } else {
                .none
            }
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: maskMode
        )
        return out_proj(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped([batch, targetLength, hiddenSize])
                .asType(outputDType)
        )
    }

    func step(
        _ hiddenStates: MLXArray,
        contextStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        cache: CohereNativeDecoderKVCache? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let targetLength = hiddenStates.dim(1)
        let outputDType = hiddenStates.dtype
        let hiddenQKV = qkv_proj(hiddenStates)
        let hiddenPieces = hiddenQKV.split(indices: [hiddenSize, hiddenSize * 2], axis: -1)

        let query = reshapeProjection(hiddenPieces[0], batch: batch, sequenceLength: targetLength)

        let key: MLXArray
        let value: MLXArray
        if let contextStates {
            if let cache, let cachedKey = cache.key, let cachedValue = cache.value {
                key = cachedKey
                value = cachedValue
            } else {
                let sourcePieces = qkv_proj(contextStates).split(indices: [hiddenSize, hiddenSize * 2], axis: -1)
                let projectedKey = reshapeProjection(sourcePieces[1], batch: batch, sequenceLength: contextStates.dim(1))
                let projectedValue = reshapeProjection(sourcePieces[2], batch: batch, sequenceLength: contextStates.dim(1))
                if let cache {
                    (key, value) = cache.prepare(key: projectedKey, value: projectedValue)
                } else {
                    key = projectedKey
                    value = projectedValue
                }
            }
        } else {
            let projectedKey = reshapeProjection(hiddenPieces[1], batch: batch, sequenceLength: targetLength)
            let projectedValue = reshapeProjection(hiddenPieces[2], batch: batch, sequenceLength: targetLength)
            if let cache {
                (key, value) = cache.append(key: projectedKey, value: projectedValue)
            } else {
                key = projectedKey
                value = projectedValue
            }
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            if let attentionMask {
                .array(attentionMask.asType(attentionDType))
            } else {
                .none
            }
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: maskMode
        )
        return out_proj(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped([batch, targetLength, hiddenSize])
                .asType(outputDType)
        )
    }

    func prepareSelfAttentionCache(_ hiddenStates: MLXArray, cache: CohereNativeDecoderKVCache) {
        let batch = hiddenStates.dim(0)
        let targetLength = hiddenStates.dim(1)
        let hiddenQKV = qkv_proj(hiddenStates)
        let hiddenPieces = hiddenQKV.split(indices: [hiddenSize, hiddenSize * 2], axis: -1)
        let projectedKey = reshapeProjection(hiddenPieces[1], batch: batch, sequenceLength: targetLength)
        let projectedValue = reshapeProjection(hiddenPieces[2], batch: batch, sequenceLength: targetLength)
        _ = cache.prepare(key: projectedKey, value: projectedValue)
    }

    func prepareContextCache(_ contextStates: MLXArray, cache: CohereNativeDecoderKVCache) {
        if cache.key != nil, cache.value != nil {
            return
        }

        let batch = contextStates.dim(0)
        let sourceLength = contextStates.dim(1)
        let sourcePieces = qkv_proj(contextStates).split(indices: [hiddenSize, hiddenSize * 2], axis: -1)
        let projectedKey = reshapeProjection(sourcePieces[1], batch: batch, sequenceLength: sourceLength)
        let projectedValue = reshapeProjection(sourcePieces[2], batch: batch, sequenceLength: sourceLength)
        _ = cache.prepare(key: projectedKey, value: projectedValue)
    }

    private func reshapeProjection(_ projection: MLXArray, batch: Int, sequenceLength: Int) -> MLXArray {
        projection
            .asType(attentionDType)
            .reshaped([batch, sequenceLength, headCount, headDimension])
            .transposed(0, 2, 1, 3)
    }
}

final class CohereNativeDecoderFeedForward: Module {
    @ModuleInfo var dense_in: Linear
    @ModuleInfo var dense_out: Linear
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

    func step(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        crossAttentionMask: MLXArray?,
        cache: CohereNativeDecoderLayerCache
    ) -> MLXArray {
        let selfResidual = hiddenStates
        let selfOutput = first_sub_layer.step(
            layer_norm_1(hiddenStates),
            attentionMask: nil,
            cache: cache.selfAttention
        )
        let hiddenStates = selfResidual + selfOutput

        let crossResidual = hiddenStates
        let crossOutput = second_sub_layer.step(
            layer_norm_2(hiddenStates),
            contextStates: encoderHiddenStates,
            attentionMask: crossAttentionMask,
            cache: cache.crossAttention
        )
        let crossHiddenStates = crossResidual + crossOutput

        let feedForwardResidual = crossHiddenStates
        return feedForwardResidual + third_sub_layer(layer_norm_3(crossHiddenStates))
    }

    func prefill(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?,
        cache: CohereNativeDecoderLayerCache
    ) -> MLXArray {
        let normalizedSelfStates = layer_norm_1(hiddenStates)
        first_sub_layer.prepareSelfAttentionCache(normalizedSelfStates, cache: cache.selfAttention)

        let selfResidual = hiddenStates
        let selfOutput = first_sub_layer(
            normalizedSelfStates,
            contextStates: nil,
            attentionMask: selfAttentionMask
        )
        let hiddenStates = selfResidual + selfOutput

        let normalizedCrossStates = layer_norm_2(hiddenStates)
        second_sub_layer.prepareContextCache(encoderHiddenStates, cache: cache.crossAttention)

        let crossResidual = hiddenStates
        let crossOutput = second_sub_layer(
            normalizedCrossStates,
            contextStates: encoderHiddenStates,
            attentionMask: crossAttentionMask
        )
        let crossHiddenStates = crossResidual + crossOutput

        let feedForwardResidual = crossHiddenStates
        return feedForwardResidual + third_sub_layer(layer_norm_3(crossHiddenStates))
    }
}

final class CohereNativeTransformerDecoderEmbedding: Module {
    @ModuleInfo var token_embedding: Embedding
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

    func step(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        crossAttentionMask: MLXArray?,
        cache: CohereNativeDecoderCache
    ) -> MLXArray {
        var hiddenStates = hiddenStates
        for (layer, layerCache) in zip(layers, cache.layers) {
            hiddenStates = layer.step(
                hiddenStates,
                encoderHiddenStates: encoderHiddenStates,
                crossAttentionMask: crossAttentionMask,
                cache: layerCache
            )
        }
        return final_layer_norm(hiddenStates)
    }

    func prefill(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?,
        cache: CohereNativeDecoderCache
    ) -> MLXArray {
        var hiddenStates = hiddenStates
        for (layer, layerCache) in zip(layers, cache.layers) {
            hiddenStates = layer.prefill(
                hiddenStates,
                encoderHiddenStates: encoderHiddenStates,
                selfAttentionMask: selfAttentionMask,
                crossAttentionMask: crossAttentionMask,
                cache: layerCache
            )
        }
        return final_layer_norm(hiddenStates)
    }
}

final class CohereNativeTransformerDecoder: Module {
    let embedding: CohereNativeTransformerDecoderEmbedding
    let core: CohereNativeTransformerDecoderCore
    let maxSequenceLength: Int

    init(config: CohereNativeModelConfig.DecoderConfig.CoreConfig, vocabularySize: Int) {
        self.maxSequenceLength = config.maxSequenceLength
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

    func step(
        _ inputIDs: MLXArray,
        positions: MLXArray,
        encoderHiddenStates: MLXArray,
        crossAttentionMask: MLXArray?,
        cache: CohereNativeDecoderCache
    ) -> MLXArray {
        let embedded = embedding(inputIDs, positions: positions)
        return core.step(
            embedded,
            encoderHiddenStates: encoderHiddenStates,
            crossAttentionMask: crossAttentionMask,
            cache: cache
        )
    }

    func prefill(
        _ inputIDs: MLXArray,
        positions: MLXArray,
        encoderHiddenStates: MLXArray,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?,
        cache: CohereNativeDecoderCache
    ) -> MLXArray {
        let embedded = embedding(inputIDs, positions: positions)
        return core.prefill(
            embedded,
            encoderHiddenStates: encoderHiddenStates,
            selfAttentionMask: selfAttentionMask,
            crossAttentionMask: crossAttentionMask,
            cache: cache
        )
    }
}

final class CohereNativeConditionalGenerationModel: Module {
    let encoder: CohereNativeConformerEncoder
    @ModuleInfo var bridge_proj: Linear
    let decoder: CohereNativeTransformerDecoder
    @ModuleInfo var lm_head: Linear
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

        let logits = lm_head(decoderHiddenStates).asType(decoderComputeDType)
        return usesLogSoftmax ? logSoftmax(logits, axis: -1) : logits
    }

    func makeDecoderCache() -> CohereNativeDecoderCache {
        CohereNativeDecoderCache(
            layerCount: decoder.core.layers.count,
            maxSequenceLength: decoder.maxSequenceLength
        )
    }

    func prepareDecoderContext(
        encoderHiddenStates: MLXArray,
        encodedLengths: [Int]
    ) -> CohereNativeDecoderContext {
        let decoderComputeDType: DType = .float32
        let decoderEncoderHiddenStates = encoderHiddenStates.asType(decoderComputeDType)
        let crossAttentionMask = Self.makeCrossAttentionMask(
            lengths: encodedLengths,
            sourceLength: decoderEncoderHiddenStates.dim(1),
            dtype: decoderComputeDType
        )
        return CohereNativeDecoderContext(
            encoderHiddenStates: decoderEncoderHiddenStates,
            crossAttentionMask: crossAttentionMask
        )
    }

    func decodeStep(
        inputIDs: MLXArray,
        positions: MLXArray,
        decoderContext: CohereNativeDecoderContext,
        cache: CohereNativeDecoderCache,
        applyLogSoftmax: Bool = true
    ) -> MLXArray {
        let decoderHiddenStates = decoder.step(
            inputIDs,
            positions: positions,
            encoderHiddenStates: decoderContext.encoderHiddenStates,
            crossAttentionMask: decoderContext.crossAttentionMask,
            cache: cache
        )
        let logits = lm_head(decoderHiddenStates)
        if usesLogSoftmax && applyLogSoftmax {
            return logSoftmax(logits, axis: -1)
        }

        return logits
    }

    func prefill(
        inputIDs: MLXArray,
        positions: MLXArray,
        decoderContext: CohereNativeDecoderContext,
        cache: CohereNativeDecoderCache,
        applyLogSoftmax: Bool = true
    ) -> MLXArray {
        let normalizedInputIDs = inputIDs.asType(.int32)
        let selfAttentionMask = Self.makeSelfAttentionMask(
            batchSize: normalizedInputIDs.dim(0),
            targetLength: normalizedInputIDs.dim(1),
            dtype: decoderContext.encoderHiddenStates.dtype
        )
        let decoderHiddenStates = decoder.prefill(
            normalizedInputIDs,
            positions: positions.asType(.int32),
            encoderHiddenStates: decoderContext.encoderHiddenStates,
            selfAttentionMask: selfAttentionMask,
            crossAttentionMask: decoderContext.crossAttentionMask,
            cache: cache
        )
        let logits = lm_head(decoderHiddenStates)
        if usesLogSoftmax && applyLogSoftmax {
            return logSoftmax(logits, axis: -1)
        }

        return logits
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

        // If the checkpoint was shipped pre-quantized (4-bit, 6-bit, 8-bit
        // MLX builds), swap matching Linear/Embedding layers for their
        // QuantizedLinear/QuantizedEmbedding counterparts *before* loading
        // weights, so the packed weights / scales / biases land on the right
        // parameter names. fp16 builds (quantization == nil) skip this step.
        if let quantization = bootstrap.config.quantization {
            quantize(
                model: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                filter: { _, module in
                    switch module {
                    case let linear as Linear:
                        return linear.weight.dim(-1) % quantization.groupSize == 0
                    case let embedding as Embedding:
                        return embedding.weight.dim(-1) % quantization.groupSize == 0
                    default:
                        return false
                    }
                }
            )
        }

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
            .asType(model.encoder.subsampling.computeDType)
        let (encoderHiddenStates, encodedLengths) = model.encode(
            inputFeatures: features,
            lengths: [features.dim(2)]
        )
        let decoderContext = model.prepareDecoderContext(
            encoderHiddenStates: encoderHiddenStates,
            encodedLengths: encodedLengths
        )
        let decoderCache = model.makeDecoderCache()
        let promptInputIDs = MLXArray(bootstrap.promptTokenIDs, [1, bootstrap.promptTokenIDs.count]).asType(.int32)
        let promptPositions = MLXArray(Array(0 ..< bootstrap.promptTokenIDs.count), [1, bootstrap.promptTokenIDs.count])
            .asType(.int32)
        let singleStepPositions = (0 ..< bootstrap.config.decoder.config.maxSequenceLength).map { position in
            MLXArray([Int32(position)], [1, 1])
        }
        let logits = model.prefill(
            inputIDs: promptInputIDs,
            positions: promptPositions,
            decoderContext: decoderContext,
            cache: decoderCache,
            applyLogSoftmax: false
        )
        eval(logits, decoderCache.arraysForEval())

        eval(encoderHiddenStates)
        decoderCache.resetForTranscription()

        let summary = CohereNativeEncoderWarmupSummary(
            outputShape: encoderHiddenStates.shape,
            encodedLengths: encodedLengths,
            decoderLogitsShape: logits.shape,
            parameterCount: weights.count
        )

        return CohereNativePreparedState(
            bootstrap: bootstrap,
            model: model,
            summary: summary,
            promptInputIDs: promptInputIDs,
            promptPositions: promptPositions,
            singleStepPositions: singleStepPositions,
            decoderCache: decoderCache
        )
    }

    static func preprocessCheckpointWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var processed: [String: MLXArray] = [:]
        processed.reserveCapacity(weights.count)

        for (name, value) in weights {
            if name.hasSuffix("num_batches_tracked") {
                continue
            }

            if name.hasSuffix(".weight"), value.ndim == 3, shouldTransposeConv1DWeight(named: name, shape: value.shape) {
                processed[name] = contiguous(value.transposed(0, 2, 1))
            } else if name.hasSuffix(".weight"), value.ndim == 4, shouldTransposeConv2DWeight(named: name, shape: value.shape) {
                processed[name] = contiguous(value.transposed(0, 2, 3, 1))
            } else {
                processed[name] = value
            }
        }

        return processed
    }

    private static func shouldTransposeConv1DWeight(named name: String, shape: [Int]) -> Bool {
        guard shape.count == 3 else { return false }

        if name.contains("depthwise_conv") {
            if shape[2] == 1, shape[1] > 1 {
                return false
            }
            return shape[1] <= shape[2]
        }

        if name.contains("pointwise_conv") {
            if shape[1] == 1, shape[2] > 1 {
                return false
            }
            return shape[2] == 1
        }

        return false
    }

    private static func shouldTransposeConv2DWeight(named name: String, shape: [Int]) -> Bool {
        guard shape.count == 4 else { return false }

        if shape[3] == 1, shape[1] > 1, shape[2] > 1 {
            return false
        }
        if shape[1] == 1, shape[2] == 1, shape[3] > 1 {
            return false
        }
        if shape[1] == 1, shape[2] > 1, shape[3] > 1 {
            return true
        }
        if shape[2] == 1, shape[3] == 1, shape[1] > 1 {
            return true
        }

        return name.contains("subsampling.conv.")
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
