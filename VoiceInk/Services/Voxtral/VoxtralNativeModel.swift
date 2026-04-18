import Foundation
import MLX
import MLXNN

struct VoxtralNativeWarmupSummary: Sendable {
    let modelDirectory: URL
    let promptTokenCount: Int
    let delayTokenCount: Int
    let shardCount: Int
    let parameterCount: Int
    let quantized: Bool
}

enum VoxtralNativeLoaderError: LocalizedError {
    case missingWeights(URL)

    var errorDescription: String? {
        switch self {
        case .missingWeights(let directory):
            return "No Voxtral weight files were found in \(directory.path)."
        }
    }
}

private func tail(_ array: MLXArray, count: Int, axis: Int) -> MLXArray {
    let resolvedAxis = axis >= 0 ? axis : array.ndim + axis
    let length = array.dim(resolvedAxis)
    guard count < length else { return array }
    return array[(length - count)..., axis: resolvedAxis]
}

final class VoxtralRotatingKVCache {
    static let allocationStep = 256

    let maxSize: Int

    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var offset = 0
    private var index = 0

    init(maxSize: Int) {
        self.maxSize = maxSize
    }

    func updateAndFetch(_ newKeys: MLXArray, _ newValues: MLXArray) -> (MLXArray, MLXArray) {
        if newKeys.dim(2) == 1 {
            return updateInPlace(newKeys, newValues)
        } else {
            return updateConcat(newKeys, newValues)
        }
    }

    private func updateConcat(_ newKeys: MLXArray, _ newValues: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil || self.values == nil {
            self.keys = tail(newKeys, count: maxSize, axis: 2)
            self.values = tail(newValues, count: maxSize, axis: 2)
            offset += newKeys.dim(2)
            index = self.keys?.dim(2) ?? 0
            return (self.keys!, self.values!)
        }

        self.keys = temporalOrder(self.keys!)
        self.values = temporalOrder(self.values!)

        self.keys = tail(concatenated([self.keys!, newKeys], axis: 2), count: maxSize, axis: 2)
        self.values = tail(concatenated([self.values!, newValues], axis: 2), count: maxSize, axis: 2)
        self.offset += newKeys.dim(2)
        self.index = self.keys?.dim(2) ?? 0
        return (self.keys!, self.values!)
    }

    private func updateInPlace(_ newKeys: MLXArray, _ newValues: MLXArray) -> (MLXArray, MLXArray) {
        let batch = newKeys.dim(0)
        let keyValueHeads = newKeys.dim(1)
        let sequenceLength = newKeys.dim(2)
        let keyHeadDimension = newKeys.dim(3)
        let valueHeadDimension = newValues.dim(3)
        let previousOffset = offset

        if self.keys == nil || self.values == nil ||
            (previousOffset >= self.keys!.dim(2) && self.keys!.dim(2) < maxSize)
        {
            let additionalSize = min(Self.allocationStep, maxSize - previousOffset)
            let newKeyBuffer = MLXArray.zeros(
                [batch, keyValueHeads, additionalSize, keyHeadDimension],
                dtype: newKeys.dtype
            )
            let newValueBuffer = MLXArray.zeros(
                [batch, keyValueHeads, additionalSize, valueHeadDimension],
                dtype: newValues.dtype
            )

            if let existingKeys = self.keys, let existingValues = self.values {
                self.keys = concatenated([existingKeys, newKeyBuffer], axis: 2)
                self.values = concatenated([existingValues, newValueBuffer], axis: 2)
            } else {
                self.keys = newKeyBuffer
                self.values = newValueBuffer
            }

            self.index = previousOffset
        }

        let trimSize = self.keys!.dim(2) - maxSize
        if trimSize > 0 {
            self.keys = trim(trimSize, from: self.keys!)
            self.values = trim(trimSize, from: self.values!)
            self.index = maxSize
        }

        if self.index == maxSize {
            self.index = 0
        }

        self.keys![index ..< index + sequenceLength, axis: 2] = newKeys
        self.values![index ..< index + sequenceLength, axis: 2] = newValues

        self.offset += sequenceLength
        self.index += sequenceLength

        if self.offset < maxSize {
            return (
                self.keys![0 ..< self.offset, axis: 2],
                self.values![0 ..< self.offset, axis: 2]
            )
        }

        return (self.keys!, self.values!)
    }

    private func trim(_ trimSize: Int, from value: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var segments: [MLXArray] = []
        if trimSize > 0 {
            segments.append(value[trimSize..., axis: 2])
        } else {
            segments.append(value)
        }
        if let append {
            segments.append(append)
        }
        return segments.count == 1 ? segments[0] : concatenated(segments, axis: 2)
    }

    private func temporalOrder(_ value: MLXArray) -> MLXArray {
        if index == value.dim(2) {
            return value
        } else if index < offset {
            return concatenated(
                [
                    value[index..., axis: 2],
                    value[..<index, axis: 2],
                ],
                axis: 2
            )
        } else {
            return value[..<index, axis: 2]
        }
    }
}

final class VoxtralTimeEmbedding {
    private let inv_freq: MLXArray

    init(dim: Int, theta: Float = 10_000) {
        let halfDim = max(1, dim / 2)
        let exponents = MLXArray(Array(0 ..< halfDim)).asType(.float32)
        let scale = Float(-Foundation.log(Double(theta)) / Double(halfDim))
        self.inv_freq = exp(exponents * scale)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let steps = t.reshaped([-1, 1]).asType(.float32)
        let embeddings = steps * inv_freq.reshaped([1, -1])
        return concatenated([cos(embeddings), sin(embeddings)], axis: -1)
    }
}

final class VoxtralAudioLanguageAdapter: Module {
    @ModuleInfo var w_in: Linear
    @ModuleInfo var w_out: Linear

    init(inputDimension: Int, outputDimension: Int) {
        self.w_in = Linear(inputDimension, outputDimension, bias: false)
        self.w_out = Linear(outputDimension, outputDimension, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w_out(gelu(w_in(x)))
    }
}

final class VoxtralCausalConv1d: Module {
    let weight: MLXArray
    let bias: MLXArray

    let stride: Int
    let kernel_size: Int
    let padding_total: Int

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.weight = MLXArray.zeros([outputChannels, kernelSize, inputChannels], dtype: .float32)
        self.bias = MLXArray.zeros([outputChannels], dtype: .float32)
        self.stride = stride
        self.kernel_size = kernelSize
        self.padding_total = kernelSize - stride
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddedInput =
            if padding_total > 0 {
                padded(x, widths: [[0, 0], [padding_total, 0], [0, 0]])
            } else {
                x
            }

        return conv1d(paddedInput, weight, stride: stride) + bias
    }
}

final class VoxtralEncoderAttention: Module {
    @ModuleInfo var q_proj: Linear
    @ModuleInfo var k_proj: Linear
    @ModuleInfo var v_proj: Linear
    @ModuleInfo var o_proj: Linear

    let n_heads: Int
    let head_dim: Int
    let scale: Float
    let rope_theta: Float

    init(dim: Int, headCount: Int, headDimension: Int, ropeTheta: Float) {
        self.q_proj = Linear(dim, headCount * headDimension, bias: true)
        self.k_proj = Linear(dim, headCount * headDimension, bias: false)
        self.v_proj = Linear(dim, headCount * headDimension, bias: true)
        self.o_proj = Linear(headCount * headDimension, dim, bias: true)
        self.n_heads = headCount
        self.head_dim = headDimension
        self.scale = pow(Float(headDimension), -0.5)
        self.rope_theta = ropeTheta
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VoxtralRotatingKVCache? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)

        var q = q_proj(x).reshaped([batch, sequenceLength, n_heads, head_dim]).transposed(0, 2, 1, 3)
        var k = k_proj(x).reshaped([batch, sequenceLength, n_heads, head_dim]).transposed(0, 2, 1, 3)
        var v = v_proj(x).reshaped([batch, sequenceLength, n_heads, head_dim]).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = MLXFast.RoPE(q, dimensions: head_dim, traditional: true, base: rope_theta, scale: 1.0, offset: offset)
        k = MLXFast.RoPE(k, dimensions: head_dim, traditional: true, base: rope_theta, scale: 1.0, offset: offset)

        if let cache {
            (k, v) = cache.updateAndFetch(k, v)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        return o_proj(attended.transposed(0, 2, 1, 3).reshaped([batch, sequenceLength, n_heads * head_dim]))
    }
}

final class VoxtralEncoderSwiGLU: Module {
    @ModuleInfo var gate_proj: Linear
    @ModuleInfo var up_proj: Linear
    @ModuleInfo var down_proj: Linear

    init(dim: Int, hiddenDimension: Int) {
        self.gate_proj = Linear(dim, hiddenDimension, bias: false)
        self.up_proj = Linear(dim, hiddenDimension, bias: false)
        self.down_proj = Linear(hiddenDimension, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(silu(gate_proj(x)) * up_proj(x))
    }
}

final class VoxtralEncoderLayer: Module {
    @ModuleInfo var attn_norm: RMSNorm
    @ModuleInfo var attention: VoxtralEncoderAttention
    @ModuleInfo var ffn_norm: RMSNorm
    @ModuleInfo var mlp: VoxtralEncoderSwiGLU

    init(dim: Int, headCount: Int, headDimension: Int, hiddenDimension: Int, ropeTheta: Float) {
        self.attn_norm = RMSNorm(dimensions: dim, eps: 1e-5)
        self.attention = VoxtralEncoderAttention(
            dim: dim,
            headCount: headCount,
            headDimension: headDimension,
            ropeTheta: ropeTheta
        )
        self.ffn_norm = RMSNorm(dimensions: dim, eps: 1e-5)
        self.mlp = VoxtralEncoderSwiGLU(dim: dim, hiddenDimension: hiddenDimension)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VoxtralRotatingKVCache? = nil
    ) -> MLXArray {
        let attended = x + attention(attn_norm(x), mask: mask, cache: cache)
        return attended + mlp(ffn_norm(attended))
    }
}

final class VoxtralCausalWhisperEncoder: Module {
    @ModuleInfo var conv1: VoxtralCausalConv1d
    @ModuleInfo var conv2: VoxtralCausalConv1d
    @ModuleInfo var layers: [VoxtralEncoderLayer]
    @ModuleInfo var norm: RMSNorm

    let sliding_window: Int

    init(config: VoxtralNativeModelConfig.Multimodal.WhisperModelArguments.EncoderArguments) {
        self.conv1 = VoxtralCausalConv1d(
            inputChannels: config.audioEncodingArguments.numMelBins,
            outputChannels: config.dimension,
            kernelSize: 3,
            stride: 1
        )
        self.conv2 = VoxtralCausalConv1d(
            inputChannels: config.dimension,
            outputChannels: config.dimension,
            kernelSize: 3,
            stride: 2
        )
        self.layers = (0 ..< config.layerCount).map { _ in
            VoxtralEncoderLayer(
                dim: config.dimension,
                headCount: config.headCount,
                headDimension: config.headDimension,
                hiddenDimension: config.hiddenDimension,
                ropeTheta: Float(config.ropeTheta)
            )
        }
        self.norm = RMSNorm(dimensions: config.dimension, eps: 1e-5)
        self.sliding_window = config.slidingWindow
    }

    func forwardConv(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed().expandedDimensions(axis: 0).asType(conv1.weight.dtype)
        x = gelu(conv1(x))
        x = gelu(conv2(x))
        return x
    }

    func forwardConvStep(
        _ newMel: MLXArray,
        conv1Tail: MLXArray?,
        conv2Tail: MLXArray?
    ) -> (output: MLXArray, conv1Tail: MLXArray, conv2Tail: MLXArray) {
        let conv1Input: MLXArray
        if let conv1Tail {
            conv1Input = concatenated([conv1Tail, newMel], axis: 1)
        } else {
            conv1Input = padded(newMel, widths: [[0, 0], [conv1.padding_total, 0], [0, 0]])
        }
        let nextConv1Tail = tail(newMel, count: conv1.padding_total, axis: 1)
        var x = gelu(conv1d(conv1Input, conv1.weight, stride: conv1.stride) + conv1.bias)

        let conv2Input: MLXArray
        if let conv2Tail {
            conv2Input = concatenated([conv2Tail, x], axis: 1)
        } else {
            conv2Input = padded(x, widths: [[0, 0], [conv2.padding_total, 0], [0, 0]])
        }
        let nextConv2Tail = tail(x, count: conv2.padding_total, axis: 1)
        x = gelu(conv1d(conv2Input, conv2.weight, stride: conv2.stride) + conv2.bias)

        return (x, nextConv1Tail, nextConv2Tail)
    }

    func forwardTransformer(_ x: MLXArray, cache: [VoxtralRotatingKVCache]? = nil) -> MLXArray {
        var hiddenStates = x
        for (index, layer) in layers.enumerated() {
            let layerCache = cache?[index]
            hiddenStates = layer(hiddenStates, mask: .causal, cache: layerCache)
        }
        return norm(hiddenStates)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = forwardConv(mel)
        for layer in layers {
            x = layer(x, mask: .causal)
        }
        return norm(x)
    }
}

final class VoxtralDecoderAttention: Module {
    @ModuleInfo var q_proj: Linear
    @ModuleInfo var k_proj: Linear
    @ModuleInfo var v_proj: Linear
    @ModuleInfo var o_proj: Linear

    let n_heads: Int
    let n_kv_heads: Int
    let head_dim: Int
    let scale: Float
    let rope_theta: Float

    init(dim: Int, headCount: Int, keyValueHeadCount: Int, headDimension: Int, ropeTheta: Float) {
        self.q_proj = Linear(dim, headCount * headDimension, bias: false)
        self.k_proj = Linear(dim, keyValueHeadCount * headDimension, bias: false)
        self.v_proj = Linear(dim, keyValueHeadCount * headDimension, bias: false)
        self.o_proj = Linear(headCount * headDimension, dim, bias: false)
        self.n_heads = headCount
        self.n_kv_heads = keyValueHeadCount
        self.head_dim = headDimension
        self.scale = pow(Float(headDimension), -0.5)
        self.rope_theta = ropeTheta
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VoxtralRotatingKVCache? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)

        var q = q_proj(x).reshaped([batch, sequenceLength, n_heads, head_dim]).transposed(0, 2, 1, 3)
        var k = k_proj(x).reshaped([batch, sequenceLength, n_kv_heads, head_dim]).transposed(0, 2, 1, 3)
        var v = v_proj(x).reshaped([batch, sequenceLength, n_kv_heads, head_dim]).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = MLXFast.RoPE(q, dimensions: head_dim, traditional: true, base: rope_theta, scale: 1.0, offset: offset)
        k = MLXFast.RoPE(k, dimensions: head_dim, traditional: true, base: rope_theta, scale: 1.0, offset: offset)

        if let cache {
            (k, v) = cache.updateAndFetch(k, v)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        return o_proj(attended.transposed(0, 2, 1, 3).reshaped([batch, sequenceLength, n_heads * head_dim]))
    }
}

final class VoxtralDecoderSwiGLU: Module {
    @ModuleInfo var gate_proj: Linear
    @ModuleInfo var up_proj: Linear
    @ModuleInfo var down_proj: Linear

    init(dim: Int, hiddenDimension: Int) {
        self.gate_proj = Linear(dim, hiddenDimension, bias: false)
        self.up_proj = Linear(dim, hiddenDimension, bias: false)
        self.down_proj = Linear(hiddenDimension, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(silu(gate_proj(x)) * up_proj(x))
    }
}

final class VoxtralAdaptiveNorm: Module {
    @ModuleInfo var linear_in: Linear
    @ModuleInfo var linear_out: Linear

    init(dim: Int, conditioningDimension: Int) {
        self.linear_in = Linear(dim, conditioningDimension, bias: false)
        self.linear_out = Linear(conditioningDimension, dim, bias: false)
    }

    func callAsFunction(_ conditioning: MLXArray) -> MLXArray {
        linear_out(gelu(linear_in(conditioning)))
    }
}

final class VoxtralDecoderLayer: Module {
    @ModuleInfo var attn_norm: RMSNorm
    @ModuleInfo var attention: VoxtralDecoderAttention
    @ModuleInfo var ada_norm: VoxtralAdaptiveNorm
    @ModuleInfo var ffn_norm: RMSNorm
    @ModuleInfo var mlp: VoxtralDecoderSwiGLU

    init(
        dim: Int,
        headCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        hiddenDimension: Int,
        ropeTheta: Float,
        conditioningDimension: Int
    ) {
        self.attn_norm = RMSNorm(dimensions: dim, eps: 1e-5)
        self.attention = VoxtralDecoderAttention(
            dim: dim,
            headCount: headCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            ropeTheta: ropeTheta
        )
        self.ada_norm = VoxtralAdaptiveNorm(dim: dim, conditioningDimension: conditioningDimension)
        self.ffn_norm = RMSNorm(dimensions: dim, eps: 1e-5)
        self.mlp = VoxtralDecoderSwiGLU(dim: dim, hiddenDimension: hiddenDimension)
    }

    func callAsFunction(
        _ x: MLXArray,
        t_cond: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VoxtralRotatingKVCache? = nil
    ) -> MLXArray {
        let attended = x + attention(attn_norm(x), mask: mask, cache: cache)
        let conditioned = ffn_norm(attended) * (1.0 + ada_norm(t_cond))
        return attended + mlp(conditioned)
    }
}

final class VoxtralLanguageModel: Module {
    @ModuleInfo var embed_tokens: Embedding
    @ModuleInfo var layers: [VoxtralDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let dimension: Int

    init(config: VoxtralNativeModelConfig) {
        self.embed_tokens = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.dimension
        )
        self.layers = (0 ..< config.layerCount).map { _ in
            VoxtralDecoderLayer(
                dim: config.dimension,
                headCount: config.headCount,
                keyValueHeadCount: config.keyValueHeadCount,
                headDimension: config.headDimension,
                hiddenDimension: config.hiddenDimension,
                ropeTheta: Float(config.ropeTheta),
                conditioningDimension: config.conditioningDimension
            )
        }
        self.norm = RMSNorm(dimensions: config.dimension, eps: 1e-5)
        self.dimension = config.dimension
    }

    func embed(_ inputIDs: MLXArray) -> MLXArray {
        embed_tokens(inputIDs)
    }

    func callAsFunction(
        _ x: MLXArray,
        t_cond: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [VoxtralRotatingKVCache]? = nil
    ) -> MLXArray {
        let conditioned = t_cond.asType(x.dtype)
        var hiddenStates = x

        for (index, layer) in layers.enumerated() {
            let layerCache = cache?[index]
            hiddenStates = layer(hiddenStates, t_cond: conditioned, mask: mask, cache: layerCache)
        }

        return embed_tokens.asLinear(norm(hiddenStates))
    }
}

final class VoxtralRealtimeModel: Module {
    @ModuleInfo var encoder: VoxtralCausalWhisperEncoder
    @ModuleInfo var adapter: VoxtralAudioLanguageAdapter
    @ModuleInfo(key: "language_model") var language_model: VoxtralLanguageModel

    let time_embedding: VoxtralTimeEmbedding
    let downsample_factor: Int
    let encoder_dimension: Int

    init(config: VoxtralNativeModelConfig) {
        let encoderConfig = config.multimodal.whisperModelArguments.encoderArguments
        let downsampleFactor = config.multimodal.whisperModelArguments.downsampleArguments.downsampleFactor

        self.encoder = VoxtralCausalWhisperEncoder(config: encoderConfig)
        self.adapter = VoxtralAudioLanguageAdapter(
            inputDimension: encoderConfig.dimension * downsampleFactor,
            outputDimension: config.dimension
        )
        self._language_model.wrappedValue = VoxtralLanguageModel(config: config)
        self.time_embedding = VoxtralTimeEmbedding(dim: config.dimension)
        self.downsample_factor = downsampleFactor
        self.encoder_dimension = encoderConfig.dimension
    }

    func encode(_ mel: MLXArray) -> MLXArray {
        let melInput: MLXArray
        if mel.dim(1) % 2 != 0 {
            melInput = mel[1..., axis: 1]
        } else {
            melInput = mel
        }

        var x = encoder(melInput)[0]

        let remainder = x.dim(0) % downsample_factor
        if remainder != 0 {
            x = x[remainder..., axis: 0]
        }

        let length = x.dim(0)
        let reshaped = x.reshaped([length / downsample_factor, encoder_dimension * downsample_factor])
        return adapter(reshaped)
    }

    func encodeStep(
        _ newMel: MLXArray,
        conv1Tail: MLXArray?,
        conv2Tail: MLXArray?,
        encoderCache: [VoxtralRotatingKVCache]?,
        downsampleBuffer: MLXArray?
    ) -> (
        embeddings: MLXArray?,
        conv1Tail: MLXArray,
        conv2Tail: MLXArray,
        encoderCache: [VoxtralRotatingKVCache],
        downsampleBuffer: MLXArray?
    ) {
        let melInput = newMel.transposed().expandedDimensions(axis: 0).asType(encoder.conv1.weight.dtype)
        let (convOutput, nextConv1Tail, nextConv2Tail) = encoder.forwardConvStep(
            melInput,
            conv1Tail: conv1Tail,
            conv2Tail: conv2Tail
        )

        let cache = encoderCache ?? encoder.layers.map { _ in
            VoxtralRotatingKVCache(maxSize: 100_000)
        }

        var hiddenStates = encoder.forwardTransformer(convOutput, cache: cache)[0]
        if let downsampleBuffer {
            hiddenStates = concatenated([downsampleBuffer, hiddenStates], axis: 0)
        }

        let completeLength = (hiddenStates.dim(0) / downsample_factor) * downsample_factor
        guard completeLength > 0 else {
            return (nil, nextConv1Tail, nextConv2Tail, cache, hiddenStates)
        }

        let remainder = hiddenStates.dim(0) - completeLength
        let nextDownsampleBuffer = remainder > 0 ? hiddenStates[completeLength..., axis: 0] : nil
        let completeStates = hiddenStates[..<completeLength, axis: 0]
        let reshaped = completeStates.reshaped(
            [completeLength / downsample_factor, encoder_dimension * downsample_factor]
        )
        return (
            adapter(reshaped),
            nextConv1Tail,
            nextConv2Tail,
            cache,
            nextDownsampleBuffer
        )
    }

    func decode(
        _ embeddings: MLXArray,
        t_cond: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: [VoxtralRotatingKVCache]? = nil
    ) -> MLXArray {
        language_model(embeddings, t_cond: t_cond, mask: mask, cache: cache)
    }
}

final class VoxtralNativePreparedState: @unchecked Sendable {
    let bootstrap: VoxtralNativeBootstrap
    let model: VoxtralRealtimeModel
    let timeConditioning: MLXArray
    let promptEmbeddings: MLXArray
    let summary: VoxtralNativeWarmupSummary

    init(
        bootstrap: VoxtralNativeBootstrap,
        model: VoxtralRealtimeModel,
        timeConditioning: MLXArray,
        promptEmbeddings: MLXArray,
        summary: VoxtralNativeWarmupSummary
    ) {
        self.bootstrap = bootstrap
        self.model = model
        self.timeConditioning = timeConditioning
        self.promptEmbeddings = promptEmbeddings
        self.summary = summary
    }
}

enum VoxtralNativeModelLoader {
    private static let mlxCacheLimit = 4 * 1024 * 1024 * 1024

    static func loadPreparedState(from bootstrap: VoxtralNativeBootstrap) throws -> VoxtralNativePreparedState {
        GPU.set(cacheLimit: mlxCacheLimit)

        let model = VoxtralRealtimeModel(config: bootstrap.modelConfig)

        if let quantization = bootstrap.modelConfig.quantization {
            quantize(
                model: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                filter: { _, module in
                    switch module {
                    case let linear as Linear:
                        linear.weight.dim(-1) % quantization.groupSize == 0
                    case let embedding as Embedding:
                        embedding.weight.dim(-1) % quantization.groupSize == 0
                    default:
                        false
                    }
                }
            )
        }

        let (weights, shardCount) = try loadWeights(from: bootstrap.modelDirectory)
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)

        let promptTokenIDs = MLXArray(bootstrap.prompt.tokenIDs).asType(.int32).expandedDimensions(axis: 0)
        let promptEmbeddings = model.language_model.embed(promptTokenIDs)[0]
        let timeConditioning = model.time_embedding(
            MLXArray([bootstrap.prompt.delayTokenCount]).asType(.float32)
        )
        eval(promptEmbeddings, timeConditioning)

        let summary = VoxtralNativeWarmupSummary(
            modelDirectory: bootstrap.modelDirectory,
            promptTokenCount: bootstrap.prompt.tokenIDs.count,
            delayTokenCount: bootstrap.prompt.delayTokenCount,
            shardCount: shardCount,
            parameterCount: weights.count,
            quantized: bootstrap.modelConfig.quantization != nil
        )

        return VoxtralNativePreparedState(
            bootstrap: bootstrap,
            model: model,
            timeConditioning: timeConditioning,
            promptEmbeddings: promptEmbeddings,
            summary: summary
        )
    }

    private static func loadWeights(from directory: URL) throws -> ([String: MLXArray], Int) {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let indexData = try Data(contentsOf: indexURL)
            let index = try JSONDecoder().decode(VoxtralWeightIndex.self, from: indexData)
            let shardFiles = Array(Set(index.weightMap.values)).sorted()

            var weights: [String: MLXArray] = [:]
            for shardFile in shardFiles {
                let shardURL = directory.appendingPathComponent(shardFile)
                let shardWeights = try loadArrays(url: shardURL)
                for (name, value) in shardWeights {
                    weights[name] = value
                }
            }

            return (weights, shardFiles.count)
        }

        let singleFileURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: singleFileURL.path) else {
            throw VoxtralNativeLoaderError.missingWeights(directory)
        }

        return (try loadArrays(url: singleFileURL), 1)
    }
}

private struct VoxtralWeightIndex: Decodable {
    let weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
