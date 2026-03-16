import Foundation

final class VoxtralNativePreparedLease: @unchecked Sendable {
    let preparedState: VoxtralNativePreparedState

    private let modelReference: String
    private let releaseHandler: @Sendable (String) async -> Void
    private let lock = NSLock()
    private var hasReleased = false

    init(
        preparedState: VoxtralNativePreparedState,
        modelReference: String,
        releaseHandler: @escaping @Sendable (String) async -> Void
    ) {
        self.preparedState = preparedState
        self.modelReference = modelReference
        self.releaseHandler = releaseHandler
    }

    func release() async {
        guard claimRelease() else { return }
        await releaseHandler(modelReference)
    }

    deinit {
        guard claimRelease() else { return }
        let modelReference = modelReference
        let releaseHandler = releaseHandler
        Task {
            await releaseHandler(modelReference)
        }
    }

    private func claimRelease() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard !hasReleased else { return false }
        hasReleased = true
        return true
    }
}

struct VoxtralNativeModelConfig: Decodable, Sendable {
    struct Quantization: Decodable, Sendable {
        let bits: Int
        let groupSize: Int

        private enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    struct Multimodal: Decodable, Sendable {
        struct WhisperModelArguments: Decodable, Sendable {
            struct DownsampleArguments: Decodable, Sendable {
                let downsampleFactor: Int

                private enum CodingKeys: String, CodingKey {
                    case downsampleFactor = "downsample_factor"
                }
            }

            struct EncoderArguments: Decodable, Sendable {
                struct AudioEncodingArguments: Decodable, Sendable {
                    let numMelBins: Int

                    private enum CodingKeys: String, CodingKey {
                        case numMelBins = "num_mel_bins"
                    }
                }

                let dimension: Int
                let layerCount: Int
                let headCount: Int
                let headDimension: Int
                let hiddenDimension: Int
                let ropeTheta: Double
                let slidingWindow: Int
                let audioEncodingArguments: AudioEncodingArguments

                private enum CodingKeys: String, CodingKey {
                    case dimension = "dim"
                    case layerCount = "n_layers"
                    case headCount = "n_heads"
                    case headDimension = "head_dim"
                    case hiddenDimension = "hidden_dim"
                    case ropeTheta = "rope_theta"
                    case slidingWindow = "sliding_window"
                    case audioEncodingArguments = "audio_encoding_args"
                }
            }

            let encoderArguments: EncoderArguments
            let downsampleArguments: DownsampleArguments

            private enum CodingKeys: String, CodingKey {
                case encoderArguments = "encoder_args"
                case downsampleArguments = "downsample_args"
            }
        }

        let whisperModelArguments: WhisperModelArguments

        private enum CodingKeys: String, CodingKey {
            case whisperModelArguments = "whisper_model_args"
        }
    }

    let dimension: Int
    let layerCount: Int
    let headCount: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let hiddenDimension: Int
    let ropeTheta: Double
    let slidingWindow: Int
    let vocabularySize: Int
    let conditioningDimension: Int
    let quantization: Quantization?
    let multimodal: Multimodal

    private enum CodingKeys: String, CodingKey {
        case dimension = "dim"
        case layerCount = "n_layers"
        case headCount = "n_heads"
        case keyValueHeadCount = "n_kv_heads"
        case headDimension = "head_dim"
        case hiddenDimension = "hidden_dim"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case vocabularySize = "vocab_size"
        case conditioningDimension = "ada_rms_norm_t_cond_dim"
        case quantization
        case multimodal
    }
}

struct VoxtralNativeBootstrap: Sendable {
    let modelDirectory: URL
    let modelConfig: VoxtralNativeModelConfig
    let tokenizer: VoxtralTekkenTokenizer
    let prompt: VoxtralStreamingPrompt

    var audioConfiguration: VoxtralTekkenSpec.Audio {
        tokenizer.audioConfiguration
    }
}

actor VoxtralNativeRuntime {
    static let shared = VoxtralNativeRuntime()

    private var preparedStates: [String: VoxtralNativePreparedState] = [:]
    private var activeLeases: Set<String> = []
    private var leaseWaiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    func acquirePreparedState(
        modelReference: String = LocalVoxtralConfiguration.modelName,
        autoDownload: Bool = false
    ) async throws -> VoxtralNativePreparedLease {
        while activeLeases.contains(modelReference) {
            await withCheckedContinuation { continuation in
                leaseWaiters[modelReference, default: []].append(continuation)
            }
        }

        activeLeases.insert(modelReference)

        do {
            let preparedState = try await preparedState(
                modelReference: modelReference,
                autoDownload: autoDownload
            )
            return VoxtralNativePreparedLease(
                preparedState: preparedState,
                modelReference: modelReference,
                releaseHandler: { [weak self] modelReference in
                    await self?.releasePreparedState(modelReference: modelReference)
                }
            )
        } catch {
            releasePreparedState(modelReference: modelReference)
            throw error
        }
    }

    func preparedState(
        modelReference: String = LocalVoxtralConfiguration.modelName,
        autoDownload: Bool = false
    ) async throws -> VoxtralNativePreparedState {
        if let preparedState = preparedStates[modelReference] {
            return preparedState
        }

        let bootstrap = try await prepareBootstrap(
            modelReference: modelReference,
            autoDownload: autoDownload
        )
        let preparedState = try VoxtralNativeModelLoader.loadPreparedState(from: bootstrap)
        preparedStates[modelReference] = preparedState
        return preparedState
    }

    func prepareBootstrap(
        modelReference: String = LocalVoxtralConfiguration.modelName,
        autoDownload: Bool = false
    ) async throws -> VoxtralNativeBootstrap {
        let modelDirectory = try await VoxtralNativeModelManager.shared.preparedModelDirectory(
            for: modelReference,
            autoDownload: autoDownload
        )

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(VoxtralNativeModelConfig.self, from: configData)
        let tokenizer = try VoxtralTekkenTokenizer(directoryURL: modelDirectory)
        let prompt = try tokenizer.buildStreamingPrompt()

        return VoxtralNativeBootstrap(
            modelDirectory: modelDirectory,
            modelConfig: config,
            tokenizer: tokenizer,
            prompt: prompt
        )
    }

    func warmupModel(
        modelReference: String = LocalVoxtralConfiguration.modelName,
        autoDownload: Bool = false
    ) async throws -> VoxtralNativeWarmupSummary {
        let preparedState = try await preparedState(
            modelReference: modelReference,
            autoDownload: autoDownload
        )
        return preparedState.summary
    }

    func hasWarmedModel(_ modelReference: String = LocalVoxtralConfiguration.modelName) -> Bool {
        preparedStates[modelReference] != nil
    }

    private func releasePreparedState(modelReference: String) {
        guard activeLeases.remove(modelReference) != nil else { return }

        guard var waiters = leaseWaiters[modelReference], !waiters.isEmpty else {
            leaseWaiters.removeValue(forKey: modelReference)
            return
        }

        let continuation = waiters.removeFirst()
        if waiters.isEmpty {
            leaseWaiters.removeValue(forKey: modelReference)
        } else {
            leaseWaiters[modelReference] = waiters
        }

        continuation.resume()
    }
}
