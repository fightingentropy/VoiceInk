import Foundation
import CoreML
@preconcurrency import WhisperKit
import os

actor WhisperKitRuntime {
    private static let computeOptions = ModelComputeOptions(
        melCompute: .cpuAndGPU,
        audioEncoderCompute: .cpuAndNeuralEngine,
        textDecoderCompute: .cpuAndNeuralEngine,
        prefillCompute: .cpuOnly
    )

    private var pipeline: WhisperKit?
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "WhisperKitRuntime")

    init(modelFolder: String) async throws {
        let config = WhisperKitConfig(
            modelFolder: modelFolder,
            computeOptions: Self.computeOptions,
            verbose: false,
            logLevel: .none,
            prewarm: false,
            load: true,
            download: false,
            useBackgroundDownloadSession: false
        )

        pipeline = try await WhisperKit(config)
    }

    static func prewarmModel(at modelFolder: String) async throws {
        let config = WhisperKitConfig(
            modelFolder: modelFolder,
            computeOptions: computeOptions,
            verbose: false,
            logLevel: .none,
            prewarm: true,
            load: false,
            download: false,
            useBackgroundDownloadSession: false
        )

        _ = try await WhisperKit(config)
    }

    func transcribe(samples: [Float], prompt: String?, language: String?) async throws -> String {
        guard let pipeline else {
            throw VoiceInkEngineError.modelLoadFailed
        }

        var decodingOptions = DecodingOptions(
            task: .transcribe,
            language: language,
            temperature: 0,
            usePrefillPrompt: true,
            usePrefillCache: true,
            detectLanguage: language == nil,
            withoutTimestamps: true,
            wordTimestamps: false
        )

        if let normalizedPrompt = normalizePrompt(prompt),
           let tokenizer = pipeline.tokenizer {
            decodingOptions.promptTokens = tokenizer.encode(text: normalizedPrompt)
            decodingOptions.usePrefillCache = false
        }

        let results = try await pipeline.transcribe(audioArray: samples, decodeOptions: decodingOptions)
        let text = results
            .map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if text.isEmpty {
            logger.error("❌ WhisperKit returned an empty transcription result")
            throw VoiceInkEngineError.transcriptionFailed
        }

        return text
    }

    func unload() async {
        await pipeline?.unloadModels()
        pipeline = nil
    }

    private func normalizePrompt(_ prompt: String?) -> String? {
        guard let prompt else { return nil }
        let normalized = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized.isEmpty ? nil : normalized
    }
}
