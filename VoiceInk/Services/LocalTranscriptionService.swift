import Foundation
import AVFoundation
import os

final class LocalTranscriptionService: TranscriptionService, PCMBufferTranscriptionService, @unchecked Sendable {
    private var whisperKitRuntime: WhisperKitRuntime?
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "LocalTranscriptionService")
    private weak var modelProvider: (any LocalModelProvider)?

    init(modelProvider: (any LocalModelProvider)? = nil) {
        self.modelProvider = modelProvider
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard model.provider == .local else {
            throw VoiceInkEngineError.modelLoadFailed
        }

        logger.notice("Initiating local transcription for model: \(model.displayName, privacy: .public)")
        let runtime = try await resolveRuntime(for: model)
        let audioSamples = try readAudioSamples(audioURL)
        let text = try await transcribe(samples: audioSamples, with: runtime)
        await releaseTransientRuntimeIfNeeded(runtime)
        return text
    }

    func transcribe(recordedPCMBuffer: Data, sampleRate: Int, model: any TranscriptionModel) async throws -> String {
        guard model.provider == .local else {
            throw VoiceInkEngineError.modelLoadFailed
        }
        guard sampleRate == 16_000 else {
            logger.error("❌ Unsupported PCM sample rate for local Whisper fast path: \(sampleRate, privacy: .public)")
            throw VoiceInkEngineError.transcriptionFailed
        }

        logger.notice("Initiating local PCM transcription for model: \(model.displayName, privacy: .public)")
        let runtime = try await resolveRuntime(for: model)
        let audioSamples = Self.decodePCM16Mono(recordedPCMBuffer)
        let text = try await transcribe(samples: audioSamples, with: runtime)
        await releaseTransientRuntimeIfNeeded(runtime)
        return text
    }

    private func resolveRuntime(for model: any TranscriptionModel) async throws -> WhisperKitRuntime {
        if let provider = modelProvider {
            let isModelLoaded = await provider.isModelLoaded
            let loadedModel = await provider.loadedLocalModel

            if isModelLoaded,
               let loadedModel,
               loadedModel.name == model.name,
               let loadedRuntime = await provider.whisperKitRuntime {
                logger.notice("Using already loaded model: \(model.name, privacy: .public)")
                whisperKitRuntime = loadedRuntime
                return loadedRuntime
            }
        }

        let availableModels = await modelProvider?.availableModels ?? []
        let resolvedModel = availableModels.first(where: { $0.name == model.name })
        guard let resolvedModel, FileManager.default.fileExists(atPath: resolvedModel.url.path) else {
            logger.error("❌ Model file not found for: \(model.name, privacy: .public)")
            throw VoiceInkEngineError.modelLoadFailed
        }

        logger.notice("Loading model: \(model.name, privacy: .public)")
        do {
            let runtime = try await WhisperKitRuntime(modelFolder: resolvedModel.url.path)
            whisperKitRuntime = runtime
            return runtime
        } catch {
            logger.error("❌ Failed to load model: \(model.name, privacy: .public) - \(error.localizedDescription, privacy: .public)")
            throw VoiceInkEngineError.modelLoadFailed
        }
    }

    private func transcribe(samples: [Float], with runtime: WhisperKitRuntime) async throws -> String {
        let currentPrompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt") ?? ""
        let currentLanguage = selectedWhisperLanguageCode()
        let text = try await runtime.transcribe(
            samples: samples,
            prompt: currentPrompt,
            language: currentLanguage
        )
        logger.notice("WhisperKit transcription completed successfully.")
        return text
    }

    private func releaseTransientRuntimeIfNeeded(_ runtime: WhisperKitRuntime) async {
        let providerRuntime = await modelProvider?.whisperKitRuntime
        if providerRuntime !== runtime {
            await runtime.unload()
            whisperKitRuntime = nil
        }
    }

    private func selectedWhisperLanguageCode() -> String? {
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"
        return selectedLanguage == "auto" ? nil : selectedLanguage
    }

    private func readAudioSamples(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return Self.decodePCM16Mono(Data(data.dropFirst(44)))
    }

    static func decodePCM16Mono(_ pcm: Data) -> [Float] {
        let sampleCount = pcm.count / 2
        guard sampleCount > 0 else { return [] }

        return pcm.withUnsafeBytes { rawBuffer -> [Float] in
            let base = rawBuffer.baseAddress!
            return [Float](unsafeUninitializedCapacity: sampleCount) { buffer, initializedCount in
                let scale: Float = 1.0 / 32767.0
                for i in 0..<sampleCount {
                    // loadUnaligned tolerates Data backings that aren't 2-byte
                    // aligned (e.g. slices of read-from-disk WAV payloads).
                    let raw = base.loadUnaligned(fromByteOffset: i * 2, as: Int16.self)
                    let sample = Float(Int16(littleEndian: raw)) * scale
                    buffer[i] = max(-1.0, min(sample, 1.0))
                }
                initializedCount = sampleCount
            }
        }
    }
}
