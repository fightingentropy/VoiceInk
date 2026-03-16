import Foundation
import AVFoundation
@preconcurrency import MLX

enum VoxtralNativeAudio {
    static let sampleRate = 16_000
    static let fftSize = 400
    static let hopLength = 160
    static let melBinCount = 128
    static let samplesPerToken = hopLength * 2 * 4
    static let globalLogMelMax: Float = 1.5
    static let rightPadTokenCount = 17

    private nonisolated(unsafe) static let melFilters: MLXArray = {
        let filters = makeMelFilterBank()
        return MLXArray(filters.flatMap { $0 }, [filters.count, filters.first?.count ?? 0])
    }()

    private nonisolated(unsafe) static let window: MLXArray = {
        let samples = (0 ..< fftSize).map { index -> Float in
            let angle = (2.0 * Float.pi * Float(index)) / Float(fftSize)
            return 0.5 - 0.5 * cos(angle)
        }
        return MLXArray(samples)
    }()

    private nonisolated(unsafe) static let dftReal: MLXArray = {
        let matrix = makeDFT(real: true)
        return MLXArray(matrix.flatMap { $0 }, [matrix.count, matrix.first?.count ?? 0])
    }()

    private nonisolated(unsafe) static let dftImag: MLXArray = {
        let matrix = makeDFT(real: false)
        return MLXArray(matrix.flatMap { $0 }, [matrix.count, matrix.first?.count ?? 0])
    }()

    static func decodePCM16(_ data: Data) -> [Float] {
        let sampleCount = data.count / MemoryLayout<Int16>.size
        return data.withUnsafeBytes { rawBuffer in
            let samples = rawBuffer.bindMemory(to: Int16.self)
            return (0 ..< sampleCount).map { index in
                Float(samples[index]) / 32767.0
            }
        }
    }

    static func zeroSamples(count: Int) -> [Float] {
        Array(repeating: 0, count: count)
    }

    static func padAudio(
        _ audio: [Float],
        leftPadTokenCount: Int = 32,
        rightPadTokenCount: Int = VoxtralNativeAudio.rightPadTokenCount
    ) -> [Float] {
        let leftPadSampleCount = leftPadTokenCount * samplesPerToken
        let rightAlign = (samplesPerToken - (audio.count % samplesPerToken)) % samplesPerToken
        let rightPadSampleCount = rightAlign + (rightPadTokenCount * samplesPerToken)
        return zeroSamples(count: leftPadSampleCount) + audio + zeroSamples(count: rightPadSampleCount)
    }

    static func loadAudioFile(_ url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let inputFormat = audioFile.processingFormat
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        )

        guard let outputFormat else {
            throw VoxtralNativeAudioError.unsupportedFormat
        }

        let inputFrameCount = AVAudioFrameCount(audioFile.length)
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: inputFrameCount
        ) else {
            throw VoxtralNativeAudioError.bufferAllocationFailed
        }

        try audioFile.read(into: inputBuffer)

        let convertedBuffer: AVAudioPCMBuffer
        if inputFormat.sampleRate == Double(sampleRate), inputFormat.channelCount == 1 {
            convertedBuffer = inputBuffer
        } else {
            guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
                throw VoxtralNativeAudioError.conversionFailed
            }

            let ratio = Double(sampleRate) / inputFormat.sampleRate
            let outputFrameCapacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio) + 1
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputFrameCapacity
            ) else {
                throw VoxtralNativeAudioError.bufferAllocationFailed
            }

            var conversionError: NSError?
            var consumedInput = false
            let status = converter.convert(
                to: outputBuffer,
                error: &conversionError
            ) { _, outStatus in
                if consumedInput {
                    outStatus.pointee = .endOfStream
                    return nil
                }

                consumedInput = true
                outStatus.pointee = .haveData
                return inputBuffer
            }

            if let conversionError {
                throw conversionError
            }

            guard status != .error else {
                throw VoxtralNativeAudioError.conversionFailed
            }

            convertedBuffer = outputBuffer
        }

        guard let channelData = convertedBuffer.floatChannelData else {
            throw VoxtralNativeAudioError.missingChannelData
        }

        let frameLength = Int(convertedBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
    }

    static func logMelSpectrogramStep(
        audioChunk: [Float],
        audioTail: [Float]?
    ) -> (mel: MLXArray, audioTail: [Float]) {
        let tailLength = fftSize - hopLength
        let combined: [Float]
        if let audioTail {
            combined = audioTail + audioChunk
        } else {
            combined = zeroSamples(count: fftSize / 2) + audioChunk
        }

        let nextTail = Array(combined.suffix(tailLength))
        let frameCount = 1 + (combined.count - fftSize) / hopLength
        guard frameCount > 0 else {
            return (MLXArray.zeros([melBinCount, 0], dtype: .float32), nextTail)
        }

        let audioArray = MLXArray(combined)
        let frames = asStrided(audioArray, [frameCount, fftSize], strides: [hopLength, 1]) *
            window.reshaped([1, fftSize])
        let specReal = frames.matmul(dftReal.transposed())
        let specImag = frames.matmul(dftImag.transposed())
        let magnitudes = specReal * specReal + specImag * specImag

        let melSpectrum = magnitudes.matmul(melFilters.transposed())
        var logSpectrum = log10(maximum(melSpectrum, 1e-10))
        logSpectrum = maximum(logSpectrum, globalLogMelMax - 8.0)
        logSpectrum = (logSpectrum + 4.0) / 4.0
        return (logSpectrum.transposed(), nextTail)
    }

    static func logMelSpectrogram(_ audio: [Float]) -> MLXArray {
        let padLength = fftSize / 2
        let paddedAudio = zeroSamples(count: padLength) + audio + zeroSamples(count: padLength)
        let frameCount = 1 + (paddedAudio.count - fftSize) / hopLength
        guard frameCount > 1 else {
            return MLXArray.zeros([melBinCount, 0], dtype: .float32)
        }

        let audioArray = MLXArray(paddedAudio)
        let frames = asStrided(audioArray, [frameCount, fftSize], strides: [hopLength, 1]) *
            window.reshaped([1, fftSize])
        let specReal = frames.matmul(dftReal.transposed())
        let specImag = frames.matmul(dftImag.transposed())
        let magnitudes = (specReal * specReal + specImag * specImag)[0 ..< (frameCount - 1), axis: 0]

        let melSpectrum = magnitudes.matmul(melFilters.transposed())
        var logSpectrum = log10(maximum(melSpectrum, 1e-10))
        logSpectrum = maximum(logSpectrum, globalLogMelMax - 8.0)
        logSpectrum = (logSpectrum + 4.0) / 4.0
        return logSpectrum.transposed()
    }

    private static func makeDFT(real: Bool) -> [[Float]] {
        let frequencyBins = fftSize / 2 + 1
        return (0 ..< frequencyBins).map { frequencyIndex in
            (0 ..< fftSize).map { sampleIndex in
                let angle = -2.0 * Float.pi * Float(frequencyIndex * sampleIndex) / Float(fftSize)
                return real ? cos(angle) : sin(angle)
            }
        }
    }

    private static func makeMelFilterBank() -> [[Float]] {
        let frequencyBins = fftSize / 2 + 1
        let fftFrequencies = (0 ..< frequencyBins).map { index in
            Float(index) * Float(sampleRate) / Float(fftSize)
        }
        let melMin = hzToMel(0)
        let melMax = hzToMel(8_000)
        let melFrequencies = (0 ..< (melBinCount + 2)).map { index in
            melMin + (Float(index) / Float(melBinCount + 1)) * (melMax - melMin)
        }
        let filterFrequencies = melFrequencies.map(melToHz)
        let filterDiffs = zip(filterFrequencies.dropFirst(), filterFrequencies).map { next, current in
            next - current
        }

        return (0 ..< melBinCount).map { melIndex in
            let lowerDiff = filterDiffs[melIndex]
            let upperDiff = filterDiffs[melIndex + 1]
            let normalization = 2.0 / (filterFrequencies[melIndex + 2] - filterFrequencies[melIndex])

            return fftFrequencies.map { frequency in
                let lower = (frequency - filterFrequencies[melIndex]) / lowerDiff
                let upper = (filterFrequencies[melIndex + 2] - frequency) / upperDiff
                return max(0, min(lower, upper)) * normalization
            }
        }
    }

    private static func hzToMel(_ frequency: Float) -> Float {
        let minimumLogHz: Float = 1_000
        let minimumLogMel: Float = 15
        let logStep = 27.0 / Float(log(6.4))
        if frequency >= minimumLogHz {
            return minimumLogMel + Float(log(Double(frequency / minimumLogHz))) * logStep
        }
        return 3.0 * frequency / 200.0
    }

    private static func melToHz(_ mel: Float) -> Float {
        let minimumLogHz: Float = 1_000
        let minimumLogMel: Float = 15
        let logStep = Float(log(6.4)) / 27.0
        if mel >= minimumLogMel {
            return minimumLogHz * Float(exp(Double(logStep * (mel - minimumLogMel))))
        }
        return 200.0 * mel / 3.0
    }
}

enum VoxtralNativeAudioError: LocalizedError {
    case unsupportedFormat
    case bufferAllocationFailed
    case conversionFailed
    case missingChannelData

    var errorDescription: String? {
        switch self {
        case .unsupportedFormat:
            return "The audio format is not supported for native Voxtral preprocessing."
        case .bufferAllocationFailed:
            return "Failed to allocate an audio buffer for native Voxtral preprocessing."
        case .conversionFailed:
            return "Failed to convert the audio file into Voxtral's native sample format."
        case .missingChannelData:
            return "The converted audio buffer did not expose float channel data."
        }
    }
}
