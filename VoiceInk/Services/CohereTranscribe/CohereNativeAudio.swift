import Foundation
import AVFoundation
import MLX

final class CohereNativeFeatureExtractor: @unchecked Sendable {
    let configuration: CohereNativeAudioConfiguration

    private let melFilters: MLXArray
    private let window: MLXArray
    private let dftReal: MLXArray
    private let dftImag: MLXArray

    init(configuration: CohereNativeAudioConfiguration) {
        self.configuration = configuration
        self.melFilters = MLXArray(
            Self.makeMelFilterBank(configuration: configuration).flatMap { $0 },
            [
                configuration.melBinCount,
                (configuration.fftSize / 2) + 1
            ]
        )
        self.window = MLXArray(Self.makeWindow(configuration: configuration))
        self.dftReal = MLXArray(
            Self.makeDFT(configuration: configuration, real: true).flatMap { $0 },
            [
                (configuration.fftSize / 2) + 1,
                configuration.fftSize
            ]
        )
        self.dftImag = MLXArray(
            Self.makeDFT(configuration: configuration, real: false).flatMap { $0 },
            [
                (configuration.fftSize / 2) + 1,
                configuration.fftSize
            ]
        )
    }

    func extractLogMelFeatures(from audio: [Float]) -> MLXArray {
        let ditheredAudio = configuration.dither > 0 ? dithered(audio) : audio
        let emphasizedAudio = applyPreemphasis(to: ditheredAudio)
        let padLength = configuration.fftSize / 2

        var paddedAudio = Self.zeroSamples(count: padLength)
        paddedAudio.append(contentsOf: emphasizedAudio)
        paddedAudio.append(contentsOf: Self.zeroSamples(count: padLength))

        let frameCount = 1 + (paddedAudio.count - configuration.fftSize) / configuration.hopLength
        guard frameCount > 0 else {
            return MLXArray.zeros([configuration.melBinCount, 0], dtype: .float32)
        }

        let audioArray = MLXArray(paddedAudio)
        let frames = asStrided(
            audioArray,
            [frameCount, configuration.fftSize],
            strides: [configuration.hopLength, 1]
        ) * window.reshaped([1, configuration.fftSize])

        let specReal = frames.matmul(dftReal.transposed())
        let specImag = frames.matmul(dftImag.transposed())
        let magnitudes = specReal * specReal + specImag * specImag
        let melSpectrum = magnitudes.matmul(melFilters.transposed())
        var logSpectrum = log(melSpectrum + Self.logZeroGuardValue)
        logSpectrum = logSpectrum.transposed()

        if configuration.normalizePerFeature {
            let mean = MLX.mean(logSpectrum, axis: 1, keepDims: true)
            let centered = logSpectrum - mean
            let frameCount = max(1, logSpectrum.dim(1))
            let varianceDenominator = max(1, frameCount - 1)
            let variance = MLX.sum(centered * centered, axis: 1, keepDims: true)
                / Float(varianceDenominator)
            let standardDeviation = MLX.sqrt(variance)
                + Self.normalizationEpsilon
            logSpectrum = (logSpectrum - mean) / standardDeviation
        }

        return logSpectrum
    }

    func extractLogMelFeatures(fromAudioFile url: URL) throws -> MLXArray {
        try extractLogMelFeatures(from: Self.loadAudioFile(url, sampleRate: configuration.sampleRate))
    }

    static func loadAudioFile(_ url: URL, sampleRate: Int) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let inputFormat = audioFile.processingFormat
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw CohereNativeAudioError.unsupportedFormat
        }

        let inputFrameCount = AVAudioFrameCount(audioFile.length)
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: inputFrameCount
        ) else {
            throw CohereNativeAudioError.bufferAllocationFailed
        }

        try audioFile.read(into: inputBuffer)

        let convertedBuffer: AVAudioPCMBuffer
        if inputFormat.sampleRate == Double(sampleRate), inputFormat.channelCount == 1 {
            convertedBuffer = inputBuffer
        } else {
            guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
                throw CohereNativeAudioError.conversionFailed
            }

            let ratio = Double(sampleRate) / inputFormat.sampleRate
            let outputFrameCapacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio) + 1
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputFrameCapacity
            ) else {
                throw CohereNativeAudioError.bufferAllocationFailed
            }

            var conversionError: NSError?
            var consumedInput = false
            let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
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
                throw CohereNativeAudioError.conversionFailed
            }

            convertedBuffer = outputBuffer
        }

        guard let channelData = convertedBuffer.floatChannelData else {
            throw CohereNativeAudioError.missingChannelData
        }

        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(convertedBuffer.frameLength)))
    }

    private func dithered(_ audio: [Float]) -> [Float] {
        var generator = SeededGenerator(seed: UInt64(audio.count))
        return audio.map { sample in
            let noise = Self.nextGaussian(using: &generator) * configuration.dither
            return sample + noise
        }
    }

    private func applyPreemphasis(to audio: [Float]) -> [Float] {
        guard !audio.isEmpty else { return [] }

        var emphasized = audio
        for index in stride(from: audio.count - 1, through: 1, by: -1) {
            emphasized[index] = audio[index] - (Self.preemphasisCoefficient * audio[index - 1])
        }
        return emphasized
    }

    private static func zeroSamples(count: Int) -> [Float] {
        Array(repeating: 0, count: count)
    }

    private static func makeWindow(configuration: CohereNativeAudioConfiguration) -> [Float] {
        guard configuration.windowSize > 0 else {
            return zeroSamples(count: configuration.fftSize)
        }

        let hannWindow: [Float]
        if configuration.windowSize == 1 {
            hannWindow = [1]
        } else {
            hannWindow = (0 ..< configuration.windowSize).map { index -> Float in
                let angle = (2.0 * Float.pi * Float(index)) / Float(configuration.windowSize - 1)
                return 0.5 - 0.5 * cos(angle)
            }
        }

        guard configuration.windowSize < configuration.fftSize else {
            return hannWindow
        }

        let totalPadding = configuration.fftSize - configuration.windowSize
        let leftPadding = totalPadding / 2
        let rightPadding = totalPadding - leftPadding
        return zeroSamples(count: leftPadding) + hannWindow + zeroSamples(count: rightPadding)
    }

    private static func makeDFT(configuration: CohereNativeAudioConfiguration, real: Bool) -> [[Float]] {
        let frequencyBins = (configuration.fftSize / 2) + 1
        return (0 ..< frequencyBins).map { frequencyIndex in
            (0 ..< configuration.fftSize).map { sampleIndex in
                let angle = -2.0 * Float.pi * Float(frequencyIndex * sampleIndex) / Float(configuration.fftSize)
                return real ? cos(angle) : sin(angle)
            }
        }
    }

    private static func makeMelFilterBank(configuration: CohereNativeAudioConfiguration) -> [[Float]] {
        let frequencyBins = (configuration.fftSize / 2) + 1
        let fftFrequencies = (0 ..< frequencyBins).map { index in
            Float(index) * Float(configuration.sampleRate) / Float(configuration.fftSize)
        }

        let melMin = hzToMel(0)
        let melMax = hzToMel(Float(configuration.sampleRate) / 2)
        let melFrequencies = (0 ..< (configuration.melBinCount + 2)).map { index in
            melMin + (Float(index) / Float(configuration.melBinCount + 1)) * (melMax - melMin)
        }
        let filterFrequencies = melFrequencies.map(melToHz)
        let filterDiffs = zip(filterFrequencies.dropFirst(), filterFrequencies).map { next, current in
            next - current
        }

        return (0 ..< configuration.melBinCount).map { melIndex in
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

    private static func nextGaussian(using generator: inout SeededGenerator) -> Float {
        let lowerBound = Float.leastNonzeroMagnitude
        let u1 = max(lowerBound, Float.random(in: 0 ... 1, using: &generator))
        let u2 = Float.random(in: 0 ..< 1, using: &generator)
        let radius = sqrt(-2 * log(u1))
        let angle = 2 * Float.pi * u2
        return radius * cos(angle)
    }

    private static let preemphasisCoefficient: Float = 0.97
    private static let logZeroGuardValue: Float = 5.960_464_5e-8
    private static let normalizationEpsilon: Float = 1e-5
}

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1
        return state
    }
}

enum CohereNativeAudioError: LocalizedError {
    case unsupportedFormat
    case bufferAllocationFailed
    case conversionFailed
    case missingChannelData

    var errorDescription: String? {
        switch self {
        case .unsupportedFormat:
            return "The audio format is not supported for the experimental native Cohere MLX path."
        case .bufferAllocationFailed:
            return "Failed to allocate audio buffers for the experimental native Cohere MLX path."
        case .conversionFailed:
            return "Failed to convert audio for the experimental native Cohere MLX path."
        case .missingChannelData:
            return "The audio buffer did not contain channel data for the experimental native Cohere MLX path."
        }
    }
}
