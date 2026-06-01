import Foundation
@preconcurrency import AVFoundation
import os

final class AudioProcessor: @unchecked Sendable {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "AudioProcessor")
    
    struct AudioFormat {
        static let targetSampleRate: Double = 16000.0
        static let targetChannels: UInt32 = 1
        static let targetBitDepth: UInt32 = 16
    }
    
    enum AudioProcessingError: LocalizedError {
        case invalidAudioFile
        case conversionFailed
        case exportFailed
        case unsupportedFormat
        case sampleExtractionFailed
        
        var errorDescription: String? {
            switch self {
            case .invalidAudioFile:
                return "The audio file is invalid or corrupted"
            case .conversionFailed:
                return "Failed to convert the audio format"
            case .exportFailed:
                return "Failed to export the processed audio"
            case .unsupportedFormat:
                return "The audio format is not supported"
            case .sampleExtractionFailed:
                return "Failed to extract audio samples"
            }
        }
    }
    
    func processAudioToSamples(_ url: URL) async throws -> [Float] {
        guard let audioFile = try? AVAudioFile(forReading: url) else {
            throw AudioProcessingError.invalidAudioFile
        }
        
        let format = audioFile.processingFormat
        let sampleRate = format.sampleRate
        let channels = format.channelCount
        let totalFrames = audioFile.length
        
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioFormat.targetSampleRate,
            channels: AudioFormat.targetChannels,
            interleaved: false
        )
        
        guard let outputFormat = outputFormat else {
            throw AudioProcessingError.unsupportedFormat
        }
        
        let chunkSize: AVAudioFrameCount = 50_000_000
        var allSamples: [Float] = []
        let estimatedOutputFrames = Int((Double(totalFrames) / sampleRate) * AudioFormat.targetSampleRate)
        allSamples.reserveCapacity(max(0, estimatedOutputFrames))
        var currentFrame: AVAudioFramePosition = 0
        
        while currentFrame < totalFrames {
            let remainingFrames = totalFrames - currentFrame
            let framesToRead = min(chunkSize, AVAudioFrameCount(remainingFrames))
            
            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRead) else {
                throw AudioProcessingError.conversionFailed
            }
            
            audioFile.framePosition = currentFrame
            try audioFile.read(into: inputBuffer, frameCount: framesToRead)
            
            if sampleRate == AudioFormat.targetSampleRate && channels == AudioFormat.targetChannels {
                let chunkSamples = convertToWhisperFormat(inputBuffer)
                allSamples.append(contentsOf: chunkSamples)
            } else {
                guard let converter = AVAudioConverter(from: format, to: outputFormat) else {
                    throw AudioProcessingError.conversionFailed
                }
                
                let ratio = AudioFormat.targetSampleRate / sampleRate
                let outputFrameCount = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio)
                
                guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
                    throw AudioProcessingError.conversionFailed
                }
                
                var error: NSError?
                let status = converter.convert(
                    to: outputBuffer,
                    error: &error,
                    withInputFrom: { inNumPackets, outStatus in
                        outStatus.pointee = .haveData
                        return inputBuffer
                    }
                )
                
                if error != nil { throw AudioProcessingError.conversionFailed }
                
                if status == .error {
                    throw AudioProcessingError.conversionFailed
                }
                
                let chunkSamples = convertToWhisperFormat(outputBuffer)
                allSamples.append(contentsOf: chunkSamples)
            }
            
            currentFrame += AVAudioFramePosition(framesToRead)
        }
        
        return allSamples
    }
    
    private func convertToWhisperFormat(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return []
        }
        
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)
        var samples = Array(repeating: Float(0), count: frameLength)
        var maxSample: Float = 0
        
        if channelCount == 1 {
            let source = channelData[0]
            for frame in 0..<frameLength {
                let sample = source[frame]
                samples[frame] = sample
                maxSample = max(maxSample, abs(sample))
            }
        } else {
            for frame in 0..<frameLength {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                let sample = sum / Float(channelCount)
                samples[frame] = sample
                maxSample = max(maxSample, abs(sample))
            }
        }
        
        if maxSample > 0 {
            let scale = 1 / maxSample
            for index in samples.indices {
                samples[index] *= scale
            }
        }
        
        return samples
    }
    func saveSamplesAsWav(samples: [Float], to url: URL) throws {
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: AudioFormat.targetSampleRate,
            channels: AudioFormat.targetChannels,
            interleaved: true
        )

        guard let outputFormat = outputFormat else {
            throw AudioProcessingError.unsupportedFormat
        }

        let buffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        )
        
        guard let buffer = buffer else {
            throw AudioProcessingError.conversionFailed
        }
        
        guard let channelData = buffer.int16ChannelData?[0] else {
            throw AudioProcessingError.conversionFailed
        }

        for index in samples.indices {
            channelData[index] = Int16(max(-1.0, min(1.0, samples[index])) * Float(Int16.max))
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)

        // Create audio file
        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatInt16,
            interleaved: true
        )

        try audioFile.write(from: buffer)
    }
} 
