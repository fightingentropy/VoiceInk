import Foundation
import Testing
@testable import VoiceInk

struct OpenAITranscriptionTests {
    @Test
    func exposesOnlyTheCurrentRealtimeOpenAIModel() {
        let openAIModels = PredefinedModels.models.filter { $0.provider == .openAI }

        #expect(openAIModels.count == 1)
        #expect(openAIModels.first?.name == OpenAIStreamingProvider.modelName)
        #expect(openAIModels.first?.displayName == "GPT Live Transcribe (OpenAI)")
        #expect(
            OpenAIStreamingProvider.webSocketURL.absoluteString
                == "wss://api.openai.com/v1/realtime?intent=transcription"
        )
    }

    @Test
    func realtimeSessionUsesMinimalDelayAndPCM24kHz() throws {
        let message = try OpenAIStreamingProvider.makeSessionUpdate(
            language: "en-GB",
            prompt: "VoiceInk terminology"
        )
        let payload = try #require(
            JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any]
        )
        let session = try #require(payload["session"] as? [String: Any])
        let audio = try #require(session["audio"] as? [String: Any])
        let input = try #require(audio["input"] as? [String: Any])
        let format = try #require(input["format"] as? [String: Any])
        let transcription = try #require(input["transcription"] as? [String: Any])

        #expect(payload["type"] as? String == "session.update")
        #expect(session["type"] as? String == "transcription")
        #expect(format["type"] as? String == "audio/pcm")
        #expect(format["rate"] as? Int == OpenAIStreamingProvider.inputSampleRate)
        #expect(transcription["model"] as? String == "gpt-live-transcribe")
        #expect(transcription["delay"] as? String == "minimal")
        #expect(transcription["languages"] as? [String] == ["en-gb"])
        #expect(transcription["prompt"] as? String == "VoiceInk terminology")
        #expect(input["turn_detection"] is NSNull)
    }

    @Test
    func automaticLanguageLeavesRealtimeDetectionEnabled() throws {
        let message = try OpenAIStreamingProvider.makeSessionUpdate(
            language: "auto",
            prompt: nil
        )
        let payload = try #require(
            JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any]
        )
        let session = try #require(payload["session"] as? [String: Any])
        let audio = try #require(session["audio"] as? [String: Any])
        let input = try #require(audio["input"] as? [String: Any])
        let transcription = try #require(input["transcription"] as? [String: Any])

        #expect(transcription["languages"] == nil)
        #expect(transcription["prompt"] == nil)
    }

    @Test
    func realtimeAudioAppendPreservesPCMBytes() throws {
        let pcm = Data([0x00, 0x01, 0xFE, 0xFF])
        let message = try OpenAIStreamingProvider.makeAudioAppendMessage(pcm)
        let payload = try #require(
            JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any]
        )

        #expect(payload["type"] as? String == "input_audio_buffer.append")
        #expect(
            Data(base64Encoded: try #require(payload["audio"] as? String)) == pcm
        )
    }

    @Test
    func streamingResamplerProducesTheSamePCMWhenChunksSplitSamples() {
        let inputSamples = (0..<16_000).map { index in
            Int16((index % 4_000) - 2_000)
        }
        let pcm = pcm16Data(from: inputSamples)

        var wholeBufferResampler = OpenAIPCM16Resampler()
        var wholeOutput = wholeBufferResampler.append(pcm)
        wholeOutput.append(wholeBufferResampler.flush())

        var chunkedResampler = OpenAIPCM16Resampler()
        var chunkedOutput = Data()
        var offset = 0
        let chunkSizes = [1, 137, 2_047, 3, 4_096, 511]
        var chunkIndex = 0

        while offset < pcm.count {
            let size = min(chunkSizes[chunkIndex % chunkSizes.count], pcm.count - offset)
            chunkedOutput.append(
                chunkedResampler.append(pcm.subdata(in: offset..<(offset + size)))
            )
            offset += size
            chunkIndex += 1
        }
        chunkedOutput.append(chunkedResampler.flush())

        #expect(wholeOutput.count == 24_000 * MemoryLayout<Int16>.size)
        #expect(chunkedOutput == wholeOutput)
    }

    @Test
    func recordedFileFallbackUsesCurrentTranscriptionModelWithoutLegacyFields() {
        let boundary = "OpenAITestBoundary"
        let body = OpenAITranscriptionService.createRequestBody(
            audioData: Data([0x01, 0x02, 0x03]),
            fileName: "recording.wav",
            prompt: "VoiceInk terminology",
            boundary: boundary
        )
        let bodyText = String(decoding: body, as: UTF8.self)

        #expect(bodyText.contains("name=\"model\"\r\n\r\ngpt-transcribe\r\n"))
        #expect(bodyText.contains("name=\"response_format\"\r\n\r\njson\r\n"))
        #expect(bodyText.contains("name=\"prompt\"\r\n\r\nVoiceInk terminology\r\n"))
        #expect(bodyText.contains("filename=\"recording.wav\""))
        #expect(bodyText.contains("Content-Type: audio/wav"))
        #expect(!bodyText.contains("name=\"language\""))
        #expect(!bodyText.contains("name=\"temperature\""))
    }

    private func pcm16Data(from samples: [Int16]) -> Data {
        var data = Data(capacity: samples.count * MemoryLayout<Int16>.size)
        for sample in samples {
            let value = UInt16(bitPattern: sample)
            data.append(UInt8(truncatingIfNeeded: value))
            data.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        return data
    }
}
