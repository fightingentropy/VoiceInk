import Foundation
import os

final class OpenAITranscriptionService: @unchecked Sendable {
    static let fallbackModel = "gpt-transcribe"

    private let endpoint = URL(string: "https://api.openai.com/v1/audio/transcriptions")!
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "OpenAITranscriptionService")

    func transcribe(
        audioData: Data,
        fileName: String,
        apiKey: String,
        prompt: String?
    ) async throws -> String {
        guard !audioData.isEmpty else {
            throw CloudTranscriptionError.noTranscriptionReturned
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let body = Self.createRequestBody(
            audioData: audioData,
            fileName: fileName,
            prompt: prompt,
            boundary: boundary
        )

        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudTranscriptionError.networkError(URLError(.badServerResponse))
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            logger.error("OpenAI transcription request failed with status \(httpResponse.statusCode): \(message, privacy: .public)")

            if httpResponse.statusCode == 401 {
                throw CloudTranscriptionError.invalidAPIKey
            }

            throw CloudTranscriptionError.apiRequestFailed(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            let response = try JSONDecoder().decode(ResponsePayload.self, from: data)
            let text = response.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                throw CloudTranscriptionError.noTranscriptionReturned
            }
            return text
        } catch let error as CloudTranscriptionError {
            throw error
        } catch {
            logger.error("Failed to decode OpenAI transcription response: \(error.localizedDescription, privacy: .public)")
            throw CloudTranscriptionError.noTranscriptionReturned
        }
    }

    static func createRequestBody(
        audioData: Data,
        fileName: String,
        prompt: String?,
        boundary: String
    ) -> Data {
        var body = Data()
        let crlf = "\r\n"

        body.appendOpenAIFormField(name: "model", value: fallbackModel, boundary: boundary)
        body.appendOpenAIFormField(name: "response_format", value: "json", boundary: boundary)

        if let prompt, !prompt.isEmpty {
            body.appendOpenAIFormField(name: "prompt", value: prompt, boundary: boundary)
        }

        let safeFileName = fileName
            .replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: "\r", with: "")
            .replacingOccurrences(of: "\n", with: "")

        body.appendOpenAIString("--\(boundary)\(crlf)")
        body.appendOpenAIString("Content-Disposition: form-data; name=\"file\"; filename=\"\(safeFileName)\"\(crlf)")
        body.appendOpenAIString("Content-Type: \(mimeType(for: safeFileName))\(crlf)\(crlf)")
        body.append(audioData)
        body.appendOpenAIString(crlf)
        body.appendOpenAIString("--\(boundary)--\(crlf)")

        return body
    }

    private static func mimeType(for fileName: String) -> String {
        switch URL(fileURLWithPath: fileName).pathExtension.lowercased() {
        case "wav":
            return "audio/wav"
        case "mp3":
            return "audio/mpeg"
        case "webm":
            return "audio/webm"
        case "ogg":
            return "audio/ogg"
        case "flac":
            return "audio/flac"
        case "m4a", "mp4":
            return "audio/mp4"
        default:
            return "application/octet-stream"
        }
    }

    private func decodeErrorMessage(from data: Data) -> String {
        if let payload = try? JSONDecoder().decode(ErrorPayload.self, from: data),
           let message = payload.error?.message,
           !message.isEmpty {
            return message
        }

        return String(data: data, encoding: .utf8) ?? "No error message"
    }

    private struct ResponsePayload: Decodable {
        let text: String
    }

    private struct ErrorPayload: Decodable {
        let error: ErrorDetail?
    }

    private struct ErrorDetail: Decodable {
        let message: String?
    }
}

private extension Data {
    mutating func appendOpenAIString(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }

    mutating func appendOpenAIFormField(name: String, value: String, boundary: String) {
        let crlf = "\r\n"
        appendOpenAIString("--\(boundary)\(crlf)")
        appendOpenAIString("Content-Disposition: form-data; name=\"\(name)\"\(crlf)\(crlf)")
        appendOpenAIString(value)
        appendOpenAIString(crlf)
    }
}
