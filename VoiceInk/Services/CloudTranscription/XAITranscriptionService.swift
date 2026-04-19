import Foundation
import os

final class XAITranscriptionService: @unchecked Sendable {
    private let endpoint = URL(string: "https://api.x.ai/v1/stt")!
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "XAITranscriptionService")

    func transcribe(
        audioData: Data,
        fileName: String,
        apiKey: String,
        language: String?
    ) async throws -> String {
        guard !audioData.isEmpty else {
            throw CloudTranscriptionError.noTranscriptionReturned
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let body = createRequestBody(
            audioData: audioData,
            fileName: fileName,
            language: normalizeLanguage(language),
            boundary: boundary
        )

        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudTranscriptionError.networkError(URLError(.badServerResponse))
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            logger.error("xAI STT request failed with status \(httpResponse.statusCode): \(message, privacy: .public)")

            if httpResponse.statusCode == 401 {
                throw CloudTranscriptionError.invalidAPIKey
            }

            throw CloudTranscriptionError.apiRequestFailed(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            let transcriptionResponse = try JSONDecoder().decode(ResponsePayload.self, from: data)
            let text = transcriptionResponse.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                throw CloudTranscriptionError.noTranscriptionReturned
            }
            return text
        } catch let error as CloudTranscriptionError {
            throw error
        } catch {
            logger.error("Failed to decode xAI STT response: \(error.localizedDescription, privacy: .public)")
            throw CloudTranscriptionError.noTranscriptionReturned
        }
    }

    private func createRequestBody(
        audioData: Data,
        fileName: String,
        language: String,
        boundary: String
    ) -> Data {
        var body = Data()
        let crlf = "\r\n"

        body.appendFormField(name: "format", value: "true", boundary: boundary)
        body.appendFormField(name: "language", value: language, boundary: boundary)

        // xAI requires the file field to be the final multipart field.
        let safeFileName = sanitizeFileName(fileName)
        body.appendString("--\(boundary)\(crlf)")
        body.appendString("Content-Disposition: form-data; name=\"file\"; filename=\"\(safeFileName)\"\(crlf)")
        body.appendString("Content-Type: \(mimeType(for: safeFileName))\(crlf)\(crlf)")
        body.append(audioData)
        body.appendString(crlf)
        body.appendString("--\(boundary)--\(crlf)")

        return body
    }

    private func normalizeLanguage(_ language: String?) -> String {
        guard let language,
              !language.isEmpty,
              language != "auto" else {
            return "en"
        }

        return language.split(separator: "-").first.map(String.init) ?? "en"
    }

    private func sanitizeFileName(_ fileName: String) -> String {
        fileName
            .replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: "\r", with: "")
            .replacingOccurrences(of: "\n", with: "")
    }

    private func mimeType(for fileName: String) -> String {
        switch URL(fileURLWithPath: fileName).pathExtension.lowercased() {
        case "wav":
            return "audio/wav"
        case "mp3":
            return "audio/mpeg"
        case "webm":
            return "audio/webm"
        case "ogg":
            return "audio/ogg"
        case "opus":
            return "audio/opus"
        case "flac":
            return "audio/flac"
        case "aac":
            return "audio/aac"
        case "m4a", "mp4":
            return "audio/mp4"
        case "mkv":
            return "video/x-matroska"
        default:
            return "application/octet-stream"
        }
    }

    private func decodeErrorMessage(from data: Data) -> String {
        if let payload = try? JSONDecoder().decode(ErrorPayload.self, from: data) {
            if let message = payload.message, !message.isEmpty {
                return message
            }
            if let message = payload.error?.message, !message.isEmpty {
                return message
            }
        }

        return String(data: data, encoding: .utf8) ?? "No error message"
    }

    private struct ResponsePayload: Decodable {
        let text: String
    }

    private struct ErrorPayload: Decodable {
        let message: String?
        let error: ErrorDetail?
    }

    private struct ErrorDetail: Decodable {
        let message: String?
    }
}

private extension Data {
    mutating func appendString(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }

    mutating func appendFormField(name: String, value: String, boundary: String) {
        let crlf = "\r\n"
        appendString("--\(boundary)\(crlf)")
        appendString("Content-Disposition: form-data; name=\"\(name)\"\(crlf)\(crlf)")
        appendString(value)
        appendString(crlf)
    }
}
