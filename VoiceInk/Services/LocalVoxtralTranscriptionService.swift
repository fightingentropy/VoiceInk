import Foundation

enum LocalVoxtralTranscriptionError: LocalizedError {
    case batchFallbackUnavailable

    var errorDescription: String? {
        switch self {
        case .batchFallbackUnavailable:
            return "Local Voxtral is streaming-only. If native startup fails, there is no batch fallback available for this model."
        }
    }
}

final class LocalVoxtralTranscriptionService: TranscriptionService, @unchecked Sendable {
    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        _ = audioURL
        _ = model
        throw LocalVoxtralTranscriptionError.batchFallbackUnavailable
    }
}
