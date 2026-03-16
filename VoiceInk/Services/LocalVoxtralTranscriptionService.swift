import Foundation

enum LocalVoxtralTranscriptionError: LocalizedError {
    case batchFallbackUnavailable

    var errorDescription: String? {
        switch self {
        case .batchFallbackUnavailable:
            return "Local Voxtral requires a running voxmlx realtime server. Start the local server and try again."
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
