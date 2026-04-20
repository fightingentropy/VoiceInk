import Foundation
import SwiftData

enum TranscriptionStatus: String, Codable {
    case pending
    case completed
    case failed
}

@Model
final class Transcription {
    var id: UUID
    var text: String
    var timestamp: Date
    var duration: TimeInterval
    var audioFileURL: String?
    var transcriptionModelName: String?
    var transcriptionDuration: TimeInterval?
    var transcriptionStatus: String?

    init(text: String,
         duration: TimeInterval,
         audioFileURL: String? = nil,
         transcriptionModelName: String? = nil,
         transcriptionDuration: TimeInterval? = nil,
         transcriptionStatus: TranscriptionStatus = .pending) {
        self.id = UUID()
        self.text = text
        self.timestamp = Date()
        self.duration = duration
        self.audioFileURL = audioFileURL
        self.transcriptionModelName = transcriptionModelName
        self.transcriptionDuration = transcriptionDuration
        self.transcriptionStatus = transcriptionStatus.rawValue
    }
}
