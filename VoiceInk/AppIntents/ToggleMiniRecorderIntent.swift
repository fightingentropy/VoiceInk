import AppIntents
import Foundation
import AppKit

struct ToggleMiniRecorderIntent: AppIntent {
    static let title: LocalizedStringResource = "Toggle VoiceInk Recorder"
    static let description = IntentDescription("Start or stop the VoiceInk mini recorder for voice transcription.")
    
    static let openAppWhenRun: Bool = false
    
    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        NotificationCenter.default.post(name: .toggleMiniRecorder, object: nil)
        
        let dialog = IntentDialog(stringLiteral: "VoiceInk recorder toggled")
        return .result(dialog: dialog)
    }
}

enum IntentError: Error, LocalizedError {
    case appNotAvailable
    case serviceNotAvailable
    
    var errorDescription: String? {
        switch self {
        case .appNotAvailable:
            return "VoiceInk app is not available"
        case .serviceNotAvailable:
            return "VoiceInk recording service is not available"
        }
    }
}
