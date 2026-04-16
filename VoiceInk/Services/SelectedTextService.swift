import Foundation
import AppKit
import SelectedTextKit
import os

class SelectedTextService {
    private static let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "SelectedTextService")

    static func fetchSelectedText() async -> String? {
        let strategies: [TextStrategy] = [.accessibility, .menuAction]
        do {
            let selectedText = try await SelectedTextManager.shared.getSelectedText(strategies: strategies)
            return selectedText
        } catch {
            logger.error("Failed to get selected text: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }
}
