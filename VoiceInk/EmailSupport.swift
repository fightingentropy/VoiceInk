import Foundation
import SwiftUI
import AppKit

struct EmailSupport {
    static func generateSupportEmailURL() -> URL? {
        let subject = "VoiceInk Support Request"
        let systemInfo = SystemInfoService.shared.getSystemInfoString()

        let body = """

        ------------------------
        ✨ **SCREEN RECORDING HIGHLY RECOMMENDED** ✨
        ▶️ Create a quick screen recording showing the issue!
        ▶️ It helps me understand and fix the problem much faster.

        📝 ISSUE DETAILS:
        - What steps did you take before the issue occurred?
        - What did you expect to happen?
        - What actually happened instead?


        ## 📋 COMMON ISSUES:
        Check out our Common Issues page before sending an email: https://github.com/fightingentropy/VoiceInk/issues
        ------------------------

        System Information:
        \(systemInfo)


        """
        let encodedBody = body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""

        return URL(string: "https://github.com/fightingentropy/VoiceInk/issues/new?title=\(subject.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")&body=\(encodedBody)")
    }
    
    static func openSupportEmail() {
        if let emailURL = generateSupportEmailURL() {
            NSWorkspace.shared.open(emailURL)
        }
    }
}
