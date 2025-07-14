import Foundation

struct AIEnhancementOutputFilter {
    private static let reasoningPatterns = [
        #"(?s)<think>.*?</think>"#,
        #"(?s)<reasoning>.*?</reasoning>"#,
        #"(?s)<analysis>.*?</analysis>"#,
    ]
    
    static func filter(_ text: String) -> String {
        var filteredText = text
        
        for pattern in reasoningPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern) {
                let range = NSRange(filteredText.startIndex..., in: filteredText)
                filteredText = regex.stringByReplacingMatches(in: filteredText, options: [], range: range, withTemplate: "")
            }
        }
        
        filteredText = filteredText.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
        filteredText = filteredText.trimmingCharacters(in: .whitespacesAndNewlines)
        
        return filteredText
    }
} 