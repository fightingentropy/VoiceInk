import Foundation

struct AIEnhancementOutputFilter {
    static func filter(_ text: String) -> String {
        let patterns = [
            #"(?s)<thinking>(.*?)</thinking>"#,
            #"(?s)<think>(.*?)</think>"#,
            #"(?s)<reasoning>(.*?)</reasoning>"#
        ]

        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern) {
                let range = NSRange(text.startIndex..., in: text)
                if let match = regex.firstMatch(in: text, options: [], range: range) {
                    // Extract content from the first capturing group
                    if match.numberOfRanges > 1, let contentRange = Range(match.range(at: 1), in: text) {
                        let extractedText = String(text[contentRange])
                        return extractedText.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                }
            }
        }
        
        // If no tags are found, return the original text as is.
        return text
    }
} 