import Foundation
import SwiftData

final class WordReplacementService: @unchecked Sendable {
    static let shared = WordReplacementService()

    private init() {}

    func applyReplacements(to text: String, using context: ModelContext) -> String {
        let descriptor = FetchDescriptor<WordReplacement>(
            predicate: #Predicate { $0.isEnabled }
        )

        guard let replacements = try? context.fetch(descriptor), !replacements.isEmpty else {
            return text // No replacements to apply
        }

        var modifiedText = text

        // Apply replacements (case-insensitive)
        for replacement in replacements {
            let originalGroup = replacement.originalText
            let replacementText = replacement.replacementText

            // Split comma-separated originals at apply time only
            let variants = originalGroup
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            for original in variants {
                modifiedText = Self.replacing(
                    original,
                    with: replacementText,
                    in: modifiedText
                )
            }
        }

        return modifiedText
    }

    static func replacing(_ original: String, with replacement: String, in text: String) -> String {
        guard usesWordBoundaries(for: original) else {
            return text.replacingOccurrences(
                of: original,
                with: replacement,
                options: .caseInsensitive
            )
        }

        let pattern = "\\b\(NSRegularExpression.escapedPattern(for: original))\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return text
        }

        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(
            in: text,
            options: [],
            range: range,
            withTemplate: NSRegularExpression.escapedTemplate(for: replacement)
        )
    }

    private static func usesWordBoundaries(for text: String) -> Bool {
        // Returns false for languages without spaces (CJK, Thai), true for spaced languages
        let nonSpacedScripts: [ClosedRange<UInt32>] = [
            0x3040...0x309F, // Hiragana
            0x30A0...0x30FF, // Katakana
            0x4E00...0x9FFF, // CJK Unified Ideographs
            0xAC00...0xD7AF, // Hangul Syllables
            0x0E00...0x0E7F, // Thai
        ]

        for scalar in text.unicodeScalars {
            for range in nonSpacedScripts {
                if range.contains(scalar.value) {
                    return false
                }
            }
        }

        return true
    }
}
