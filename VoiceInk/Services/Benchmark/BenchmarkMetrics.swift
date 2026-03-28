import Foundation

struct BenchmarkTranscriptMetrics: Sendable {
    let rawWordErrorRate: Double
    let canonicalWordErrorRate: Double
    let canonicalReference: String
    let canonicalHypothesis: String

    var canonicalExactMatch: Bool {
        canonicalReference == canonicalHypothesis
    }
}

func benchmarkTranscriptMetrics(reference: String, hypothesis: String) -> BenchmarkTranscriptMetrics {
    let canonicalReferenceWords = canonicalWords(from: reference)
    let canonicalHypothesisWords = canonicalWords(from: hypothesis)

    let canonicalReference = canonicalReferenceWords.joined(separator: " ")
    let canonicalHypothesis = canonicalHypothesisWords.joined(separator: " ")

    let canonicalWordErrorRate = benchmarkWordErrorRate(
        referenceWords: canonicalReferenceWords,
        hypothesisWords: canonicalHypothesisWords
    )

    return BenchmarkTranscriptMetrics(
        rawWordErrorRate: benchmarkWordErrorRate(reference: reference, hypothesis: hypothesis),
        canonicalWordErrorRate: canonicalWordErrorRate,
        canonicalReference: canonicalReference,
        canonicalHypothesis: canonicalHypothesis
    )
}

private func benchmarkWordErrorRate(reference: String, hypothesis: String) -> Double {
    let referenceWords = normalizedWords(from: reference)
    let hypothesisWords = normalizedWords(from: hypothesis)
    return benchmarkWordErrorRate(referenceWords: referenceWords, hypothesisWords: hypothesisWords)
}

private func benchmarkWordErrorRate(referenceWords: [String], hypothesisWords: [String]) -> Double {
    guard !referenceWords.isEmpty else { return hypothesisWords.isEmpty ? 0 : 1 }

    let distance = levenshteinDistance(referenceWords, hypothesisWords)
    return Double(distance) / Double(referenceWords.count)
}

private func normalizedWords(from text: String) -> [String] {
    text
        .lowercased()
        .replacingOccurrences(of: "[^a-z0-9 ]", with: " ", options: .regularExpression)
        .split(whereSeparator: \.isWhitespace)
        .map(String.init)
}

private func canonicalWords(from text: String) -> [String] {
    let preprocessed = text
        .lowercased()
        .replacingOccurrences(of: "p.m.", with: "pm")
        .replacingOccurrences(of: "a.m.", with: "am")
        .replacingOccurrences(of: "-", with: " ")
        .replacingOccurrences(of: "[^a-z0-9$. ]", with: " ", options: .regularExpression)

    let rawTokens = preprocessed
        .split(whereSeparator: \.isWhitespace)
        .map(String.init)

    var canonicalTokens: [String] = []
    canonicalTokens.reserveCapacity(rawTokens.count)

    var index = 0
    while index < rawTokens.count {
        let token = rawTokens[index]

        if token.hasPrefix("$"), token.count > 1 {
            canonicalTokens.append(String(token.dropFirst()))
            index += 1
            continue
        }

        if token == "dollar" || token == "dollars" {
            index += 1
            continue
        }

        if isNumberWord(token) {
            let slice = Array(rawTokens[index...])
            if let parsed = parseNumberWords(from: slice) {
                canonicalTokens.append(parsed.value)
                index += parsed.consumed
                continue
            }
        }

        canonicalTokens.append(token)
        index += 1
    }

    return canonicalTokens
}

private func levenshteinDistance(_ lhs: [String], _ rhs: [String]) -> Int {
    if lhs.isEmpty { return rhs.count }
    if rhs.isEmpty { return lhs.count }

    var previous = Array(0 ... rhs.count)

    for (lhsIndex, lhsWord) in lhs.enumerated() {
        var current = [lhsIndex + 1]
        current.reserveCapacity(rhs.count + 1)

        for (rhsIndex, rhsWord) in rhs.enumerated() {
            let insertion = current[rhsIndex] + 1
            let deletion = previous[rhsIndex + 1] + 1
            let substitution = previous[rhsIndex] + (lhsWord == rhsWord ? 0 : 1)
            current.append(min(insertion, deletion, substitution))
        }

        previous = current
    }

    return previous[rhs.count]
}

private struct ParsedNumberWords {
    let consumed: Int
    let value: String
}

private let simpleNumberWords: [String: Int] = [
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90
]

private let numberScales: [String: Int] = [
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000
]

private let fractionalNumberWords: [String: Int] = [
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9
]

private func isNumberWord(_ token: String) -> Bool {
    simpleNumberWords[token] != nil || numberScales[token] != nil || token == "and" || token == "point"
}

private func parseNumberWords(from tokens: [String]) -> ParsedNumberWords? {
    var consumed = 0
    var total = 0
    var current = 0
    var fractionalDigits = ""
    var seenNumber = false
    var seenPoint = false

    while consumed < tokens.count {
        let token = tokens[consumed]

        if token == "and", seenNumber, !seenPoint {
            consumed += 1
            continue
        }

        if token == "point", seenNumber, !seenPoint {
            seenPoint = true
            consumed += 1
            continue
        }

        if seenPoint {
            guard let digit = fractionalNumberWords[token] else {
                break
            }
            fractionalDigits.append(String(digit))
            consumed += 1
            continue
        }

        if let baseValue = simpleNumberWords[token] {
            current += baseValue
            seenNumber = true
            consumed += 1
            continue
        }

        if token == "hundred" {
            guard seenNumber else { break }
            current = max(1, current) * 100
            consumed += 1
            continue
        }

        if let scale = numberScales[token], scale >= 1_000 {
            guard seenNumber else { break }
            total += max(1, current) * scale
            current = 0
            consumed += 1
            continue
        }

        break
    }

    guard seenNumber else { return nil }

    let integerValue = total + current
    let normalizedValue = fractionalDigits.isEmpty ? "\(integerValue)" : "\(integerValue).\(fractionalDigits)"
    return ParsedNumberWords(consumed: consumed, value: normalizedValue)
}

extension Array where Element == Double {
    var benchmarkAverage: Double? {
        guard !isEmpty else { return nil }
        return reduce(0, +) / Double(count)
    }
}
