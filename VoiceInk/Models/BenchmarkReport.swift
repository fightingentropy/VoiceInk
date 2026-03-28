import Foundation

enum BenchmarkCorpusSource: String, Codable, CaseIterable, Identifiable, Sendable {
    case standard
    case recentRecordings

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .standard:
            return "Standard Corpus"
        case .recentRecordings:
            return "Recent Recordings"
        }
    }

    var description: String {
        switch self {
        case .standard:
            return "Stable synthetic phrases for consistent model-to-model comparisons."
        case .recentRecordings:
            return "Your most recent completed recordings captured by VoiceInk."
        }
    }
}

enum BenchmarkModelStatus: String, Codable, Sendable {
    case ok
    case unavailable
    case failed
}

struct BenchmarkDeviceSnapshot: Codable, Sendable {
    let hardwareModel: String
    let operatingSystemVersion: String
    let appVersion: String
    let appBuild: String
}

struct BenchmarkRunReport: Codable, Identifiable, Sendable {
    let id: String
    let generatedAt: Date
    let corpusSource: BenchmarkCorpusSource
    let device: BenchmarkDeviceSnapshot
    let corpus: [BenchmarkCorpusSummary]
    let results: [BenchmarkModelResult]
}

struct BenchmarkCorpusSummary: Codable, Identifiable, Sendable {
    let id: String
    let referenceText: String
    let audioSeconds: TimeInterval
}

struct BenchmarkModelResult: Codable, Identifiable, Sendable {
    let modelName: String
    let displayName: String
    let providerName: String
    let status: BenchmarkModelStatus
    let detail: String?
    let averageElapsedSeconds: TimeInterval?
    let averageRealtimeFactor: Double?
    let averageWordErrorRate: Double?
    let averageCanonicalWordErrorRate: Double?
    let averageCanonicalExactMatchRate: Double?
    let samples: [BenchmarkSampleResult]

    var id: String { modelName }
}

struct BenchmarkSampleResult: Codable, Identifiable, Sendable {
    let id: String
    let referenceText: String
    let transcript: String
    let canonicalReferenceText: String
    let canonicalTranscript: String
    let elapsedSeconds: TimeInterval
    let realtimeFactor: Double
    let wordErrorRate: Double
    let canonicalWordErrorRate: Double
    let canonicalExactMatchRate: Double
}
