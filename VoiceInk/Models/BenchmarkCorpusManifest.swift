import Foundation

struct BenchmarkCorpusManifest: Codable {
    let version: Int
    let generatedAt: Date
    var items: [BenchmarkCorpusManifestItem]

    init(version: Int = 1, generatedAt: Date = Date(), items: [BenchmarkCorpusManifestItem]) {
        self.version = version
        self.generatedAt = generatedAt
        self.items = items
    }
}

struct BenchmarkCorpusManifestItem: Codable, Identifiable {
    let transcriptionID: String
    let capturedAt: Date
    let referenceText: String
    let audioPath: String
    let durationSeconds: TimeInterval?
    let transcriptionModelName: String?

    var id: String { transcriptionID }
}
