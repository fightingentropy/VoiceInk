import Foundation
import SwiftData
import OSLog

final class RecentBenchmarkCorpusService: @unchecked Sendable {
    static let shared = RecentBenchmarkCorpusService()

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "RecentBenchmarkCorpus")
    private let writer = RecentBenchmarkCorpusWriter()

    private init() {}

    @MainActor
    func bootstrapIfNeeded(modelContext: ModelContext) async {
        guard !FileManager.default.fileExists(atPath: AppStoragePaths.recentBenchmarkManifestURL.path) else {
            return
        }

        let descriptor = FetchDescriptor<Transcription>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )

        do {
            let candidates = try modelContext.fetch(descriptor)
                .compactMap(Self.makeCandidate(from:))
            await writer.seedIfEmpty(with: Array(candidates.prefix(RecentBenchmarkCorpusWriter.maxItems)))
        } catch {
            logger.error("Failed to bootstrap benchmark corpus: \(error.localizedDescription, privacy: .public)")
        }
    }

    func captureCompletedRecording(transcription: Transcription, audioURL: URL) {
        guard let candidate = Self.makeCandidate(from: transcription, audioURL: audioURL) else {
            return
        }

        Task {
            await writer.store(candidate)
        }
    }

    private static func makeCandidate(from transcription: Transcription) -> BenchmarkCorpusCandidate? {
        guard let audioFileURL = transcription.audioFileURL,
              let audioURL = URL(string: audioFileURL) else {
            return nil
        }

        return makeCandidate(from: transcription, audioURL: audioURL)
    }

    private static func makeCandidate(from transcription: Transcription, audioURL: URL) -> BenchmarkCorpusCandidate? {
        guard transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue else {
            return nil
        }

        let referenceText = transcription.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !referenceText.isEmpty else {
            return nil
        }

        guard audioURL.isFileURL,
              isEligibleRecordedAudioURL(audioURL),
              FileManager.default.fileExists(atPath: audioURL.path) else {
            return nil
        }

        return BenchmarkCorpusCandidate(
            transcriptionID: transcription.id.uuidString,
            capturedAt: transcription.timestamp,
            referenceText: referenceText,
            sourceAudioURL: audioURL,
            durationSeconds: transcription.duration > 0 ? transcription.duration : nil,
            transcriptionModelName: transcription.transcriptionModelName
        )
    }

    private static func isEligibleRecordedAudioURL(_ url: URL) -> Bool {
        let fileName = url.lastPathComponent.lowercased()
        return !fileName.hasPrefix("transcribed_") && !fileName.hasPrefix("retranscribed_")
    }
}

private struct BenchmarkCorpusCandidate: Sendable {
    let transcriptionID: String
    let capturedAt: Date
    let referenceText: String
    let sourceAudioURL: URL
    let durationSeconds: TimeInterval?
    let transcriptionModelName: String?

    var copiedAudioURL: URL {
        let fileExtension = sourceAudioURL.pathExtension.isEmpty ? "wav" : sourceAudioURL.pathExtension
        return AppStoragePaths.recentBenchmarkCorpusDirectory
            .appendingPathComponent("\(transcriptionID).\(fileExtension)", isDirectory: false)
    }

    var manifestItem: BenchmarkCorpusManifestItem {
        BenchmarkCorpusManifestItem(
            transcriptionID: transcriptionID,
            capturedAt: capturedAt,
            referenceText: referenceText,
            audioPath: copiedAudioURL.path,
            durationSeconds: durationSeconds,
            transcriptionModelName: transcriptionModelName
        )
    }
}

private actor RecentBenchmarkCorpusWriter {
    static let maxItems = 20

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "RecentBenchmarkCorpusWriter")
    private let fileManager = FileManager.default

    func seedIfEmpty(with candidates: [BenchmarkCorpusCandidate]) async {
        guard !candidates.isEmpty else { return }

        let manifest = loadManifest()
        guard manifest.items.isEmpty else { return }

        do {
            try persist(candidates)
        } catch {
            logger.error("Failed to seed benchmark corpus: \(error.localizedDescription, privacy: .public)")
        }
    }

    func store(_ candidate: BenchmarkCorpusCandidate) async {
        var manifest = loadManifest()

        do {
            try ensureCorpusDirectory()
            try copyAudio(from: candidate.sourceAudioURL, to: candidate.copiedAudioURL)

            manifest.items.removeAll { $0.transcriptionID == candidate.transcriptionID }
            manifest.items.insert(candidate.manifestItem, at: 0)
            manifest.items.sort { $0.capturedAt > $1.capturedAt }
            manifest.items = Array(manifest.items.prefix(Self.maxItems))

            try cleanupOrphanedAudioFiles(keeping: Set(manifest.items.map(\.audioPath)))
            try writeManifest(manifest)
        } catch {
            logger.error("Failed to store benchmark recording: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func persist(_ candidates: [BenchmarkCorpusCandidate]) throws {
        try ensureCorpusDirectory()

        let orderedCandidates = Array(candidates.prefix(Self.maxItems))
            .sorted { $0.capturedAt > $1.capturedAt }

        var items: [BenchmarkCorpusManifestItem] = []
        items.reserveCapacity(orderedCandidates.count)

        for candidate in orderedCandidates {
            try copyAudio(from: candidate.sourceAudioURL, to: candidate.copiedAudioURL)
            items.append(candidate.manifestItem)
        }

        try cleanupOrphanedAudioFiles(keeping: Set(items.map(\.audioPath)))
        try writeManifest(BenchmarkCorpusManifest(items: items))
    }

    private func ensureCorpusDirectory() throws {
        try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.recentBenchmarkCorpusDirectory)
    }

    private func copyAudio(from sourceURL: URL, to destinationURL: URL) throws {
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.copyItem(at: sourceURL, to: destinationURL)
    }

    private func cleanupOrphanedAudioFiles(keeping keptAudioPaths: Set<String>) throws {
        guard fileManager.fileExists(atPath: AppStoragePaths.recentBenchmarkCorpusDirectory.path) else {
            return
        }

        let entries = try fileManager.contentsOfDirectory(
            at: AppStoragePaths.recentBenchmarkCorpusDirectory,
            includingPropertiesForKeys: nil
        )

        for entry in entries {
            if entry == AppStoragePaths.recentBenchmarkManifestURL {
                continue
            }

            if !keptAudioPaths.contains(entry.path) {
                try? fileManager.removeItem(at: entry)
            }
        }
    }

    private func writeManifest(_ manifest: BenchmarkCorpusManifest) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let normalized = BenchmarkCorpusManifest(items: manifest.items)
        let data = try encoder.encode(normalized)
        try data.write(to: AppStoragePaths.recentBenchmarkManifestURL, options: .atomic)
    }

    private func loadManifest() -> BenchmarkCorpusManifest {
        guard let data = try? Data(contentsOf: AppStoragePaths.recentBenchmarkManifestURL) else {
            return BenchmarkCorpusManifest(items: [])
        }

        do {
            return try JSONDecoder().decode(BenchmarkCorpusManifest.self, from: data)
        } catch {
            logger.error("Failed to decode benchmark manifest, rebuilding it: \(error.localizedDescription, privacy: .public)")
            return BenchmarkCorpusManifest(items: [])
        }
    }
}
