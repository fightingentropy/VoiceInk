import AVFoundation
import Foundation

enum BenchmarkCorpusError: LocalizedError {
    case corpusUnavailable(BenchmarkCorpusSource)
    case corpusEmpty(BenchmarkCorpusSource)
    case processFailed(String, Int32, String)
    case invalidAudioFile(String)

    var errorDescription: String? {
        switch self {
        case .corpusUnavailable(let source):
            return "\(source.displayName) is not available yet."
        case .corpusEmpty(let source):
            return "\(source.displayName) does not contain any usable benchmark samples."
        case .processFailed(let executable, let status, let output):
            return "\(executable) failed with status \(status): \(output)"
        case .invalidAudioFile(let path):
            return "Invalid benchmark audio file: \(path)"
        }
    }
}

struct LoadedBenchmarkCorpusSample: Identifiable, Sendable {
    let id: String
    let referenceText: String
    let audioURL: URL
    let durationSeconds: TimeInterval
    let audioSamples: [Float]
    let pcm16MonoData: Data
}

actor BenchmarkCorpusManager {
    private let fileManager = FileManager.default

    func hasCorpus(_ source: BenchmarkCorpusSource) -> Bool {
        switch source {
        case .standard:
            return fileManager.fileExists(atPath: AppStoragePaths.standardBenchmarkManifestURL.path)
        case .recentRecordings:
            return fileManager.fileExists(atPath: AppStoragePaths.recentBenchmarkManifestURL.path)
        }
    }

    func loadCorpus(source: BenchmarkCorpusSource) async throws -> [LoadedBenchmarkCorpusSample] {
        switch source {
        case .standard:
            return try await loadStandardCorpus()
        case .recentRecordings:
            return try loadRecentCorpus()
        }
    }

    private func loadStandardCorpus() async throws -> [LoadedBenchmarkCorpusSample] {
        let manifestURL = AppStoragePaths.standardBenchmarkManifestURL
        if !isManifestUsable(at: manifestURL) {
            try generateStandardCorpus()
        }
        return try loadManifest(from: manifestURL)
    }

    private func loadRecentCorpus() throws -> [LoadedBenchmarkCorpusSample] {
        let manifestURL = AppStoragePaths.recentBenchmarkManifestURL
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            throw BenchmarkCorpusError.corpusUnavailable(.recentRecordings)
        }

        let samples = try loadManifest(from: manifestURL)
        guard !samples.isEmpty else {
            throw BenchmarkCorpusError.corpusEmpty(.recentRecordings)
        }
        return samples
    }

    private func generateStandardCorpus() throws {
        try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.standardBenchmarkCorpusDirectory)

        let phrases = Self.standardCorpusPhrases
        var manifestItems: [BenchmarkCorpusManifestItem] = []
        manifestItems.reserveCapacity(phrases.count)

        for (index, phrase) in phrases.enumerated() {
            let audioURL = AppStoragePaths.standardBenchmarkCorpusDirectory
                .appendingPathComponent("standard-\(index).wav")
            try synthesizeSpeech(phrase, to: audioURL)
            let duration = try audioDuration(at: audioURL)
            manifestItems.append(
                BenchmarkCorpusManifestItem(
                    transcriptionID: "standard-\(index)",
                    capturedAt: Date(timeIntervalSince1970: 1_700_000_000 + Double(index)),
                    referenceText: phrase,
                    audioPath: audioURL.path,
                    durationSeconds: duration,
                    transcriptionModelName: nil
                )
            )
        }

        let manifest = BenchmarkCorpusManifest(items: manifestItems)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: AppStoragePaths.standardBenchmarkManifestURL, options: .atomic)
    }

    private func loadManifest(from manifestURL: URL) throws -> [LoadedBenchmarkCorpusSample] {
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(BenchmarkCorpusManifest.self, from: data)

        let samples = try manifest.items.compactMap { item -> LoadedBenchmarkCorpusSample? in
            let audioURL = URL(fileURLWithPath: item.audioPath)
            guard fileManager.fileExists(atPath: audioURL.path) else { return nil }

            let audioData = try Data(contentsOf: audioURL)
            guard audioData.count > 44 else {
                throw BenchmarkCorpusError.invalidAudioFile(audioURL.path)
            }

            let pcmData = Data(audioData.dropFirst(44))
            let audioSamples = LocalTranscriptionService.decodePCM16Mono(pcmData)
            let duration: TimeInterval
            if let storedDuration = item.durationSeconds {
                duration = storedDuration
            } else {
                duration = try audioDuration(at: audioURL)
            }

            return LoadedBenchmarkCorpusSample(
                id: item.id,
                referenceText: item.referenceText,
                audioURL: audioURL,
                durationSeconds: duration,
                audioSamples: audioSamples,
                pcm16MonoData: pcmData
            )
        }

        guard !samples.isEmpty else {
            let source: BenchmarkCorpusSource = manifestURL == AppStoragePaths.standardBenchmarkManifestURL ? .standard : .recentRecordings
            throw BenchmarkCorpusError.corpusEmpty(source)
        }

        return samples
    }

    private func isManifestUsable(at url: URL) -> Bool {
        guard fileManager.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(BenchmarkCorpusManifest.self, from: data),
              !manifest.items.isEmpty else {
            return false
        }

        return manifest.items.allSatisfy { item in
            fileManager.fileExists(atPath: item.audioPath)
        }
    }

    private func synthesizeSpeech(_ text: String, to outputURL: URL) throws {
        let intermediateURL = outputURL.deletingPathExtension().appendingPathExtension("aiff")
        defer { try? fileManager.removeItem(at: intermediateURL) }

        try runProcess(
            executable: "/usr/bin/say",
            arguments: ["-v", "Samantha", "-o", intermediateURL.path, text]
        )
        try runProcess(
            executable: "/usr/bin/afconvert",
            arguments: ["-f", "WAVE", "-d", "LEI16@16000", "-c", "1", intermediateURL.path, outputURL.path]
        )
    }

    private func audioDuration(at url: URL) throws -> TimeInterval {
        let audioFile = try AVAudioFile(forReading: url)
        return Double(audioFile.length) / audioFile.processingFormat.sampleRate
    }

    private func runProcess(executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(decoding: data, as: UTF8.self)
            throw BenchmarkCorpusError.processFailed(executable, process.terminationStatus, output)
        }
    }

    private static let standardCorpusPhrases: [String] = [
        "Voice Ink is running a local benchmark suite on Apple Silicon.",
        "Please schedule a follow up with Sarah for next Thursday at two PM.",
        "The quarterly revenue was one hundred twenty three point four million dollars.",
        "Whisper Large V3 Turbo should stay fast on a MacBook Pro with an M4 Pro chip.",
        "Voxtral and Cohere are local Apple Silicon transcription models in this app.",
        "Parakeet version two is fast, but accuracy varies depending on the audio sample.",
        "This benchmark compares speed and transcript quality for every supported local model."
    ]
}
