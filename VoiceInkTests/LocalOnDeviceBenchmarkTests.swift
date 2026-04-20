import AVFoundation
import Foundation
import Testing
@testable import VoiceInk

@Suite(.serialized, .timeLimit(.minutes(15)))
struct LocalOnDeviceBenchmarkTests {
    @Test
    func benchmarkOnDeviceModels() async throws {
        guard Self.shouldRunBenchmarks else {
            return
        }

        let corpus = try makeCorpus()
        defer {
            for item in corpus {
                guard item.cleanupAfterRun else { continue }
                try? FileManager.default.removeItem(at: item.audioURL)
            }
        }

        let report = try await LocalBenchmarkRunner(corpus: corpus).run()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)

        let outputURL = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("voiceink-local-benchmark-\(Int(Date().timeIntervalSince1970)).json")
        try data.write(to: outputURL, options: .atomic)

        print("VOICEINK_BENCHMARK_REPORT \(outputURL.path)")
        print(String(decoding: data, as: UTF8.self))

        #expect(
            report.results.contains(where: {
                $0.displayName == "Voxtral Realtime (Local MLX)" && $0.status == "ok"
            }),
            "Native Voxtral benchmark did not complete successfully."
        )
    }

    private static var shouldRunBenchmarks: Bool {
        if ProcessInfo.processInfo.environment["VOICEINK_RUN_LOCAL_BENCHMARKS"] == "1" {
            return true
        }

        let sentinelPath = "/tmp/voiceink-run-local-benchmarks"
        return FileManager.default.fileExists(atPath: sentinelPath)
    }

    private func makeCorpus() throws -> [BenchmarkCorpusItem] {
        if let manifestURL = preferredBenchmarkManifestURL() {
            let corpus = try loadCorpus(fromManifestAt: manifestURL)
            if !corpus.isEmpty {
                return corpus
            }
        }

        let phrases = [
            "Voice Ink is running a native Voxtral benchmark on Apple Silicon.",
            "Please schedule a follow up with Sarah for next Thursday at two PM.",
            "The quarterly revenue was one hundred twenty three point four million dollars.",
            "Mistral Voxtral runs locally on Apple Silicon.",
            "Parakeet version two is fast, but Voxtral feels more accurate in practice.",
            "Deepgram Nova three and ElevenLabs Scribe version two are cloud transcription models.",
            "The benchmark compares speed against accuracy for every local model."
        ]

        let tempDirectory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)

        return try phrases.enumerated().map { index, phrase in
            let audioURL = tempDirectory.appendingPathComponent("benchmark-\(index).wav")
            try synthesizeSpeech(phrase, to: audioURL)
            let duration = try audioDuration(at: audioURL)
            return BenchmarkCorpusItem(
                referenceText: phrase,
                audioURL: audioURL,
                durationSeconds: duration,
                cleanupAfterRun: true
            )
        }
    }

    private func loadCorpus(fromManifestAt manifestURL: URL) throws -> [BenchmarkCorpusItem] {
        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        let manifestItems: [ExternalBenchmarkItem]

        if let manifest = try? decoder.decode(BenchmarkCorpusManifest.self, from: data) {
            manifestItems = manifest.items.map {
                ExternalBenchmarkItem(
                    referenceText: $0.referenceText,
                    audioPath: $0.audioPath,
                    durationSeconds: $0.durationSeconds
                )
            }
        } else {
            manifestItems = try decoder.decode(ExternalBenchmarkManifest.self, from: data).items
        }

        return try manifestItems.map { item in
            let audioURL = URL(fileURLWithPath: item.audioPath)
            let duration = try item.durationSeconds ?? audioDuration(at: audioURL)
            return BenchmarkCorpusItem(
                referenceText: item.referenceText,
                audioURL: audioURL,
                durationSeconds: duration,
                cleanupAfterRun: false
            )
        }
    }

    private func preferredBenchmarkManifestURL() -> URL? {
        if let manifestURL = externalBenchmarkManifestURL() {
            return manifestURL
        }

        let appManagedManifestURL = AppStoragePaths.recentBenchmarkManifestURL
        guard FileManager.default.fileExists(atPath: appManagedManifestURL.path) else {
            return nil
        }

        return appManagedManifestURL
    }

    private func externalBenchmarkManifestURL() -> URL? {
        if let manifestPath = ProcessInfo.processInfo.environment["VOICEINK_BENCHMARK_MANIFEST"],
           !manifestPath.isEmpty {
            return URL(fileURLWithPath: manifestPath)
        }

        let sentinelURL = URL(fileURLWithPath: "/tmp/voiceink-benchmark-manifest-path")
        guard let manifestPath = try? String(contentsOf: sentinelURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !manifestPath.isEmpty else {
            return nil
        }

        return URL(fileURLWithPath: manifestPath)
    }

    private func synthesizeSpeech(_ text: String, to outputURL: URL) throws {
        let intermediateURL = outputURL.deletingPathExtension().appendingPathExtension("aiff")
        defer { try? FileManager.default.removeItem(at: intermediateURL) }

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
            throw BenchmarkError.processFailed(executable, process.terminationStatus, output)
        }
    }
}

private struct LocalBenchmarkRunner {
    let corpus: [BenchmarkCorpusItem]

    func run() async throws -> BenchmarkReport {
        var results: [ModelBenchmarkResult] = []
        results.append(try await benchmarkVoxtralNative())
        results.append(try await benchmarkParakeet(named: "parakeet-tdt-0.6b-v2"))
        results.append(try await benchmarkAppleSpeech())

        return BenchmarkReport(
            generatedAt: Date(),
            corpus: corpus.map {
                BenchmarkCorpusSummary(
                    referenceText: $0.referenceText,
                    audioSeconds: $0.durationSeconds
                )
            },
            results: results
        )
    }

    private func benchmarkVoxtralNative() async throws -> ModelBenchmarkResult {
        guard let model = PredefinedModels.models.first(where: { $0.name == "voxtral-mini-realtime-local" }) as? LocalVoxtralModel else {
            throw BenchmarkError.modelMissing("voxtral-mini-realtime-local")
        }

        let lease = try await VoxtralNativeRuntime.shared.acquirePreparedState(
            modelReference: LocalVoxtralConfiguration.modelName,
            autoDownload: true
        )
        defer {
            Task {
                await lease.release()
            }
        }

        _ = try await streamingTranscript(
            for: corpus[0].audioURL,
            preparedState: lease.preparedState
        )

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for item in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await streamingTranscript(
                for: item.audioURL,
                preparedState: lease.preparedState
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(item: item, transcript: transcript, elapsed: elapsed))
        }

        return ModelBenchmarkResult(
            displayName: model.displayName,
            status: "ok",
            detail: nil,
            averageElapsedSeconds: samples.map(\.elapsedSeconds).average,
            averageRealtimeFactor: samples.map(\.realtimeFactor).average,
            averageWordErrorRate: samples.map(\.wordErrorRate).average,
            averageCanonicalWordErrorRate: samples.map(\.canonicalWordErrorRate).average,
            averageCanonicalExactMatchRate: samples.map(\.canonicalExactMatchRate).average,
            samples: samples
        )
    }

    private func benchmarkParakeet(named modelName: String) async throws -> ModelBenchmarkResult {
        guard let model = PredefinedModels.models.first(where: { $0.name == modelName }) as? ParakeetModel else {
            throw BenchmarkError.modelMissing(modelName)
        }

        let service = ParakeetTranscriptionService()
        do {
            _ = try await service.transcribe(audioURL: corpus[0].audioURL, model: model)
        } catch {
            await service.cleanup()
            return ModelBenchmarkResult(
                displayName: model.displayName,
                status: "unavailable",
                detail: error.localizedDescription,
                averageElapsedSeconds: nil,
                averageRealtimeFactor: nil,
                averageWordErrorRate: nil,
                averageCanonicalWordErrorRate: nil,
                averageCanonicalExactMatchRate: nil,
                samples: []
            )
        }

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for item in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await service.transcribe(audioURL: item.audioURL, model: model)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(item: item, transcript: transcript, elapsed: elapsed))
        }

        await service.cleanup()
        return ModelBenchmarkResult(
            displayName: model.displayName,
            status: "ok",
            detail: nil,
            averageElapsedSeconds: samples.map(\.elapsedSeconds).average,
            averageRealtimeFactor: samples.map(\.realtimeFactor).average,
            averageWordErrorRate: samples.map(\.wordErrorRate).average,
            averageCanonicalWordErrorRate: samples.map(\.canonicalWordErrorRate).average,
            averageCanonicalExactMatchRate: samples.map(\.canonicalExactMatchRate).average,
            samples: samples
        )
    }

    private func benchmarkAppleSpeech() async throws -> ModelBenchmarkResult {
        guard let model = PredefinedModels.models.first(where: { $0.name == "apple-speech" }) as? NativeAppleModel else {
            throw BenchmarkError.modelMissing("apple-speech")
        }

        let service = NativeAppleTranscriptionService()
        do {
            _ = try await service.transcribe(audioURL: corpus[0].audioURL, model: model)
        } catch {
            return ModelBenchmarkResult(
                displayName: model.displayName,
                status: "unavailable",
                detail: error.localizedDescription,
                averageElapsedSeconds: nil,
                averageRealtimeFactor: nil,
                averageWordErrorRate: nil,
                averageCanonicalWordErrorRate: nil,
                averageCanonicalExactMatchRate: nil,
                samples: []
            )
        }

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for item in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await service.transcribe(audioURL: item.audioURL, model: model)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(item: item, transcript: transcript, elapsed: elapsed))
        }

        return ModelBenchmarkResult(
            displayName: model.displayName,
            status: "ok",
            detail: nil,
            averageElapsedSeconds: samples.map(\.elapsedSeconds).average,
            averageRealtimeFactor: samples.map(\.realtimeFactor).average,
            averageWordErrorRate: samples.map(\.wordErrorRate).average,
            averageCanonicalWordErrorRate: samples.map(\.canonicalWordErrorRate).average,
            averageCanonicalExactMatchRate: samples.map(\.canonicalExactMatchRate).average,
            samples: samples
        )
    }

    private func makeSampleResult(
        item: BenchmarkCorpusItem,
        transcript: String,
        elapsed: TimeInterval
    ) -> BenchmarkSampleResult {
        let metrics = transcriptMetrics(reference: item.referenceText, hypothesis: transcript)

        return BenchmarkSampleResult(
            referenceText: item.referenceText,
            transcript: transcript,
            canonicalReferenceText: metrics.canonicalReference,
            canonicalTranscript: metrics.canonicalHypothesis,
            elapsedSeconds: elapsed,
            realtimeFactor: item.durationSeconds / elapsed,
            wordErrorRate: metrics.rawWordErrorRate,
            canonicalWordErrorRate: metrics.canonicalWordErrorRate,
            canonicalExactMatchRate: metrics.canonicalExactMatch ? 1 : 0
        )
    }

    private func streamingTranscript(
        for audioURL: URL,
        preparedState: VoxtralNativePreparedState
    ) async throws -> String {
        let samples = try VoxtralNativeAudio.loadAudioFile(audioURL)
        let (stream, continuation) = AsyncStream.makeStream(
            of: StreamingTranscriptionEvent.self,
            bufferingPolicy: .unbounded
        )
        let engine = VoxtralNativeStreamingEngine(
            preparedState: preparedState,
            continuation: continuation
        )

        let recorder = BenchmarkEventRecorder()
        let consumer = Task {
            for await event in stream {
                await recorder.record(event)
            }
        }

        for chunk in pcmChunks(from: samples, chunkSampleCount: 1_280) {
            try await engine.ingestPCM16(chunk)
        }

        try await engine.finalize()
        continuation.finish()
        await consumer.value
        return await recorder.finalTranscript()
    }

    private func pcmChunks(from samples: [Float], chunkSampleCount: Int) -> [Data] {
        let int16Samples = samples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * Float(Int16.max))
        }

        var chunks: [Data] = []
        chunks.reserveCapacity(max(1, int16Samples.count / chunkSampleCount))

        var index = 0
        while index < int16Samples.count {
            let end = min(index + chunkSampleCount, int16Samples.count)
            let slice = int16Samples[index ..< end]
            chunks.append(slice.withUnsafeBytes { Data($0) })
            index = end
        }

        return chunks
    }
}

private actor BenchmarkEventRecorder {
    private var committedTexts: [String] = []
    private var partialTexts: [String] = []

    func record(_ event: StreamingTranscriptionEvent) {
        switch event {
        case .committed(let text):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                committedTexts.append(trimmed)
            }
        case .partial(let text):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                partialTexts.append(trimmed)
            }
        case .sessionStarted, .error:
            break
        }
    }

    func finalTranscript() -> String {
        if let committed = committedTexts.last {
            return committed
        }
        return partialTexts.last ?? ""
    }
}

private struct BenchmarkReport: Codable {
    let generatedAt: Date
    let corpus: [BenchmarkCorpusSummary]
    let results: [ModelBenchmarkResult]
}

private struct BenchmarkCorpusSummary: Codable {
    let referenceText: String
    let audioSeconds: TimeInterval
}

private struct BenchmarkCorpusItem {
    let referenceText: String
    let audioURL: URL
    let durationSeconds: TimeInterval
    let cleanupAfterRun: Bool
}

private struct ExternalBenchmarkManifest: Decodable {
    let items: [ExternalBenchmarkItem]
}

private struct ExternalBenchmarkItem: Decodable {
    let referenceText: String
    let audioPath: String
    let durationSeconds: TimeInterval?
}

private struct ModelBenchmarkResult: Codable {
    let displayName: String
    let status: String
    let detail: String?
    let averageElapsedSeconds: TimeInterval?
    let averageRealtimeFactor: Double?
    let averageWordErrorRate: Double?
    let averageCanonicalWordErrorRate: Double?
    let averageCanonicalExactMatchRate: Double?
    let samples: [BenchmarkSampleResult]
}

private struct BenchmarkSampleResult: Codable {
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

private enum BenchmarkError: LocalizedError {
    case modelMissing(String)
    case processFailed(String, Int32, String)

    var errorDescription: String? {
        switch self {
        case .modelMissing(let modelName):
            return "Missing benchmark model: \(modelName)"
        case .processFailed(let executable, let status, let output):
            return "\(executable) failed with status \(status): \(output)"
        }
    }
}

private func wordErrorRate(reference: String, hypothesis: String) -> Double {
    let referenceWords = normalizedWords(from: reference)
    let hypothesisWords = normalizedWords(from: hypothesis)

    guard !referenceWords.isEmpty else { return hypothesisWords.isEmpty ? 0 : 1 }

    let distance = levenshteinDistance(referenceWords, hypothesisWords)
    return Double(distance) / Double(referenceWords.count)
}

private func transcriptMetrics(reference: String, hypothesis: String) -> BenchmarkTranscriptMetrics {
    let canonicalReferenceWords = canonicalWords(from: reference)
    let canonicalHypothesisWords = canonicalWords(from: hypothesis)

    let canonicalReference = canonicalReferenceWords.joined(separator: " ")
    let canonicalHypothesis = canonicalHypothesisWords.joined(separator: " ")

    let canonicalWordErrorRate = wordErrorRate(
        referenceWords: canonicalReferenceWords,
        hypothesisWords: canonicalHypothesisWords
    )

    return BenchmarkTranscriptMetrics(
        rawWordErrorRate: wordErrorRate(reference: reference, hypothesis: hypothesis),
        canonicalWordErrorRate: canonicalWordErrorRate,
        canonicalReference: canonicalReference,
        canonicalHypothesis: canonicalHypothesis
    )
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
            let value = String(token.dropFirst())
            canonicalTokens.append(value)
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

private func wordErrorRate(referenceWords: [String], hypothesisWords: [String]) -> Double {
    guard !referenceWords.isEmpty else { return hypothesisWords.isEmpty ? 0 : 1 }

    let distance = levenshteinDistance(referenceWords, hypothesisWords)
    return Double(distance) / Double(referenceWords.count)
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

private struct BenchmarkTranscriptMetrics {
    let rawWordErrorRate: Double
    let canonicalWordErrorRate: Double
    let canonicalReference: String
    let canonicalHypothesis: String

    var canonicalExactMatch: Bool {
        canonicalReference == canonicalHypothesis
    }
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

private extension Array where Element == Double {
    var average: Double {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / Double(count)
    }
}
