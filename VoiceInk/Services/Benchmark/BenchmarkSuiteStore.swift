import AppKit
import Darwin
import Foundation

@MainActor
final class BenchmarkSuiteStore: ObservableObject {
    @Published var selectedCorpusSource: BenchmarkCorpusSource = .standard
    @Published private(set) var reports: [BenchmarkRunReport] = []
    @Published private(set) var latestReport: BenchmarkRunReport?
    @Published private(set) var previousComparableReport: BenchmarkRunReport?
    @Published private(set) var recentCorpusAvailable = false
    @Published private(set) var isRunning = false
    @Published private(set) var currentStepDescription: String?
    @Published private(set) var completedModelCount = 0
    @Published private(set) var totalModelCount = 0
    @Published private(set) var lastErrorMessage: String?

    private let corpusManager = BenchmarkCorpusManager()

    func refresh() {
        reports = loadReports()
        latestReport = reports.first
        previousComparableReport = findPreviousComparableReport(for: latestReport)

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.recentCorpusAvailable = await self.corpusManager.hasCorpus(.recentRecordings)
        }
    }

    func runBenchmarks(
        models: [any TranscriptionModel],
        whisperModelManager: WhisperModelManager,
        parakeetModelManager: ParakeetModelManager
    ) {
        guard !isRunning else { return }

        Task {
            await runBenchmarksTask(
                models: models,
                whisperModelManager: whisperModelManager,
                parakeetModelManager: parakeetModelManager
            )
        }
    }

    func openReportsFolder() {
        do {
            try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.benchmarkReportsDirectory)
            NSWorkspace.shared.activateFileViewerSelecting([AppStoragePaths.benchmarkReportsDirectory])
        } catch {
            lastErrorMessage = error.localizedDescription
        }
    }

    func exportLatestReportMarkdown() {
        guard let latestReport else { return }

        do {
            let previousComparable = findPreviousComparableReport(for: latestReport)
            let outputURL = try writeMarkdownReport(
                for: latestReport,
                previousComparable: previousComparable
            )
            NSWorkspace.shared.open(outputURL)
        } catch {
            lastErrorMessage = error.localizedDescription
        }
    }

    func openCorpusFolder() {
        let targetDirectory: URL
        switch selectedCorpusSource {
        case .standard:
            targetDirectory = AppStoragePaths.standardBenchmarkCorpusDirectory
        case .recentRecordings:
            targetDirectory = AppStoragePaths.recentBenchmarkCorpusDirectory
        }

        do {
            try AppStoragePaths.createDirectoryIfNeeded(at: targetDirectory)
            NSWorkspace.shared.activateFileViewerSelecting([targetDirectory])
        } catch {
            lastErrorMessage = error.localizedDescription
        }
    }

    private func runBenchmarksTask(
        models: [any TranscriptionModel],
        whisperModelManager: WhisperModelManager,
        parakeetModelManager: ParakeetModelManager
    ) async {
        isRunning = true
        lastErrorMessage = nil
        currentStepDescription = "Preparing corpus"
        completedModelCount = 0

        defer {
            isRunning = false
            currentStepDescription = nil
        }

        do {
            let corpus = try await corpusManager.loadCorpus(source: selectedCorpusSource)
            let benchmarkableModels = models.filter(\.supportsOnDeviceBenchmarking)
            totalModelCount = benchmarkableModels.count

            let results = try await runModelBenchmarks(
                benchmarkableModels,
                corpus: corpus,
                whisperModelManager: whisperModelManager,
                parakeetModelManager: parakeetModelManager
            )

            let report = BenchmarkRunReport(
                id: Self.makeReportID(),
                generatedAt: Date(),
                corpusSource: selectedCorpusSource,
                device: Self.makeDeviceSnapshot(),
                corpus: corpus.map {
                    BenchmarkCorpusSummary(
                        id: $0.id,
                        referenceText: $0.referenceText,
                        audioSeconds: $0.durationSeconds
                    )
                },
                results: results
            )

            try persist(report)
            refresh()
        } catch {
            lastErrorMessage = error.localizedDescription
        }
    }

    private func runModelBenchmarks(
        _ models: [any TranscriptionModel],
        corpus: [LoadedBenchmarkCorpusSample],
        whisperModelManager: WhisperModelManager,
        parakeetModelManager: ParakeetModelManager
    ) async throws -> [BenchmarkModelResult] {
        var results: [BenchmarkModelResult] = []
        results.reserveCapacity(models.count)

        for model in models {
            currentStepDescription = "Benchmarking \(model.displayName)"
            let result = await benchmarkResult(
                for: model,
                corpus: corpus,
                whisperModelManager: whisperModelManager,
                parakeetModelManager: parakeetModelManager
            )
            results.append(result)
            completedModelCount += 1
        }

        return results
    }

    private func benchmarkResult(
        for model: any TranscriptionModel,
        corpus: [LoadedBenchmarkCorpusSample],
        whisperModelManager: WhisperModelManager,
        parakeetModelManager: ParakeetModelManager
    ) async -> BenchmarkModelResult {
        do {
            switch model.provider {
            case .local:
                guard let whisperModel = model as? LocalModel else {
                    return makeFailureResult(for: model, detail: "Unsupported local Whisper model.")
                }
                return try await benchmarkWhisper(
                    whisperModel,
                    corpus: corpus,
                    whisperModelManager: whisperModelManager
                )
            case .cohereTranscribe:
                guard let cohereModel = model as? LocalCohereTranscribeModel else {
                    return makeFailureResult(for: model, detail: "Unsupported Cohere model.")
                }
                return try await benchmarkCohere(cohereModel, corpus: corpus)
            case .localVoxtral:
                guard let voxtralModel = model as? LocalVoxtralModel else {
                    return makeFailureResult(for: model, detail: "Unsupported Voxtral model.")
                }
                return try await benchmarkVoxtral(voxtralModel, corpus: corpus)
            case .parakeet:
                guard let parakeetModel = model as? ParakeetModel else {
                    return makeFailureResult(for: model, detail: "Unsupported Parakeet model.")
                }
                return try await benchmarkParakeet(
                    parakeetModel,
                    corpus: corpus,
                    parakeetModelManager: parakeetModelManager
                )
            case .nativeApple:
                guard let nativeAppleModel = model as? NativeAppleModel else {
                    return makeFailureResult(for: model, detail: "Unsupported Apple Speech model.")
                }
                return try await benchmarkAppleSpeech(nativeAppleModel, corpus: corpus)
            case .elevenLabs, .custom:
                return makeUnavailableResult(for: model, detail: "Cloud models are excluded from on-device benchmarks.")
            }
        } catch {
            return makeFailureResult(for: model, detail: error.localizedDescription)
        }
    }

    private func benchmarkWhisper(
        _ model: LocalModel,
        corpus: [LoadedBenchmarkCorpusSample],
        whisperModelManager: WhisperModelManager
    ) async throws -> BenchmarkModelResult {
        guard let managedModel = whisperModelManager.availableModels.first(where: { $0.name == model.name }) else {
            return makeUnavailableResult(for: model, detail: "Model is not downloaded.")
        }

        let runtime = try await WhisperKitRuntime(modelFolder: managedModel.url.path)
        defer {
            Task {
                await runtime.unload()
            }
        }

        let prompt = normalizedPrompt(UserDefaults.standard.string(forKey: "TranscriptionPrompt"))
        let language = selectedLanguageCode()

        _ = try await runtime.transcribe(
            samples: corpus[0].audioSamples,
            prompt: prompt,
            language: language
        )

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for sample in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await runtime.transcribe(
                samples: sample.audioSamples,
                prompt: prompt,
                language: language
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(sample: sample, transcript: transcript, elapsed: elapsed))
        }

        return makeModelResult(for: model, status: .ok, detail: nil, samples: samples)
    }

    private func benchmarkCohere(
        _ model: LocalCohereTranscribeModel,
        corpus: [LoadedBenchmarkCorpusSample]
    ) async throws -> BenchmarkModelResult {
        guard CohereNativeModelManager.shared.isModelDownloaded() else {
            return makeUnavailableResult(for: model, detail: "Model is not downloaded.")
        }

        let service = CohereTranscribeTranscriptionService()
        try await service.warmup(for: model)

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for sample in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await service.transcribe(
                recordedPCMBuffer: sample.pcm16MonoData,
                sampleRate: 16_000,
                model: model
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(sample: sample, transcript: transcript, elapsed: elapsed))
        }

        return makeModelResult(for: model, status: .ok, detail: nil, samples: samples)
    }

    private func benchmarkVoxtral(
        _ model: LocalVoxtralModel,
        corpus: [LoadedBenchmarkCorpusSample]
    ) async throws -> BenchmarkModelResult {
        switch VoxtralNativeModelManager.shared.availability(for: LocalVoxtralConfiguration.modelName) {
        case .missing:
            return makeUnavailableResult(for: model, detail: "Model is not downloaded.")
        case .appManaged, .externalLocalPath:
            break
        }

        let lease = try await VoxtralNativeRuntime.shared.acquirePreparedState(
            modelReference: LocalVoxtralConfiguration.modelName,
            autoDownload: false
        )
        defer {
            Task {
                await lease.release()
            }
        }

        _ = try await streamingTranscript(for: corpus[0], preparedState: lease.preparedState)

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for sample in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await streamingTranscript(for: sample, preparedState: lease.preparedState)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(sample: sample, transcript: transcript, elapsed: elapsed))
        }

        return makeModelResult(for: model, status: .ok, detail: nil, samples: samples)
    }

    private func benchmarkParakeet(
        _ model: ParakeetModel,
        corpus: [LoadedBenchmarkCorpusSample],
        parakeetModelManager: ParakeetModelManager
    ) async throws -> BenchmarkModelResult {
        guard parakeetModelManager.isParakeetModelDownloaded(named: model.name) else {
            return makeUnavailableResult(for: model, detail: "Model is not downloaded.")
        }

        let service = ParakeetTranscriptionService()
        // `service` is now an actor; `defer` cannot `await`, so wrap the
        // cleanup call in a do/catch-style explicit release after the loop.
        do {
            _ = try await service.transcribe(audioURL: corpus[0].audioURL, model: model)

            var samples: [BenchmarkSampleResult] = []
            samples.reserveCapacity(corpus.count)

            for sample in corpus {
                let start = CFAbsoluteTimeGetCurrent()
                let transcript = try await service.transcribe(audioURL: sample.audioURL, model: model)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                samples.append(makeSampleResult(sample: sample, transcript: transcript, elapsed: elapsed))
            }

            await service.cleanup()
            return makeModelResult(for: model, status: .ok, detail: nil, samples: samples)
        } catch {
            await service.cleanup()
            throw error
        }
    }

    private func benchmarkAppleSpeech(
        _ model: NativeAppleModel,
        corpus: [LoadedBenchmarkCorpusSample]
    ) async throws -> BenchmarkModelResult {
        let service = NativeAppleTranscriptionService()

        do {
            _ = try await service.transcribe(audioURL: corpus[0].audioURL, model: model)
        } catch {
            return makeUnavailableResult(for: model, detail: error.localizedDescription)
        }

        var samples: [BenchmarkSampleResult] = []
        samples.reserveCapacity(corpus.count)

        for sample in corpus {
            let start = CFAbsoluteTimeGetCurrent()
            let transcript = try await service.transcribe(audioURL: sample.audioURL, model: model)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(makeSampleResult(sample: sample, transcript: transcript, elapsed: elapsed))
        }

        return makeModelResult(for: model, status: .ok, detail: nil, samples: samples)
    }

    private func makeSampleResult(
        sample: LoadedBenchmarkCorpusSample,
        transcript: String,
        elapsed: TimeInterval
    ) -> BenchmarkSampleResult {
        let metrics = benchmarkTranscriptMetrics(reference: sample.referenceText, hypothesis: transcript)

        return BenchmarkSampleResult(
            id: sample.id,
            referenceText: sample.referenceText,
            transcript: transcript,
            canonicalReferenceText: metrics.canonicalReference,
            canonicalTranscript: metrics.canonicalHypothesis,
            elapsedSeconds: elapsed,
            realtimeFactor: sample.durationSeconds / elapsed,
            wordErrorRate: metrics.rawWordErrorRate,
            canonicalWordErrorRate: metrics.canonicalWordErrorRate,
            canonicalExactMatchRate: metrics.canonicalExactMatch ? 1 : 0
        )
    }

    private func makeModelResult(
        for model: any TranscriptionModel,
        status: BenchmarkModelStatus,
        detail: String?,
        samples: [BenchmarkSampleResult]
    ) -> BenchmarkModelResult {
        BenchmarkModelResult(
            modelName: model.name,
            displayName: model.displayName,
            providerName: model.provider.rawValue,
            status: status,
            detail: detail,
            averageElapsedSeconds: samples.map(\.elapsedSeconds).benchmarkAverage,
            averageRealtimeFactor: samples.map(\.realtimeFactor).benchmarkAverage,
            averageWordErrorRate: samples.map(\.wordErrorRate).benchmarkAverage,
            averageCanonicalWordErrorRate: samples.map(\.canonicalWordErrorRate).benchmarkAverage,
            averageCanonicalExactMatchRate: samples.map(\.canonicalExactMatchRate).benchmarkAverage,
            samples: samples
        )
    }

    private func makeUnavailableResult(for model: any TranscriptionModel, detail: String) -> BenchmarkModelResult {
        makeModelResult(for: model, status: .unavailable, detail: detail, samples: [])
    }

    private func makeFailureResult(for model: any TranscriptionModel, detail: String) -> BenchmarkModelResult {
        makeModelResult(for: model, status: .failed, detail: detail, samples: [])
    }

    private func streamingTranscript(
        for sample: LoadedBenchmarkCorpusSample,
        preparedState: VoxtralNativePreparedState
    ) async throws -> String {
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

        for chunk in pcmChunks(from: sample.audioSamples, chunkSampleCount: 1_280) {
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

    private func persist(_ report: BenchmarkRunReport) throws {
        try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.benchmarkReportsDirectory)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        try data.write(to: jsonReportURL(for: report), options: .atomic)
        _ = try writeMarkdownReport(
            for: report,
            previousComparable: reports.first(where: { $0.corpusSource == report.corpusSource })
        )
    }

    private func loadReports() -> [BenchmarkRunReport] {
        let directory = AppStoragePaths.benchmarkReportsDirectory
        guard FileManager.default.fileExists(atPath: directory.path),
              let urls = try? FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
              ) else {
            return []
        }

        let decoder = JSONDecoder()
        let reports = urls
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> BenchmarkRunReport? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(BenchmarkRunReport.self, from: data)
            }
            .sorted { $0.generatedAt > $1.generatedAt }

        return reports
    }

    private func findPreviousComparableReport(for latestReport: BenchmarkRunReport?) -> BenchmarkRunReport? {
        guard let latestReport else { return nil }
        return reports.dropFirst().first(where: { $0.corpusSource == latestReport.corpusSource })
    }

    private func normalizedPrompt(_ prompt: String?) -> String? {
        guard let normalized = prompt?.trimmingCharacters(in: .whitespacesAndNewlines),
              !normalized.isEmpty else {
            return nil
        }

        return normalized
    }

    private func selectedLanguageCode() -> String? {
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"
        return selectedLanguage == "auto" ? nil : selectedLanguage
    }

    private static func makeReportID() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "benchmark-\(formatter.string(from: Date()))"
    }

    private static func makeDeviceSnapshot() -> BenchmarkDeviceSnapshot {
        BenchmarkDeviceSnapshot(
            hardwareModel: hardwareModelIdentifier(),
            operatingSystemVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown",
            appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "Unknown"
        )
    }

    private static func hardwareModelIdentifier() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        guard size > 0 else { return "Unknown Mac" }

        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &buffer, &size, nil, 0)
        let bytes = buffer.prefix { $0 != 0 }.map(UInt8.init)
        return String(decoding: bytes, as: UTF8.self)
    }
}

extension BenchmarkSuiteStore {
    nonisolated static func markdownContents(
        for report: BenchmarkRunReport,
        previousComparable: BenchmarkRunReport?
    ) -> String {
        var lines: [String] = []
        lines.append("# VoiceInk Benchmark Report")
        lines.append("")
        lines.append("- Generated: \(report.generatedAt.formatted(date: .abbreviated, time: .standard))")
        lines.append("- Corpus: \(report.corpusSource.displayName)")
        lines.append("- Device: \(report.device.hardwareModel)")
        lines.append("- OS: \(report.device.operatingSystemVersion)")
        lines.append("- App: \(report.device.appVersion) (\(report.device.appBuild))")
        if report.corpusSource == .recentRecordings {
            lines.append("- Accuracy note: Recent Recordings uses the saved transcript text for each recording as the reference. Speed remains useful, but accuracy reflects agreement with those saved transcripts rather than neutral hand-labeled ground truth.")
        }
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Model | Status | Avg s | RTF | cWER | EM | Delta |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")

        for result in report.results {
            let previousResult = previousComparable?.results.first(where: { $0.modelName == result.modelName })
            lines.append(
                "| \(markdownCell(result.displayName)) | \(markdownCell(markdownStatusText(result.status))) | \(markdownMetric(result.averageElapsedSeconds)) | \(markdownMetric(result.averageRealtimeFactor)) | \(markdownMetric(result.averageCanonicalWordErrorRate)) | \(markdownMetric(result.averageCanonicalExactMatchRate)) | \(markdownCell(markdownDeltaSummary(current: result, previous: previousResult) ?? result.detail ?? "—")) |"
            )
        }

        lines.append("")
        lines.append("## Corpus")
        lines.append("")
        lines.append("| Sample | Seconds | Reference |")
        lines.append("| --- | ---: | --- |")

        for sample in report.corpus {
            lines.append(
                "| \(markdownCell(sample.id)) | \(markdownMetric(sample.audioSeconds)) | \(markdownCell(sample.referenceText)) |"
            )
        }

        for result in report.results where !result.samples.isEmpty {
            lines.append("")
            lines.append("## \(result.displayName)")
            lines.append("")
            lines.append("| Sample | Elapsed s | RTF | WER | cWER | Match | Transcript |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")

            for sample in result.samples {
                lines.append(
                    "| \(markdownCell(sample.id)) | \(markdownMetric(sample.elapsedSeconds)) | \(markdownMetric(sample.realtimeFactor)) | \(markdownMetric(sample.wordErrorRate)) | \(markdownMetric(sample.canonicalWordErrorRate)) | \(markdownMetric(sample.canonicalExactMatchRate)) | \(markdownCell(sample.transcript)) |"
                )
            }
        }

        lines.append("")
        return lines.joined(separator: "\n")
    }

    nonisolated private static func markdownStatusText(_ status: BenchmarkModelStatus) -> String {
        switch status {
        case .ok:
            return "OK"
        case .unavailable:
            return "Missing"
        case .failed:
            return "Failed"
        }
    }

    nonisolated private static func markdownMetric(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.2f", value)
    }

    nonisolated private static func markdownCell(_ value: String) -> String {
        value
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
    }

    nonisolated private static func markdownDeltaSummary(
        current: BenchmarkModelResult,
        previous: BenchmarkModelResult?
    ) -> String? {
        guard let previous,
              current.status == .ok,
              previous.status == .ok else {
            return nil
        }

        let rtfDelta = markdownDeltaString(
            current: current.averageRealtimeFactor,
            previous: previous.averageRealtimeFactor
        )
        let cwerDelta = markdownDeltaString(
            current: current.averageCanonicalWordErrorRate,
            previous: previous.averageCanonicalWordErrorRate
        )

        let parts = [
            rtfDelta.map { "RTF \($0)" },
            cwerDelta.map { "cWER \($0)" }
        ].compactMap { $0 }

        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: ", ")
    }

    nonisolated private static func markdownDeltaString(current: Double?, previous: Double?) -> String? {
        guard let current, let previous else { return nil }
        let delta = current - previous
        if abs(delta) < 0.005 {
            return "0.00"
        }

        let sign = delta >= 0 ? "+" : ""
        return "\(sign)\(String(format: "%.2f", delta))"
    }

    private func jsonReportURL(for report: BenchmarkRunReport) -> URL {
        AppStoragePaths.benchmarkReportsDirectory
            .appendingPathComponent("\(report.id).json", isDirectory: false)
    }

    private func markdownReportURL(for report: BenchmarkRunReport) -> URL {
        AppStoragePaths.benchmarkReportsDirectory
            .appendingPathComponent("\(report.id).md", isDirectory: false)
    }

    private func writeMarkdownReport(
        for report: BenchmarkRunReport,
        previousComparable: BenchmarkRunReport?
    ) throws -> URL {
        try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.benchmarkReportsDirectory)
        let outputURL = markdownReportURL(for: report)
        let markdown = Self.markdownContents(for: report, previousComparable: previousComparable)
        try markdown.write(to: outputURL, atomically: true, encoding: .utf8)
        return outputURL
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
