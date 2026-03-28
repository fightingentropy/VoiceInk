import Testing
@testable import VoiceInk

struct BenchmarkMetricsTests {
    @Test
    func canonicalMetricsNormalizeNumberWords() {
        let metrics = benchmarkTranscriptMetrics(
            reference: "The quarterly revenue was one hundred twenty three point four million dollars.",
            hypothesis: "The quarterly revenue was 123.4 million."
        )

        #expect(metrics.canonicalReference.contains("123.4"))
        #expect(metrics.canonicalHypothesis.contains("123.4"))
        #expect(metrics.canonicalWordErrorRate < metrics.rawWordErrorRate)
    }

    @Test
    func builtInLocalModelsRemainBenchmarkable() {
        let benchmarkableProviders: Set<ModelProvider> = [
            .local,
            .parakeet,
            .nativeApple,
            .localVoxtral,
            .cohereTranscribe
        ]

        let localCatalog = PredefinedModels.models.filter { benchmarkableProviders.contains($0.provider) }
        let allBenchmarkable = localCatalog.allSatisfy { $0.supportsOnDeviceBenchmarking }

        #expect(!localCatalog.isEmpty)
        #expect(allBenchmarkable)
    }

    @Test
    func markdownExportIncludesSummaryAndComparison() {
        let previousReport = BenchmarkRunReport(
            id: "benchmark-previous",
            generatedAt: .distantPast,
            corpusSource: .standard,
            device: BenchmarkDeviceSnapshot(
                hardwareModel: "Mac16,7",
                operatingSystemVersion: "macOS 26.0",
                appVersion: "1.0",
                appBuild: "100"
            ),
            corpus: [
                BenchmarkCorpusSummary(
                    id: "sample-1",
                    referenceText: "Hello world",
                    audioSeconds: 1.2
                )
            ],
            results: [
                BenchmarkModelResult(
                    modelName: "whisper-large-v3-turbo",
                    displayName: "Whisper Large v3 Turbo",
                    providerName: ModelProvider.local.rawValue,
                    status: .ok,
                    detail: nil,
                    averageElapsedSeconds: 1.5,
                    averageRealtimeFactor: 0.8,
                    averageWordErrorRate: 0.1,
                    averageCanonicalWordErrorRate: 0.08,
                    averageCanonicalExactMatchRate: 1,
                    samples: []
                )
            ]
        )

        let currentReport = BenchmarkRunReport(
            id: "benchmark-current",
            generatedAt: .now,
            corpusSource: .standard,
            device: BenchmarkDeviceSnapshot(
                hardwareModel: "Mac16,7",
                operatingSystemVersion: "macOS 26.0",
                appVersion: "1.1",
                appBuild: "101"
            ),
            corpus: [
                BenchmarkCorpusSummary(
                    id: "sample-1",
                    referenceText: "Hello world",
                    audioSeconds: 1.2
                )
            ],
            results: [
                BenchmarkModelResult(
                    modelName: "whisper-large-v3-turbo",
                    displayName: "Whisper Large v3 Turbo",
                    providerName: ModelProvider.local.rawValue,
                    status: .ok,
                    detail: nil,
                    averageElapsedSeconds: 1.3,
                    averageRealtimeFactor: 0.92,
                    averageWordErrorRate: 0.1,
                    averageCanonicalWordErrorRate: 0.05,
                    averageCanonicalExactMatchRate: 1,
                    samples: [
                        BenchmarkSampleResult(
                            id: "sample-1",
                            referenceText: "Hello world",
                            transcript: "Hello world",
                            canonicalReferenceText: "hello world",
                            canonicalTranscript: "hello world",
                            elapsedSeconds: 1.3,
                            realtimeFactor: 0.92,
                            wordErrorRate: 0,
                            canonicalWordErrorRate: 0,
                            canonicalExactMatchRate: 1
                        )
                    ]
                )
            ]
        )

        let markdown = BenchmarkSuiteStore.markdownContents(
            for: currentReport,
            previousComparable: previousReport
        )

        #expect(markdown.contains("# VoiceInk Benchmark Report"))
        #expect(markdown.contains("## Summary"))
        #expect(markdown.contains("Whisper Large v3 Turbo"))
        #expect(markdown.contains("RTF +0.12"))
        #expect(markdown.contains("cWER -0.03"))
        #expect(markdown.contains("## Whisper Large v3 Turbo"))
    }
}
