import SwiftUI

struct BenchmarkSettingsSectionView: View {
    @EnvironmentObject private var whisperModelManager: WhisperModelManager
    @EnvironmentObject private var parakeetModelManager: ParakeetModelManager
    @EnvironmentObject private var transcriptionModelManager: TranscriptionModelManager
    @StateObject private var benchmarkStore = BenchmarkSuiteStore()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Benchmarks")
                    .font(.headline)

                Spacer()

                Button("Export Markdown") {
                    benchmarkStore.exportLatestReportMarkdown()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(benchmarkStore.latestReport == nil)

                Button("Corpus Folder") {
                    benchmarkStore.openCorpusFolder()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Reports Folder") {
                    benchmarkStore.openReportsFolder()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Text("Run the same on-device corpus across your local models and keep the results for future comparisons when you add new ones.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("Corpus", selection: $benchmarkStore.selectedCorpusSource) {
                ForEach(BenchmarkCorpusSource.allCases) { source in
                    Text(source.displayName)
                        .tag(source)
                        .disabled(source == .recentRecordings && !benchmarkStore.recentCorpusAvailable)
                }
            }
            .pickerStyle(.menu)

            Text(benchmarkStore.selectedCorpusSource.description)
                .font(.caption)
                .foregroundStyle(.secondary)

            if benchmarkStore.selectedCorpusSource == .recentRecordings {
                Text("Recent Recordings is best for practical latency checks on your own speech. Treat its accuracy numbers as source-transcript agreement, not neutral ground truth.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 12) {
                Button(benchmarkStore.isRunning ? "Running…" : "Run Benchmarks") {
                    benchmarkStore.runBenchmarks(
                        models: transcriptionModelManager.allAvailableModels,
                        whisperModelManager: whisperModelManager,
                        parakeetModelManager: parakeetModelManager
                    )
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(benchmarkStore.isRunning)

                if benchmarkStore.isRunning {
                    ProgressView(
                        value: Double(benchmarkStore.completedModelCount),
                        total: Double(max(benchmarkStore.totalModelCount, 1))
                    )
                    .frame(maxWidth: 140)

                    if let currentStepDescription = benchmarkStore.currentStepDescription {
                        Text(currentStepDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let lastErrorMessage = benchmarkStore.lastErrorMessage {
                Text(lastErrorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if let latestReport = benchmarkStore.latestReport {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Latest Run")
                            .font(.subheadline.weight(.semibold))
                        Spacer()
                        Text(latestReport.generatedAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Text("\(latestReport.device.hardwareModel) • \(latestReport.device.operatingSystemVersion)")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    VStack(spacing: 8) {
                        BenchmarkHeaderRowView()

                        ForEach(latestReport.results) { result in
                            BenchmarkResultRowView(
                                result: result,
                                previousResult: benchmarkStore.previousComparableReport?.results.first(where: { $0.modelName == result.modelName })
                            )
                        }
                    }
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(.windowBackgroundColor).opacity(0.55))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.secondary.opacity(0.14), lineWidth: 1)
                )
            } else {
                Text("No benchmark runs yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear {
            benchmarkStore.refresh()
        }
        .onChange(of: benchmarkStore.selectedCorpusSource) { _, newValue in
            if newValue == .recentRecordings && !benchmarkStore.recentCorpusAvailable {
                benchmarkStore.selectedCorpusSource = .standard
            }
        }
    }
}

private struct BenchmarkHeaderRowView: View {
    var body: some View {
        HStack(spacing: 12) {
            Text("Model")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("Status")
                .frame(width: 80, alignment: .leading)
            Text("RTF")
                .frame(width: 70, alignment: .trailing)
            Text("cWER")
                .frame(width: 70, alignment: .trailing)
            Text("Avg s")
                .frame(width: 70, alignment: .trailing)
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.secondary)
    }
}

private struct BenchmarkResultRowView: View {
    let result: BenchmarkModelResult
    let previousResult: BenchmarkModelResult?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(result.displayName)
                        .font(.system(size: 12, weight: .medium))
                    Text(result.providerName)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Text(statusText)
                    .frame(width: 80, alignment: .leading)
                    .foregroundStyle(statusColor)

                Text(metricString(result.averageRealtimeFactor))
                    .frame(width: 70, alignment: .trailing)
                    .monospacedDigit()

                Text(metricString(result.averageCanonicalWordErrorRate))
                    .frame(width: 70, alignment: .trailing)
                    .monospacedDigit()

                Text(metricString(result.averageElapsedSeconds))
                    .frame(width: 70, alignment: .trailing)
                    .monospacedDigit()
            }
            .font(.caption)

            if let deltaSummary {
                Text(deltaSummary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else if let detail = result.detail, result.status != .ok {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private var statusText: String {
        switch result.status {
        case .ok:
            return "OK"
        case .unavailable:
            return "Missing"
        case .failed:
            return "Failed"
        }
    }

    private var statusColor: Color {
        switch result.status {
        case .ok:
            return .green
        case .unavailable:
            return .orange
        case .failed:
            return .red
        }
    }

    private func metricString(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.2f", value)
    }

    private var deltaSummary: String? {
        guard let previousResult,
              result.status == .ok,
              previousResult.status == .ok else {
            return nil
        }

        let rtfDelta = deltaString(
            current: result.averageRealtimeFactor,
            previous: previousResult.averageRealtimeFactor,
            invertPreferred: false
        )
        let cwerDelta = deltaString(
            current: result.averageCanonicalWordErrorRate,
            previous: previousResult.averageCanonicalWordErrorRate,
            invertPreferred: true
        )

        if rtfDelta == nil && cwerDelta == nil {
            return nil
        }

        return [rtfDelta.map { "Δ RTF \($0)" }, cwerDelta.map { "Δ cWER \($0)" }]
            .compactMap { $0 }
            .joined(separator: " • ")
    }

    private func deltaString(current: Double?, previous: Double?, invertPreferred: Bool) -> String? {
        guard let current, let previous else { return nil }
        let delta = current - previous
        if abs(delta) < 0.005 {
            return "0.00"
        }

        let sign = delta >= 0 ? "+" : ""
        let formatted = "\(sign)\(String(format: "%.2f", delta))"
        if invertPreferred {
            return delta <= 0 ? "\(formatted) better" : "\(formatted) worse"
        }
        return delta >= 0 ? "\(formatted) better" : "\(formatted) worse"
    }
}
