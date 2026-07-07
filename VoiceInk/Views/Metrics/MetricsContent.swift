import SwiftUI
import SwiftData
import os

struct MetricsContent: View {
    private let logger = Logger(subsystem: "com.fightingentropy.VoiceInk", category: "MetricsContent")
    let modelContext: ModelContext

    @State private var totalCount: Int = 0
    @State private var totalWords: Int = 0
    @State private var totalDuration: TimeInterval = 0
    @State private var recentTranscriptions: [Transcription] = []
    @State private var isLoadingMetrics: Bool = true
    @State private var metricsTask: Task<Void, Never>?

    var body: some View {
        Group {
            if totalCount == 0 && !isLoadingMetrics {
                emptyStateView
            } else if isLoadingMetrics {
                ProgressView("Loading metrics...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 24) {
                        metricsSection

                        if !recentTranscriptions.isEmpty {
                            recentActivitySection
                        }
                    }
                    .padding(.vertical, 28)
                    .padding(.horizontal, 32)
                }
                .background(Color(.windowBackgroundColor))
            }
        }
        .task {
            await loadMetricsEfficiently()
        }
        .onReceive(NotificationCenter.default.publisher(for: .transcriptionCreated)) { _ in
            metricsTask?.cancel()
            metricsTask = Task {
                await loadMetricsEfficiently()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .transcriptionCompleted)) { _ in
            metricsTask?.cancel()
            metricsTask = Task {
                await loadMetricsEfficiently()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .transcriptionDeleted)) { _ in
            metricsTask?.cancel()
            metricsTask = Task {
                await loadMetricsEfficiently()
            }
        }
        .onDisappear {
            metricsTask?.cancel()
        }
    }
    
    private func loadMetricsEfficiently() async {
        await MainActor.run {
            self.isLoadingMetrics = true
        }

        let modelContainer = modelContext.container

        let backgroundContext = ModelContext(modelContainer)

        do {
            guard !Task.isCancelled else {
                await MainActor.run {
                    self.isLoadingMetrics = false
                }
                return
            }

            let completedFilter = #Predicate<Transcription> { $0.transcriptionStatus == "completed" }
            let count = try backgroundContext.fetchCount(FetchDescriptor<Transcription>(predicate: completedFilter))

            guard !Task.isCancelled else {
                await MainActor.run {
                    self.isLoadingMetrics = false
                }
                return
            }

            var descriptor = FetchDescriptor<Transcription>(predicate: completedFilter)
            descriptor.propertiesToFetch = [\.text, \.duration]

            var words = 0
            var duration: TimeInterval = 0

            try backgroundContext.enumerate(descriptor) { transcription in
                words += transcription.text.split(whereSeparator: \.isWhitespace).count
                duration += transcription.duration
            }

            guard !Task.isCancelled else {
                await MainActor.run {
                    self.isLoadingMetrics = false
                }
                return
            }

            await MainActor.run {
                self.totalCount = count
                self.totalWords = words
                self.totalDuration = duration
                self.recentTranscriptions = self.fetchRecentTranscriptions()
                self.isLoadingMetrics = false
            }
        } catch {
            logger.error("Error loading metrics: \(error.localizedDescription, privacy: .public)")
            await MainActor.run {
                self.isLoadingMetrics = false
            }
        }
    }

    @MainActor
    private func fetchRecentTranscriptions() -> [Transcription] {
        var descriptor = FetchDescriptor<Transcription>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = 5
        return (try? modelContext.fetch(descriptor)) ?? []
    }

    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Image(systemName: "waveform")
                .font(.system(size: 56, weight: .semibold))
                .foregroundColor(.secondary)
            Text("No Transcriptions Yet")
                .font(.title3.weight(.semibold))
            Text("Start your first recording to unlock value insights.")
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.windowBackgroundColor))
    }
    
    // MARK: - Sections
    
    private var metricsSection: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 16)], spacing: 16) {
            MetricCard(
                icon: "mic.fill",
                title: "Sessions Recorded",
                value: "\(totalCount)",
                detail: "VoiceInk sessions completed",
                color: .purple
            )

            MetricCard(
                icon: "text.alignleft",
                title: "Words Dictated",
                value: Formatters.formattedNumber(totalWords),
                detail: "words generated",
                color: Color(nsColor: .controlAccentColor)
            )
            
            MetricCard(
                icon: "speedometer",
                title: "Words Per Minute",
                value: averageWordsPerMinute > 0
                    ? String(format: "%.1f", averageWordsPerMinute)
                    : "–",
                detail: "VoiceInk vs. typing by hand",
                color: .yellow
            )
            
            MetricCard(
                icon: "keyboard.fill",
                title: "Keystrokes Saved",
                value: Formatters.formattedNumber(totalKeystrokesSaved),
                detail: "fewer keystrokes",
                color: .orange
            )
        }
    }
    
    private var recentActivitySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Activity")
                    .font(.headline)

                Spacer()

                Button("Show All") {
                    NotificationCenter.default.post(
                        name: .navigateToDestination,
                        object: nil,
                        userInfo: ["destination": "History"]
                    )
                }
                .buttonStyle(.borderless)
            }

            VStack(spacing: 0) {
                ForEach(recentTranscriptions) { transcription in
                    recentActivityRow(transcription)

                    if transcription.id != recentTranscriptions.last?.id {
                        Divider()
                    }
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(.thinMaterial)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.07), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.05), radius: 2, y: 1)
        }
    }

    private func recentActivityRow(_ transcription: Transcription) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(transcription.timestamp, format: .dateTime.month(.abbreviated).day().hour().minute())
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                if transcription.duration > 0 {
                    Text(transcription.duration.formatTiming())
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary, in: Capsule())
                }

                Spacer()
            }

            Text(transcription.text)
                .font(.system(size: 12))
                .foregroundStyle(.primary)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - Computed Metrics

    private var averageWordsPerMinute: Double {
        guard totalDuration > 0 else { return 0 }
        return Double(totalWords) / (totalDuration / 60.0)
    }

    private var totalKeystrokesSaved: Int {
        Int(Double(totalWords) * 5.0)
    }
}

private enum Formatters {
    static let numberFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter
    }()

    static func formattedNumber(_ value: Int) -> String {
        return numberFormatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }
}
