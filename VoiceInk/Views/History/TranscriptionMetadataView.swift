import SwiftUI

struct TranscriptionMetadataView: View {
    let transcription: Transcription

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Details")
                    .font(.system(size: 14, weight: .semibold))

                VStack(alignment: .leading, spacing: 8) {
                    metadataRow(
                        icon: "calendar",
                        label: "Date",
                        value: transcription.timestamp.formatted(date: .abbreviated, time: .shortened)
                    )

                    Divider()

                    metadataRow(
                        icon: "hourglass",
                        label: "Duration",
                        value: transcription.duration.formatTiming()
                    )

                    if let modelName = transcription.transcriptionModelName {
                        Divider()
                        metadataRow(
                            icon: "cpu.fill",
                            label: "Transcription Model",
                            value: modelName
                        )

                        if let duration = transcription.transcriptionDuration {
                            Divider()
                            metadataRow(
                                icon: "clock.fill",
                                label: "Transcription Time",
                                value: duration.formatTiming()
                            )
                        }
                    }

                }
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(.thinMaterial)
                )
            }
            .padding(12)
        }
        .background(Color(NSColor.controlBackgroundColor))
    }

    private func metadataRow(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.secondary)
                .frame(width: 20, height: 20)

            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)

            Spacer(minLength: 0)

            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.primary)
                .lineLimit(1)
        }
    }
}
