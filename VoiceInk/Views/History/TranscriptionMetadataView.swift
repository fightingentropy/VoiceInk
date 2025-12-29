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
                        value: formatTiming(transcription.duration)
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
                                value: formatTiming(duration)
                            )
                        }
                    }

                    if let aiModel = transcription.aiEnhancementModelName {
                        Divider()
                        metadataRow(
                            icon: "sparkles",
                            label: "Enhancement Model",
                            value: aiModel
                        )

                        if let duration = transcription.enhancementDuration {
                            Divider()
                            metadataRow(
                                icon: "clock.fill",
                                label: "Enhancement Time",
                                value: formatTiming(duration)
                            )
                        }
                    }

                    if let promptName = transcription.promptName {
                        Divider()
                        metadataRow(
                            icon: "text.bubble.fill",
                            label: "Prompt",
                            value: promptName
                        )
                    }

                    if let powerModeValue = powerModeDisplay(
                        name: transcription.powerModeName,
                        emoji: transcription.powerModeEmoji
                    ) {
                        Divider()
                        metadataRow(
                            icon: "bolt.fill",
                            label: "Power Mode",
                            value: powerModeValue
                        )
                    }
                }
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(.thinMaterial)
                )

                if transcription.aiRequestSystemMessage != nil || transcription.aiRequestUserMessage != nil {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("AI Request")
                            .font(.system(size: 14, weight: .semibold))

                        ScrollView {
                            VStack(alignment: .leading, spacing: 12) {
                                if let systemMsg = transcription.aiRequestSystemMessage, !systemMsg.isEmpty {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text("System Prompt")
                                            .font(.system(size: 11, weight: .semibold))
                                            .foregroundColor(.secondary)
                                        Text(systemMsg)
                                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                                            .lineSpacing(2)
                                            .textSelection(.enabled)
                                            .foregroundColor(.primary)
                                    }
                                }

                                if let userMsg = transcription.aiRequestUserMessage, !userMsg.isEmpty {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text("User Message")
                                            .font(.system(size: 11, weight: .semibold))
                                            .foregroundColor(.secondary)
                                        Text(userMsg)
                                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                                            .lineSpacing(2)
                                            .textSelection(.enabled)
                                            .foregroundColor(.primary)
                                    }
                                }
                            }
                            .padding(14)
                        }
                        .frame(minHeight: 250, maxHeight: 500)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(.thinMaterial)
                        )
                    }
                }
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

    private func formatTiming(_ duration: TimeInterval) -> String {
        if duration < 1 {
            return String(format: "%.0fms", duration * 1000)
        }
        if duration < 60 {
            return String(format: "%.1fs", duration)
        }
        let minutes = Int(duration) / 60
        let seconds = duration.truncatingRemainder(dividingBy: 60)
        return String(format: "%dm %.0fs", minutes, seconds)
    }

    private func powerModeDisplay(name: String?, emoji: String?) -> String? {
        guard name != nil || emoji != nil else { return nil }

        switch (emoji?.trimmingCharacters(in: .whitespacesAndNewlines), name?.trimmingCharacters(in: .whitespacesAndNewlines)) {
        case let (.some(emojiValue), .some(nameValue)) where !emojiValue.isEmpty && !nameValue.isEmpty:
            return "\(emojiValue) \(nameValue)"
        case let (.some(emojiValue), _) where !emojiValue.isEmpty:
            return emojiValue
        case let (_, .some(nameValue)) where !nameValue.isEmpty:
            return nameValue
        default:
            return nil
        }
    }
}
