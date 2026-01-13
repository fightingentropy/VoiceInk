import SwiftUI

struct ExperimentalFeaturesSection: View {
    @AppStorage("isExperimentalFeaturesEnabled") private var isExperimentalFeaturesEnabled = false
    @ObservedObject private var playbackController = PlaybackController.shared
    @ObservedObject private var mediaController = MediaController.shared
    @State private var expandedSections: Set<ExpandableSection> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Compact header
            HStack(spacing: 8) {
                Image(systemName: "flask")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
                    .frame(width: 16, height: 16)

                Text("Experimental")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)

                Spacer()

                Toggle("", isOn: $isExperimentalFeaturesEnabled)
                    .toggleStyle(.switch)
                    .scaleEffect(0.8)
                    .onChange(of: isExperimentalFeaturesEnabled) { _, newValue in
                        if !newValue {
                            playbackController.isPauseMediaEnabled = false
                        }
                    }
            }
            .padding(.horizontal, 12)
            .padding(.top, 10)
            .padding(.bottom, 8)

            // Content
            VStack(alignment: .leading, spacing: 6) {
                Text("Features that may be unstable.")
                    .font(.system(size: 11))
                    .foregroundColor(Color(NSColor.tertiaryLabelColor))

                if isExperimentalFeaturesEnabled {
                    ExpandableToggleSection(
                        section: .pauseMedia,
                        title: "Pause Media",
                        helpText: "Pause playback during recording",
                        isEnabled: $playbackController.isPauseMediaEnabled,
                        expandedSections: $expandedSections
                    ) {
                        HStack(spacing: 6) {
                            Text("Resume")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                            Picker("", selection: $mediaController.audioResumptionDelay) {
                                Text("0s").tag(0.0)
                                Text("1s").tag(1.0)
                                Text("2s").tag(2.0)
                                Text("3s").tag(3.0)
                                Text("4s").tag(4.0)
                                Text("5s").tag(5.0)
                            }
                            .pickerStyle(.menu)
                            .frame(width: 60)
                            .scaleEffect(0.9)
                            Spacer()
                        }
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 12)
        }
        .animation(.easeInOut(duration: 0.2), value: isExperimentalFeaturesEnabled)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(NSColor.controlBackgroundColor).opacity(0.5))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color(NSColor.separatorColor).opacity(0.5), lineWidth: 0.5)
        )
    }
}
