import SwiftUI

struct LocalVoxtralModelCardView: View {
    let model: LocalVoxtralModel
    let isCurrent: Bool
    var setDefaultAction: () -> Void

    @AppStorage(LocalVoxtralConfiguration.endpointKey) private var endpoint = LocalVoxtralConfiguration.defaultEndpoint
    @AppStorage(LocalVoxtralConfiguration.modelNameKey) private var modelName = LocalVoxtralConfiguration.defaultModelName
    @AppStorage(LocalVoxtralConfiguration.autoManageKey) private var autoManageServer = true

    @ObservedObject private var serverManager = LocalVoxtralServerManager.shared
    @State private var isExpanded = false

    private var isConfigured: Bool {
        !endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var computedLaunchCommand: String {
        LocalVoxtralConfiguration.launchCommand(endpointString: endpoint, modelName: modelName)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    headerSection
                    metadataSection
                    descriptionSection
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                actionSection
            }
            .padding(16)

            if isExpanded {
                Divider()
                    .padding(.horizontal, 16)

                configurationSection
                    .padding(16)
            }
        }
        .background(CardBackground(isSelected: isCurrent, useAccentGradientWhenSelected: isCurrent))
        .onAppear {
            Task {
                await serverManager.refreshStatus()
            }
        }
        .onChange(of: endpoint) { _, _ in
            Task {
                await serverManager.refreshStatus()
            }
        }
        .onChange(of: modelName) { _, _ in
            Task {
                await serverManager.refreshStatus()
            }
        }
    }

    private var headerSection: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(model.displayName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(.labelColor))

            statusBadge

            Spacer()
        }
    }

    private var statusBadge: some View {
        Group {
            if isCurrent {
                Text("Default")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.accentColor))
                    .foregroundColor(.white)
            } else if serverManager.isReady {
                Text(serverManager.status == .runningManaged ? "Managed" : "Ready")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemGreen).opacity(0.18)))
                    .foregroundColor(Color(.systemGreen))
            } else if case .starting = serverManager.status {
                Text("Starting")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.accentColor.opacity(0.16)))
                    .foregroundColor(Color.accentColor)
            } else if case .failed = serverManager.status {
                Text("Error")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemRed).opacity(0.14)))
                    .foregroundColor(Color(.systemRed))
            } else {
                Text("Local")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.systemOrange).opacity(0.16)))
                    .foregroundColor(Color(.systemOrange))
            }
        }
    }

    private var metadataSection: some View {
        HStack(spacing: 12) {
            Label(model.provider.rawValue, systemImage: "desktopcomputer")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            Label(model.language, systemImage: "globe")
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

            HStack(spacing: 3) {
                Text("Speed")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))
                progressDotsWithNumber(value: model.speed * 10)
            }
            .lineLimit(1)

            HStack(spacing: 3) {
                Text("Accuracy")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))
                progressDotsWithNumber(value: model.accuracy * 10)
            }
            .lineLimit(1)
        }
        .lineLimit(1)
    }

    private var descriptionSection: some View {
        Text(model.description)
            .font(.system(size: 11))
            .foregroundColor(Color(.secondaryLabelColor))
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 4)
    }

    private var actionSection: some View {
        HStack(spacing: 8) {
            if isCurrent {
                Text("Default Model")
                    .font(.system(size: 12))
                    .foregroundColor(Color(.secondaryLabelColor))
            } else if isConfigured {
                Button(action: setDefaultAction) {
                    Text("Set as Default")
                        .font(.system(size: 12))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Text("Setup Required")
                    .font(.system(size: 12))
                    .foregroundColor(Color(.secondaryLabelColor))
            }

            Button(action: {
                withAnimation(.interpolatingSpring(stiffness: 170, damping: 20)) {
                    isExpanded.toggle()
                }
            }) {
                Image(systemName: isExpanded ? "chevron.up.circle" : "gearshape.circle")
                    .font(.system(size: 16))
            }
            .buttonStyle(.plain)
            .foregroundColor(.secondary)
        }
    }

    private var configurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Local Server")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(.labelColor))

            TextField("Realtime Endpoint", text: $endpoint)
                .textFieldStyle(.roundedBorder)

            TextField("Model Name", text: $modelName)
                .textFieldStyle(.roundedBorder)

            Toggle("Automatically start and monitor the local server", isOn: $autoManageServer)
                .toggleStyle(.switch)

            VStack(alignment: .leading, spacing: 6) {
                Text("Start Command")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Color(.secondaryLabelColor))

                Text(computedLaunchCommand)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(Color(.secondaryLabelColor))
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color(.windowBackgroundColor).opacity(0.45))
                    )
            }

            HStack(spacing: 8) {
                Button("Copy Command") {
                    _ = ClipboardManager.copyToClipboard(computedLaunchCommand)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(action: startServer) {
                    HStack(spacing: 6) {
                        if case .starting = serverManager.status {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        Text(serverManager.isReady ? "Restart Server" : "Start Server")
                            .font(.system(size: 12, weight: .medium))
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(!isConfigured || !serverManager.canManageServer || serverManager.status == .starting)

                Button("Stop Server") {
                    serverManager.stopServer()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(serverManager.status != .runningManaged && serverManager.status != .starting)

                Button("Refresh Status") {
                    Task {
                        await serverManager.refreshStatus()
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Text("Run the voxmlx MLX server on this Mac for the lowest-latency Voxtral setup on Apple Silicon.")
                .font(.caption)
                .foregroundColor(.secondary)

            Text("Status: \(serverManager.status.title)")
                .font(.caption)
                .foregroundColor(.secondary)

            if !serverManager.canManageServer {
                Text("Automatic server management only works for localhost endpoints. Remote endpoints can still be used, but VoiceInk will not start them for you.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if let detail = serverManager.status.detail, !detail.isEmpty {
                Text(detail)
                    .font(.caption)
                    .foregroundColor(Color(.systemRed))
            } else if !serverManager.latestOutput.isEmpty {
                Text(serverManager.latestOutput)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func startServer() {
        Task {
            do {
                try await serverManager.startServer()
            } catch {
                _ = error
            }
        }
    }
}
