import SwiftUI

struct DiagnosticsSettingsView: View {
    @State private var isExportingLogs = false
    @State private var exportedLogURL: URL?
    @State private var showLogExportError = false
    @State private var logExportError: String = ""
    @State private var isSystemInfoCopied = false

    var body: some View {
        LabeledContent {
            HStack(spacing: 8) {
                if let url = exportedLogURL {
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }

                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                }

                Button("Export") {
                    exportDiagnosticLogs()
                }
                .disabled(isExportingLogs)
            }
        } label: {
            HStack(spacing: 4) {
                if isExportingLogs {
                    ProgressView()
                        .controlSize(.small)
                }
                Text("Export Logs")
            }
        }
        .alert("Export Failed", isPresented: $showLogExportError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(logExportError)
        }

        LabeledContent("System Info") {
            Button {
                copySystemInfo()
            } label: {
                if isSystemInfoCopied {
                    Label("Copied", systemImage: "checkmark")
                } else {
                    Text("Copy")
                }
            }
            .buttonStyle(.bordered)
        }
    }

    private func copySystemInfo() {
        SystemInfoService.shared.copySystemInfoToClipboard()
        isSystemInfoCopied = true

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            isSystemInfoCopied = false
        }
    }

    private func exportDiagnosticLogs() {
        isExportingLogs = true
        exportedLogURL = nil

        Task {
            do {
                let url = try await LogExporter.shared.exportLogs()
                await MainActor.run {
                    exportedLogURL = url
                    isExportingLogs = false
                }
            } catch {
                await MainActor.run {
                    logExportError = error.localizedDescription
                    showLogExportError = true
                    isExportingLogs = false
                }
            }
        }
    }
}
