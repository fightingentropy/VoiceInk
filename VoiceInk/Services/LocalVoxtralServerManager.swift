import Foundation
import os

@MainActor
final class LocalVoxtralServerManager: ObservableObject {
    static let shared = LocalVoxtralServerManager()

    enum Status: Equatable {
        case stopped
        case starting
        case runningManaged
        case runningExternal
        case failed(String)

        var title: String {
            switch self {
            case .stopped:
                return "Stopped"
            case .starting:
                return "Starting"
            case .runningManaged:
                return "Running (Managed)"
            case .runningExternal:
                return "Running (External)"
            case .failed:
                return "Failed"
            }
        }

        var detail: String? {
            switch self {
            case .failed(let message):
                return message
            default:
                return nil
            }
        }
    }

    @Published private(set) var status: Status = .stopped
    @Published private(set) var latestOutput: String = ""

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "LocalVoxtralServerManager")
    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var isStopping = false
    private var statusTask: Task<Void, Never>?

    private init() {}

    var isReady: Bool {
        switch status {
        case .runningManaged, .runningExternal:
            return true
        default:
            return false
        }
    }

    var canManageServer: Bool {
        LocalVoxtralConfiguration.serverBinding(from: LocalVoxtralConfiguration.endpointString) != nil
    }

    var launchCommand: String {
        LocalVoxtralConfiguration.launchCommand()
    }

    func refreshStatus() async {
        if await isHealthCheckPassing() {
            status = process?.isRunning == true ? .runningManaged : .runningExternal
            return
        }

        if process?.isRunning == true {
            status = .starting
        } else if case .failed = status {
            // Preserve the failure message until a successful refresh/start.
        } else {
            status = .stopped
        }
    }

    func prepareServerIfNeeded() async throws {
        if await isHealthCheckPassing() {
            status = process?.isRunning == true ? .runningManaged : .runningExternal
            return
        }

        guard LocalVoxtralConfiguration.autoManageServer else {
            throw StreamingTranscriptionError.connectionFailed(
                "Local Voxtral server is not reachable. Start it from the model settings or enable automatic server management."
            )
        }

        try await startServer()
    }

    func startServer() async throws {
        if await isHealthCheckPassing() {
            status = process?.isRunning == true ? .runningManaged : .runningExternal
            return
        }

        guard canManageServer else {
            throw StreamingTranscriptionError.connectionFailed(
                "Automatic Local Voxtral server management only supports localhost endpoints."
            )
        }

        if let process, process.isRunning {
            isStopping = true
            process.terminate()
            cleanupProcessResources()
            try? await Task.sleep(nanoseconds: 300_000_000)
            isStopping = false
        }

        latestOutput = ""
        status = .starting
        isStopping = false

        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", launchCommand]
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.terminationHandler = { [weak self] process in
            Task { @MainActor in
                self?.handleProcessTermination(process)
            }
        }

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let self else { return }
            Task { @MainActor in
                self.appendOutput(from: data)
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let self else { return }
            Task { @MainActor in
                self.appendOutput(from: data)
            }
        }

        do {
            try process.run()
        } catch {
            cleanupProcessResources()
            status = .failed(error.localizedDescription)
            throw StreamingTranscriptionError.connectionFailed(error.localizedDescription)
        }

        self.process = process
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe

        try await waitForHealthyServer(timeoutSeconds: 30)
        startStatusMonitoring()
    }

    func stopServer() {
        guard let process else {
            isStopping = false
            status = .stopped
            return
        }

        isStopping = true
        statusTask?.cancel()
        statusTask = nil

        if process.isRunning {
            process.terminate()
        } else {
            cleanupProcessResources()
            isStopping = false
            status = .stopped
        }
    }

    func shutdownManagedServer() {
        isStopping = true
        statusTask?.cancel()
        statusTask = nil
        if process?.isRunning == true {
            process?.terminate()
        }
        cleanupProcessResources()
    }

    // MARK: - Private

    private func waitForHealthyServer(timeoutSeconds: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeoutSeconds)

        while Date() < deadline {
            if await isHealthCheckPassing() {
                status = .runningManaged
                return
            }

            if let process, !process.isRunning {
                let message = latestOutput.trimmingCharacters(in: .whitespacesAndNewlines)
                let detail = message.isEmpty ? "voxmlx exited before it became ready." : message
                status = .failed(detail)
                throw StreamingTranscriptionError.connectionFailed(detail)
            }

            try? await Task.sleep(nanoseconds: 500_000_000)
        }

        let message = latestOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = message.isEmpty ? "Timed out waiting for Local Voxtral to become ready." : message
        status = .failed(detail)
        throw StreamingTranscriptionError.connectionFailed(detail)
    }

    private func startStatusMonitoring() {
        statusTask?.cancel()
        statusTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard !Task.isCancelled else { return }
                await self.refreshStatus()
            }
        }
    }

    private func isHealthCheckPassing() async -> Bool {
        guard let url = LocalVoxtralConfiguration.healthCheckURL(from: LocalVoxtralConfiguration.endpointString) else {
            return false
        }

        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 2
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }
            return (200...299).contains(httpResponse.statusCode)
        } catch {
            return false
        }
    }

    private func appendOutput(from data: Data) {
        guard let text = String(data: data, encoding: .utf8), !text.isEmpty else { return }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        logger.notice("Local Voxtral: \(trimmed, privacy: .public)")
        let joined = latestOutput.isEmpty ? trimmed : "\(latestOutput)\n\(trimmed)"
        latestOutput = String(joined.suffix(2_000))
    }

    private func handleProcessTermination(_ process: Process) {
        guard self.process === process else { return }

        let output = latestOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        cleanupProcessResources()
        statusTask?.cancel()
        statusTask = nil

        if isStopping {
            isStopping = false
            status = .stopped
            return
        }

        let message = output.isEmpty ? "Local Voxtral server stopped unexpectedly." : output
        status = .failed(message)
    }

    private func cleanupProcessResources() {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdoutPipe = nil
        stderrPipe = nil
        process = nil
    }
}
