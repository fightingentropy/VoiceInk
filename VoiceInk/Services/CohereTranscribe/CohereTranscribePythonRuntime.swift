import Foundation
import os

actor CohereTranscribePythonRuntime {
    static let shared = CohereTranscribePythonRuntime()

    private struct WorkerRequest: Encodable {
        let id: String
        let command: String
        let model: String?
        let audioPath: String?
        let language: String?
        let preferCompile: Bool?
    }

    private struct WorkerEnvelope: Decodable {
        struct ResultPayload: Decodable {
            let text: String?
            let device: String?
            let compileEnabled: Bool?
            let model: String?
            let workerVersion: Int?

            private enum CodingKeys: String, CodingKey {
                case text
                case device
                case compileEnabled = "compile_enabled"
                case model
                case workerVersion = "worker_version"
            }
        }

        struct ErrorPayload: Decodable {
            let message: String
        }

        let id: String?
        let event: String?
        let ok: Bool?
        let result: ResultPayload?
        let error: ErrorPayload?
    }

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "CohereTranscribeRuntime")

    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutBuffer = Data()
    private var pendingRequests: [String: CheckedContinuation<WorkerEnvelope, Error>] = [:]
    private var startupContinuation: CheckedContinuation<Void, Error>?

    private init() {}

    func prepareModel() async throws {
        try await ensureProcessStarted()
        _ = try await sendRequest(
            WorkerRequest(
                id: UUID().uuidString,
                command: "load",
                model: LocalCohereTranscribeConfiguration.modelName,
                audioPath: nil,
                language: nil,
                preferCompile: true
            )
        )
    }

    func warmup(audioURL: URL, language: String) async throws {
        try await ensureProcessStarted()
        _ = try await sendRequest(
            WorkerRequest(
                id: UUID().uuidString,
                command: "warmup",
                model: LocalCohereTranscribeConfiguration.modelName,
                audioPath: audioURL.path,
                language: language,
                preferCompile: true
            )
        )
    }

    func transcribe(audioURL: URL, language: String) async throws -> String {
        try await ensureProcessStarted()
        let response = try await sendRequest(
            WorkerRequest(
                id: UUID().uuidString,
                command: "transcribe",
                model: LocalCohereTranscribeConfiguration.modelName,
                audioPath: audioURL.path,
                language: language,
                preferCompile: true
            )
        )

        return response.result?.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    func shutdown() async {
        cleanupProcess()
    }

    private func ensureProcessStarted() async throws {
        if let process, process.isRunning {
            return
        }

        let isInstalled = await MainActor.run { CohereTranscribeEnvironmentManager.shared.isRuntimeInstalled }
        guard isInstalled else {
            throw CohereTranscribeSetupError.runtimeNotInstalled
        }

        let scriptURL = await MainActor.run {
            CohereTranscribeEnvironmentManager.shared.workerScriptURL
        }
        guard let scriptURL else {
            throw CohereTranscribeSetupError.workerScriptMissing
        }

        let pythonExecutableURL = await MainActor.run {
            CohereTranscribeEnvironmentManager.shared.pythonExecutableURL
        }
        let cacheDirectory = await MainActor.run {
            CohereTranscribeEnvironmentManager.shared.cacheDirectory
        }
        let torchCompileCacheDirectory = await MainActor.run {
            CohereTranscribeEnvironmentManager.shared.torchCompileCacheDirectory
        }
        let accessToken = await MainActor.run {
            APIKeyManager.shared.getAPIKey(forProvider: LocalCohereTranscribeConfiguration.huggingFaceProviderName)
        }

        guard let accessToken, !accessToken.isEmpty else {
            throw CohereTranscribeSetupError.missingAccessToken
        }

        try await withCheckedThrowingContinuation { continuation in
            startupContinuation = continuation

            do {
                try AppStoragePaths.createDirectoryIfNeeded(at: cacheDirectory)
                try AppStoragePaths.createDirectoryIfNeeded(at: torchCompileCacheDirectory)

                let process = Process()
                process.executableURL = pythonExecutableURL
                process.arguments = ["-u", scriptURL.path]

                var childEnvironment = ProcessInfo.processInfo.environment
                childEnvironment["HF_HOME"] = cacheDirectory.path
                childEnvironment["HF_TOKEN"] = accessToken
                childEnvironment["PYTHONUNBUFFERED"] = "1"
                childEnvironment["PYTHONNOUSERSITE"] = "1"
                childEnvironment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                childEnvironment["TORCHINDUCTOR_CACHE_DIR"] = torchCompileCacheDirectory.path
                process.environment = childEnvironment

                let stdinPipe = Pipe()
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()

                process.standardInput = stdinPipe
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe

                stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                    let data = handle.availableData
                    guard !data.isEmpty else { return }
                    Task {
                        await self?.consumeStandardOutput(data)
                    }
                }

                stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                    let data = handle.availableData
                    guard !data.isEmpty,
                          let line = String(data: data, encoding: .utf8)?
                            .trimmingCharacters(in: .whitespacesAndNewlines),
                          !line.isEmpty else { return }
                    Task {
                        await self?.logWorkerStderr(line)
                    }
                }

                process.terminationHandler = { [weak self] terminatedProcess in
                    Task {
                        await self?.handleProcessExit(status: terminatedProcess.terminationStatus)
                    }
                }

                try process.run()
                self.process = process
                self.stdinHandle = stdinPipe.fileHandleForWriting
            } catch {
                startupContinuation = nil
                continuation.resume(throwing: error)
            }
        }
    }

    private func sendRequest(_ request: WorkerRequest) async throws -> WorkerEnvelope {
        return try await withCheckedThrowingContinuation { continuation in
            pendingRequests[request.id] = continuation

            do {
                let encoded = try JSONEncoder().encode(request)
                guard let stdinHandle else {
                    pendingRequests.removeValue(forKey: request.id)
                    continuation.resume(throwing: CohereTranscribeSetupError.runtimeNotInstalled)
                    return
                }

                var payload = encoded
                payload.append(0x0A)
                try stdinHandle.write(contentsOf: payload)
            } catch {
                pendingRequests.removeValue(forKey: request.id)
                continuation.resume(throwing: error)
            }
        }
    }

    private func consumeStandardOutput(_ data: Data) {
        stdoutBuffer.append(data)

        while let newlineRange = stdoutBuffer.range(of: Data([0x0A])) {
            let lineData = stdoutBuffer.subdata(in: 0..<newlineRange.lowerBound)
            stdoutBuffer.removeSubrange(0..<newlineRange.upperBound)

            guard !lineData.isEmpty else { continue }
            handleWorkerLine(lineData)
        }
    }

    private func handleWorkerLine(_ lineData: Data) {
        do {
            let envelope = try JSONDecoder().decode(WorkerEnvelope.self, from: lineData)

            if envelope.event == "ready" {
                startupContinuation?.resume()
                startupContinuation = nil
                return
            }

            guard let requestID = envelope.id else { return }

            guard let continuation = pendingRequests.removeValue(forKey: requestID) else { return }
            if envelope.ok == true {
                continuation.resume(returning: envelope)
            } else {
                continuation.resume(
                    throwing: CohereTranscribeSetupError.commandFailed(
                        envelope.error?.message ?? "Cohere Transcribe worker returned an unknown error."
                    )
                )
            }
        } catch {
            logger.error("Failed to decode Cohere worker response: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func logWorkerStderr(_ line: String) {
        logger.error("Cohere worker stderr: \(line, privacy: .public)")
    }

    private func handleProcessExit(status: Int32) {
        let error = CohereTranscribeSetupError.commandFailed(
            "Cohere Transcribe worker exited with status \(status)."
        )

        startupContinuation?.resume(throwing: error)
        startupContinuation = nil

        for (_, continuation) in pendingRequests {
            continuation.resume(throwing: error)
        }
        pendingRequests.removeAll()

        cleanupProcess()
    }

    private func cleanupProcess() {
        process?.standardOutput = nil
        process?.standardError = nil

        if let process, process.isRunning {
            process.terminate()
        }

        process = nil
        stdinHandle = nil
        stdoutBuffer.removeAll(keepingCapacity: false)
    }
}
