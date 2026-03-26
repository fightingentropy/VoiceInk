import Foundation
import SwiftUI
import os

enum CohereTranscribeSetupError: LocalizedError {
    case pythonNotFound
    case workerScriptMissing
    case runtimeNotInstalled
    case missingAccessToken
    case invalidAccessToken
    case modelAccessNotGranted
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case .pythonNotFound:
            return "Python 3 could not be found. Install Python 3.10 or newer to use Cohere Transcribe locally."
        case .workerScriptMissing:
            return "The bundled Cohere Transcribe worker script is missing."
        case .runtimeNotInstalled:
            return "The Cohere Transcribe local runtime is not installed yet."
        case .missingAccessToken:
            return "A Hugging Face access token is required for this gated model."
        case .invalidAccessToken:
            return "The Hugging Face access token is invalid."
        case .modelAccessNotGranted:
            return "The token is valid, but access to Cohere Transcribe has not been granted on Hugging Face yet."
        case .commandFailed(let message):
            return message
        }
    }
}

@MainActor
final class CohereTranscribeEnvironmentManager: ObservableObject {
    static let shared = CohereTranscribeEnvironmentManager()

    enum InstallState: Equatable {
        case idle
        case installing
        case installed
        case failed(String)

        var errorMessage: String? {
            if case .failed(let message) = self {
                return message
            }
            return nil
        }
    }

    @Published private(set) var installState: InstallState = .idle
    @Published private(set) var hasAccessToken = false

    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "CohereTranscribeEnvironment")
    private let urlSession = URLSession(configuration: .default)

    private init() {
        hasAccessToken = APIKeyManager.shared.hasAPIKey(forProvider: LocalCohereTranscribeConfiguration.huggingFaceProviderName)
        refreshInstallState()
    }

    var runtimeDirectory: URL {
        AppStoragePaths.cohereTranscribeRuntimeDirectory
    }

    var cacheDirectory: URL {
        AppStoragePaths.cohereTranscribeCacheDirectory
    }

    var torchCompileCacheDirectory: URL {
        AppStoragePaths.cohereTranscribeTorchCompileCacheDirectory
    }

    var runtimeVersionFileURL: URL {
        runtimeDirectory.appendingPathComponent("runtime-version.txt")
    }

    var virtualEnvironmentDirectory: URL {
        runtimeDirectory.appendingPathComponent("venv", isDirectory: true)
    }

    var pythonExecutableURL: URL {
        virtualEnvironmentDirectory
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("python3", isDirectory: false)
    }

    var workerScriptURL: URL? {
        Bundle.main.url(forResource: LocalCohereTranscribeConfiguration.workerScriptName, withExtension: "py")
    }

    var modelAssetsDirectory: URL {
        AppStoragePaths.cohereTranscribeDirectory
    }

    var isRuntimeInstalled: Bool {
        guard FileManager.default.isExecutableFile(atPath: pythonExecutableURL.path) else {
            return false
        }

        guard let versionData = try? Data(contentsOf: runtimeVersionFileURL),
              let versionText = String(data: versionData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              versionText == String(LocalCohereTranscribeConfiguration.runtimeVersion) else {
            return false
        }

        return true
    }

    var hasHuggingFaceAccessToken: Bool {
        hasAccessToken
    }

    var isConfigured: Bool {
        isRuntimeInstalled && hasHuggingFaceAccessToken
    }

    func refreshInstallState() {
        installState = isRuntimeInstalled ? .installed : .idle
    }

    func verifyAndSaveAccessToken(_ token: String) async throws {
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw CohereTranscribeSetupError.missingAccessToken
        }

        var request = URLRequest(
            url: URL(string: "https://huggingface.co/\(LocalCohereTranscribeConfiguration.modelName)/resolve/main/config.json")!
        )
        request.setValue("Bearer \(trimmedToken)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 60
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        let (_, response) = try await urlSession.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CohereTranscribeSetupError.commandFailed("Hugging Face returned an invalid response while verifying access.")
        }

        switch httpResponse.statusCode {
        case 200:
            _ = APIKeyManager.shared.saveAPIKey(
                trimmedToken,
                forProvider: LocalCohereTranscribeConfiguration.huggingFaceProviderName
            )
            hasAccessToken = true
            postSetupDidChange()
        case 401:
            throw CohereTranscribeSetupError.invalidAccessToken
        case 403, 404:
            throw CohereTranscribeSetupError.modelAccessNotGranted
        default:
            throw CohereTranscribeSetupError.commandFailed(
                "Hugging Face access verification failed with HTTP \(httpResponse.statusCode)."
            )
        }
    }

    func clearAccessToken() {
        _ = APIKeyManager.shared.deleteAPIKey(forProvider: LocalCohereTranscribeConfiguration.huggingFaceProviderName)
        hasAccessToken = false
        postSetupDidChange()
    }

    func installRuntimeIfNeeded() async {
        guard installState != .installing else { return }
        guard !isRuntimeInstalled else {
            installState = .installed
            postSetupDidChange()
            return
        }

        installState = .installing

        do {
            guard let bootstrapPython = findBootstrapPython() else {
                throw CohereTranscribeSetupError.pythonNotFound
            }

            let fileManager = FileManager.default
            try AppStoragePaths.createDirectoryIfNeeded(at: AppStoragePaths.applicationSupportDirectory)
            try AppStoragePaths.createDirectoryIfNeeded(at: runtimeDirectory)
            try AppStoragePaths.createDirectoryIfNeeded(at: cacheDirectory)
            try AppStoragePaths.createDirectoryIfNeeded(at: torchCompileCacheDirectory)

            if fileManager.fileExists(atPath: virtualEnvironmentDirectory.path) {
                try fileManager.removeItem(at: virtualEnvironmentDirectory)
            }

            _ = try await runCommand(
                executableURL: bootstrapPython,
                arguments: ["-m", "venv", virtualEnvironmentDirectory.path]
            )

            let runtimePython = pythonExecutableURL
            let installEnvironment = [
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PYTHONNOUSERSITE": "1"
            ]

            _ = try await runCommand(
                executableURL: runtimePython,
                arguments: ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                environment: installEnvironment
            )

            _ = try await runCommand(
                executableURL: runtimePython,
                arguments: [
                    "-m", "pip", "install",
                    "torch",
                    "transformers>=4.56,<5.3,!=5.0.*,!=5.1.*",
                    "huggingface_hub",
                    "sentencepiece",
                    "protobuf",
                    "soundfile",
                    "librosa"
                ],
                environment: installEnvironment
            )

            try String(LocalCohereTranscribeConfiguration.runtimeVersion)
                .write(to: runtimeVersionFileURL, atomically: true, encoding: .utf8)

            installState = .installed
            postSetupDidChange()
        } catch {
            let message = error.localizedDescription
            logger.error("Cohere Transcribe runtime install failed: \(message, privacy: .public)")
            installState = .failed(message)
        }
    }

    func deleteManagedAssets() async {
        await CohereTranscribePythonRuntime.shared.shutdown()

        do {
            if FileManager.default.fileExists(atPath: modelAssetsDirectory.path) {
                try FileManager.default.removeItem(at: modelAssetsDirectory)
            }
            installState = .idle
            postSetupDidChange()
        } catch {
            let message = error.localizedDescription
            logger.error("Failed to delete Cohere Transcribe assets: \(message, privacy: .public)")
            installState = .failed(message)
        }
    }

    private func postSetupDidChange() {
        hasAccessToken = APIKeyManager.shared.hasAPIKey(
            forProvider: LocalCohereTranscribeConfiguration.huggingFaceProviderName
        )
        NotificationCenter.default.post(name: .cohereTranscribeSetupDidChange, object: nil)
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }

    private func findBootstrapPython() -> URL? {
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
            "/usr/bin/python3"
        ]

        return candidates
            .map(URL.init(fileURLWithPath:))
            .first(where: { FileManager.default.isExecutableFile(atPath: $0.path) })
    }

    private func runCommand(
        executableURL: URL,
        arguments: [String],
        environment: [String: String] = [:]
    ) async throws -> String {
        return try await Task.detached(priority: .userInitiated) {
            let task = Process()
            task.executableURL = executableURL
            task.arguments = arguments

            var mergedEnvironment = ProcessInfo.processInfo.environment
            environment.forEach { mergedEnvironment[$0.key] = $0.value }
            task.environment = mergedEnvironment

            let outputPipe = Pipe()
            task.standardOutput = outputPipe
            task.standardError = outputPipe

            try task.run()
            task.waitUntilExit()

            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

            guard task.terminationStatus == 0 else {
                throw CohereTranscribeSetupError.commandFailed(output.isEmpty ? "Command failed without output." : output)
            }

            return output
        }.value
    }
}
