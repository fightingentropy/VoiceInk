import Foundation
import OSLog

class VADModelManager {
    static let shared = VADModelManager()
    private let logger = Logger(subsystem: "VADModelManager", category: "ModelManagement")
    
    private let modelURL = URL(string: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin")!
    private var modelPath: URL? {
        guard let appSupportDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        // Using the same directory structure as WhisperState for consistency
        let modelsDir = appSupportDir.appendingPathComponent("com.prakashjoshipax.VoiceInk/WhisperModels")
        return modelsDir.appendingPathComponent("ggml-silero-v5.1.2.bin")
    }

    private init() {
        if let modelPath = modelPath {
            let directory = modelPath.deletingLastPathComponent()
            if !FileManager.default.fileExists(atPath: directory.path) {
                do {
                    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
                    logger.log("Created directory for VAD model at \(directory.path)")
                } catch {
                    logger.error("Failed to create model directory: \(error.localizedDescription)")
                }
            }
        }
    }

    func getModelPath() async -> String? {
        guard let modelPath = modelPath else {
            logger.error("Could not construct VAD model path.")
            return nil
        }

        if FileManager.default.fileExists(atPath: modelPath.path) {
            logger.log("VAD model already exists at \(modelPath.path)")
            return modelPath.path
        } else {
            logger.log("VAD model not found, downloading...")
            return await downloadModel(to: modelPath)
        }
    }

    private func downloadModel(to path: URL) async -> String? {
        return await withCheckedContinuation { continuation in
            let task = URLSession.shared.downloadTask(with: modelURL) { location, response, error in
                DispatchQueue.main.async {
                    if let error = error {
                        self.logger.error("Failed to download VAD model: \(error.localizedDescription)")
                        continuation.resume(returning: nil)
                        return
                    }

                    guard let location = location else {
                        self.logger.error("Download location is nil.")
                        continuation.resume(returning: nil)
                        return
                    }

                    do {
                        // Ensure the destination directory exists
                        let directory = path.deletingLastPathComponent()
                        if !FileManager.default.fileExists(atPath: directory.path) {
                             try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
                        }
                        try FileManager.default.moveItem(at: location, to: path)
                        self.logger.log("Successfully downloaded and moved VAD model to \(path.path)")
                        continuation.resume(returning: path.path)
                    } catch {
                        self.logger.error("Failed to move VAD model to destination: \(error.localizedDescription)")
                        continuation.resume(returning: nil)
                    }
                }
            }
            task.resume()
        }
    }
} 