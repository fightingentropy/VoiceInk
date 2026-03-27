import Foundation

enum CohereNativeModelLocator {
    enum Availability: Equatable {
        case appManaged(URL)
        case externalLocalPath(URL)
        case missing
    }

    static func availability(for modelReference: String) -> Availability {
        let trimmed = modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .missing }

        let localURL = URL(fileURLWithPath: trimmed)
        if FileManager.default.fileExists(atPath: localURL.path),
           isModelDirectoryComplete(localURL) {
            return .externalLocalPath(localURL)
        }

        guard let repositoryID = repositoryID(from: trimmed) else {
            return .missing
        }

        let managedDirectory = appManagedModelDirectory(for: repositoryID)
        return isModelDirectoryComplete(managedDirectory) ? .appManaged(managedDirectory) : .missing
    }

    static func repositoryID(from modelReference: String) -> String? {
        let trimmed = modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.contains("/"), !trimmed.contains("://"), !trimmed.hasPrefix("/") else {
            return nil
        }
        return trimmed
    }

    static func appManagedModelDirectory(for repositoryID: String) -> URL {
        AppStoragePaths.cohereTranscribeNativeDirectory
            .appendingPathComponent(repositoryID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }

    static func isModelDirectoryComplete(_ directory: URL) -> Bool {
        let requiredFiles = [
            "config.json",
            "tokenizer.model",
            "model.safetensors"
        ]

        return requiredFiles.allSatisfy { filename in
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(filename).path
            )
        }
    }
}
