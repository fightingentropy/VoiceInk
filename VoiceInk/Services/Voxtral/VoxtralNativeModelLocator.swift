import Foundation

enum VoxtralNativeModelLocator {
    enum Availability: Equatable {
        case appManaged(URL)
        case externalLocalPath(URL)
        case missing

        var directoryURL: URL? {
            switch self {
            case .appManaged(let url), .externalLocalPath(let url):
                return url
            case .missing:
                return nil
            }
        }
    }

    private static let requiredBaseFiles = [
        "config.json",
        "tekken.json",
    ]

    static func availability(for modelReference: String) -> Availability {
        if let externalPath = existingLocalPath(from: modelReference), isModelDirectoryComplete(externalPath) {
            return .externalLocalPath(externalPath)
        }

        if let repoID = repositoryID(from: modelReference) {
            let appManaged = appManagedModelDirectory(for: repoID)
            if isModelDirectoryComplete(appManaged) {
                return .appManaged(appManaged)
            }
        }

        return .missing
    }

    static func repositoryID(from modelReference: String) -> String? {
        let trimmed = modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard !trimmed.hasPrefix("/") && !trimmed.hasPrefix("~") && !trimmed.hasPrefix(".") else {
            return nil
        }

        let parts = trimmed.split(separator: "/")
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty }) else { return nil }
        return trimmed
    }

    static func appManagedModelDirectory(for repoID: String) -> URL {
        let slug = repoID.replacingOccurrences(of: "/", with: "--")
        return AppStoragePaths.voxtralModelsDirectory
            .appendingPathComponent(slug, isDirectory: true)
    }

    static func existingLocalPath(from modelReference: String) -> URL? {
        let expanded = NSString(string: modelReference).expandingTildeInPath
        guard expanded != modelReference || modelReference.hasPrefix("~") || modelReference.hasPrefix("/") || modelReference.hasPrefix(".") else {
            return nil
        }

        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }
        return url
    }

    static func isModelDirectoryComplete(_ directory: URL) -> Bool {
        guard requiredBaseFiles.allSatisfy({ fileExists(named: $0, in: directory) }) else {
            return false
        }

        let shardFiles = expectedWeightFiles(in: directory)
        return !shardFiles.isEmpty && shardFiles.allSatisfy { fileExists(named: $0, in: directory) }
    }

    static func expectedWeightFiles(in directory: URL) -> [String] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")

        if let indexData = try? Data(contentsOf: indexURL),
           let index = try? JSONDecoder().decode(VoxtralModelWeightIndex.self, from: indexData) {
            let shardFiles = Set(index.weightMap.values)
            if !shardFiles.isEmpty {
                return Array(shardFiles).sorted()
            }
        }

        return fileExists(named: "model.safetensors", in: directory) ? ["model.safetensors"] : []
    }

    private static func fileExists(named name: String, in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
    }
}

private struct VoxtralModelWeightIndex: Decodable {
    let weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
