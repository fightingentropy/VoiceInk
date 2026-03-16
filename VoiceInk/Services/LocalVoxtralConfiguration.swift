import Foundation

enum LocalVoxtralConfiguration {
    static let modelNameKey = "LocalVoxtralModelName"
    static let defaultModelName = "T0mSIlver/Voxtral-Mini-4B-Realtime-2602-MLX-4bit"

    static var modelName: String {
        let stored = UserDefaults.standard.string(forKey: modelNameKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return stored?.isEmpty == false ? stored! : defaultModelName
    }

    static var resolvedModelReference: String {
        VoxtralNativeModelLocator.resolvedModelReference(for: modelName)
    }
}
