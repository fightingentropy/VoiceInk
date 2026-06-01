import Foundation

enum LocalCohereTranscribeConfiguration {
    // Default to the 8-bit MLX variant: it keeps a substantially smaller
    // footprint than fp16 while preserving fp16 parity on the repository sample.
    static let defaultRepository = "beshkenadze/cohere-transcribe-03-2026-mlx-8bit"
    static let defaultRevision = "d1f843476f84846e6fe7aa58a6033f17882f0ec9"
    static let fallbackLanguage = "en"

    // UserDefaults keys let power users pin to a different MLX build
    // (e.g. the fp16 or 6-bit variant) without shipping a new release.
    static let repositoryOverrideKey = "LocalCohereTranscribeRepository"
    static let revisionOverrideKey = "LocalCohereTranscribeRevision"

    static var nativeModelRepository: String {
        let stored = UserDefaults.standard.string(forKey: repositoryOverrideKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (stored?.isEmpty == false) ? stored! : defaultRepository
    }

    static var nativeModelRevision: String {
        let stored = UserDefaults.standard.string(forKey: revisionOverrideKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (stored?.isEmpty == false) ? stored! : defaultRevision
    }
}
