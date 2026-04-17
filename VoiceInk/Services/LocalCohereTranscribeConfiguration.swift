import Foundation

enum LocalCohereTranscribeConfiguration {
    // Default to the 4-bit MLX variant for ~60% smaller memory footprint
    // and noticeably faster inference on Apple Silicon vs the fp16 build,
    // with negligible WER impact for this 2B encoder-decoder.
    static let defaultRepository = "beshkenadze/cohere-transcribe-03-2026-mlx-4bit"
    static let defaultRevision = "104bc4391b5b1a12b040859793d7148525e1a08c"
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
