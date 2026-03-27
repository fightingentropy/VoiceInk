import Foundation

enum CohereNativePromptBuilder {
    static let startOfContextToken = "<|startofcontext|>"
    static let startOfTranscriptToken = "<|startoftranscript|>"
    static let endOfTextToken = "<|endoftext|>"
    static let noSpeechToken = "<|nospeech|>"
    static let undefinedEmotionToken = "<|emo:undefined|>"
    static let noInverseTextNormalizationToken = "<|noitn|>"
    static let noTimestampToken = "<|notimestamp|>"
    static let noDiarizationToken = "<|nodiarize|>"
    static let punctuationToken = "<|pnc|>"
    static let noPunctuationToken = "<|nopnc|>"
    private static let noSpaceLanguages: Set<String> = ["ja", "zh"]

    static func buildPrompt(language: String, punctuation: Bool = true) -> String {
        let resolvedLanguage = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let punctuationMarker = punctuation ? punctuationToken : noPunctuationToken

        return [
            startOfContextToken,
            startOfTranscriptToken,
            undefinedEmotionToken,
            languageToken(for: resolvedLanguage),
            languageToken(for: resolvedLanguage),
            punctuationMarker,
            noInverseTextNormalizationToken,
            noTimestampToken,
            noDiarizationToken
        ].joined()
    }

    static func requiresLeadingSpace(language: String) -> Bool {
        !noSpaceLanguages.contains(language.lowercased())
    }

    private static func languageToken(for language: String) -> String {
        "<|\(language)|>"
    }
}
