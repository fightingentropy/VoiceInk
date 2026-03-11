import Foundation

struct ReasoningConfig {
    static let geminiReasoningModels: Set<String> = [
        "gemini-2.5-flash"
    ]

    static let openAIReasoningModels: Set<String> = [
        "gpt-5.4"
    ]

    static let cerebrasReasoningModels: Set<String> = [
        "gpt-oss-120b"
    ]

    static func getReasoningParameter(for modelName: String) -> String? {
        if geminiReasoningModels.contains(modelName) {
            return "low"
        } else if openAIReasoningModels.contains(modelName) {
            return "low"
        } else if cerebrasReasoningModels.contains(modelName) {
            return "low"
        }
        return nil
    }
}
