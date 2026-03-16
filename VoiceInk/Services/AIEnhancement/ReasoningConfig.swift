import Foundation

struct ReasoningConfig {
    static let openAIReasoningModels: Set<String> = [
        "gpt-5.4"
    ]

    static func getReasoningParameter(for modelName: String) -> String? {
        openAIReasoningModels.contains(modelName) ? "low" : nil
    }
}
