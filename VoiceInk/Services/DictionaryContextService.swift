import Foundation
import SwiftUI

class DictionaryContextService {
    static let shared = DictionaryContextService()
    
    private init() {}
    
    /// Gets dictionary context information to be included in AI enhancement
    func getDictionaryContext() -> String {
        guard let dictionaryWords = getDictionaryWords(), !dictionaryWords.isEmpty else {
            return ""
        }
        
        let wordsText = dictionaryWords.joined(separator: ", ")
        return "Important Vocabulary: \(wordsText)"
    }
    
    /// Gets all custom dictionary words from UserDefaults
    private func getDictionaryWords() -> [String]? {
        guard let data = UserDefaults.standard.data(forKey: "CustomDictionaryItems") else {
            return nil
        }
        
        do {
            let items = try JSONDecoder().decode([DictionaryItem].self, from: data)
            let words = items.map { $0.word }
            return words.isEmpty ? nil : words
        } catch {
            return nil
        }
    }
}
