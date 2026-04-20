import Foundation
import SwiftData
import OSLog

final class DictionaryMigrationService: @unchecked Sendable {
    static let shared = DictionaryMigrationService()
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "DictionaryMigration")

    private let migrationCompletedKey = "HasMigratedDictionaryToSwiftData_v2"
    private let wordReplacementsKey = "wordReplacements"

    private init() {}

    /// Migrates dictionary data from UserDefaults to SwiftData
    /// This is a one-time operation that preserves all existing user data
    func migrateIfNeeded(context: ModelContext) {
        // Check if migration has already been completed
        if UserDefaults.standard.bool(forKey: migrationCompletedKey) {
            logger.info("Dictionary migration already completed, skipping")
            return
        }

        logger.info("Starting dictionary migration from UserDefaults to SwiftData")

        var replacementsMigrated = 0

        // Migrate word replacements
        if let replacements = UserDefaults.standard.dictionary(forKey: wordReplacementsKey) as? [String: String] {
            logger.info("Found \(replacements.count, privacy: .public) word replacements to migrate")

            for (originalText, replacementText) in replacements {
                let wordReplacement = WordReplacement(
                    originalText: originalText,
                    replacementText: replacementText
                )
                context.insert(wordReplacement)
                replacementsMigrated += 1
            }

            logger.info("Successfully migrated \(replacementsMigrated, privacy: .public) word replacements")
        } else {
            logger.info("No word replacements found to migrate")
        }

        // Save the migrated data
        do {
            try context.save()
            logger.info("Successfully saved migrated data to SwiftData")

            // Mark migration as completed
            UserDefaults.standard.set(true, forKey: migrationCompletedKey)
            logger.info("Migration completed successfully")
        } catch {
            logger.error("Failed to save migrated data: \(error.localizedDescription, privacy: .public)")
        }
    }
}
