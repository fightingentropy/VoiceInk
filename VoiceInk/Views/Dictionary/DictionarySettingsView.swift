import SwiftUI
import SwiftData

struct DictionarySettingsView: View {
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                heroSection
                wordReplacementsContent
            }
        }
        .frame(minWidth: 600, minHeight: 500)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private var heroSection: some View {
        CompactHeroSection(
            icon: "brain.filled.head.profile",
            title: "Dictionary Settings",
            description: "Automatically replace specific words or phrases after transcription.",
            maxDescriptionWidth: 500
        )
    }
    
    private var wordReplacementsContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("Word Replacements")
                    .font(.title2)
                    .fontWeight(.semibold)

                Spacer()

                HStack(spacing: 12) {
                    Button(action: {
                        DictionaryImportExportService.shared.importDictionary(into: modelContext)
                    }) {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 18))
                            .foregroundColor(.blue)
                    }
                    .buttonStyle(.plain)
                    .help("Import word replacements")

                    Button(action: {
                        DictionaryImportExportService.shared.exportDictionary(from: modelContext)
                    }) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 18))
                            .foregroundColor(.blue)
                    }
                    .buttonStyle(.plain)
                    .help("Export word replacements")
                }
            }

            WordReplacementView()
                .background(CardBackground(isSelected: false))
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 40)
    }
}
