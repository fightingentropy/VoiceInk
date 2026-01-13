import SwiftUI

enum ExpandableSection: Hashable {
    case soundFeedback
    case systemMute
    case pauseMedia
    case clipboardRestore
    case customCancel
    case middleClick
}

struct ExpandableToggleSection<Content: View>: View {
    let section: ExpandableSection
    let title: String
    let helpText: String
    @Binding var isEnabled: Bool
    @Binding var expandedSections: Set<ExpandableSection>
    let content: Content

    init(
        section: ExpandableSection,
        title: String,
        helpText: String,
        isEnabled: Binding<Bool>,
        expandedSections: Binding<Set<ExpandableSection>>,
        @ViewBuilder content: () -> Content
    ) {
        self.section = section
        self.title = title
        self.helpText = helpText
        self._isEnabled = isEnabled
        self._expandedSections = expandedSections
        self.content = content()
    }

    private var isExpanded: Bool {
        expandedSections.contains(section)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Toggle("", isOn: $isEnabled)
                    .toggleStyle(.switch)
                    .scaleEffect(0.75)
                    .frame(width: 36)
                    .onChange(of: isEnabled) { _, newValue in
                        withAnimation(.easeInOut(duration: 0.15)) {
                            if newValue {
                                _ = expandedSections.insert(section)
                            } else {
                                expandedSections.remove(section)
                            }
                        }
                    }

                Text(title)
                    .font(.system(size: 12))
                    .foregroundColor(isEnabled ? .primary : .secondary)

                Spacer()

                if isEnabled {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(Color(NSColor.tertiaryLabelColor))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .animation(.easeInOut(duration: 0.15), value: isExpanded)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                if isEnabled {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        if isExpanded {
                            expandedSections.remove(section)
                        } else {
                            _ = expandedSections.insert(section)
                        }
                    }
                }
            }
            .help(helpText)

            if isEnabled && isExpanded {
                content
                    .transition(.opacity.combined(with: .move(edge: .top)))
                    .padding(.leading, 42)
                    .padding(.top, 2)
            }
        }
    }
}
