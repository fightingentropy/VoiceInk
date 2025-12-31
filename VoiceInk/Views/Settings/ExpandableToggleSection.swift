import SwiftUI

struct ExpandableToggleSection<Content: View>: View {
    let title: String
    let helpText: String
    @Binding var isEnabled: Bool
    @Binding var isExpanded: Bool
    let content: Content

    init(
        title: String,
        helpText: String,
        isEnabled: Binding<Bool>,
        isExpanded: Binding<Bool>,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.helpText = helpText
        self._isEnabled = isEnabled
        self._isExpanded = isExpanded
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Toggle(isOn: $isEnabled) {
                    Text(title)
                }
                .toggleStyle(.switch)
                .help(helpText)

                if isEnabled {
                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.secondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .animation(.easeInOut(duration: 0.2), value: isExpanded)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                if isEnabled {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                }
            }

            if isEnabled && isExpanded {
                content
                    .transition(.opacity.combined(with: .move(edge: .top)))
                    .padding(.top, 4)
            }
        }
    }
}
