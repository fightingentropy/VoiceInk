import SwiftUI

enum MonochromeStyle {
    // Semantic colors keep the monochrome identity while inheriting native
    // vibrancy, contrast, and inactive-window behavior from AppKit.
    static let canvas = Color(nsColor: .windowBackgroundColor)
    static let sidebar = Color(nsColor: .underPageBackgroundColor)
    static let subtleFill = Color.primary.opacity(0.045)
    static let raisedFill = Color.primary.opacity(0.075)
    static let selectedFill = Color.primary.opacity(0.12)
    static let hairline = Color.primary.opacity(0.10)
    static let primaryText = Color.primary
    static let secondaryText = Color.secondary
    static let dimText = Color(nsColor: .tertiaryLabelColor)
}

extension View {
    func monochromeSurface(
        cornerRadius: CGFloat = 14,
        tint: Color = MonochromeStyle.subtleFill
    ) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        return self
            .background(tint, in: shape)
            .overlay(shape.stroke(MonochromeStyle.hairline, lineWidth: 0.75))
    }
}

struct MonochromeActionButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(MonochromeStyle.primaryText)
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(
                        configuration.isPressed
                            ? MonochromeStyle.selectedFill
                            : MonochromeStyle.raisedFill
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(MonochromeStyle.hairline, lineWidth: 0.75)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
    }
}

struct MonochromeIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(
                configuration.isPressed
                    ? MonochromeStyle.primaryText
                    : MonochromeStyle.secondaryText
            )
            .frame(width: 26, height: 26)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(configuration.isPressed ? MonochromeStyle.selectedFill : Color.clear)
            )
    }
}

struct MonochromeDivider: View {
    var body: some View {
        Rectangle()
            .fill(MonochromeStyle.hairline)
            .frame(height: 0.75)
    }
}

struct MonochromeSettingsSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(title)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .fontWidth(.condensed)
                .foregroundStyle(.secondary)
                .padding(.leading, 4)

            VStack(alignment: .leading, spacing: 0) {
                content()
            }
            .padding(.horizontal, 15)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.75)
            }
            .shadow(color: .black.opacity(0.12), radius: 12, y: 5)
        }
    }
}

struct MonochromeToggleRow<Label: View>: View {
    @Binding var isOn: Bool
    @ViewBuilder let label: () -> Label

    var body: some View {
        HStack(spacing: 12) {
            label()
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .fontWidth(.condensed)
                .foregroundStyle(MonochromeStyle.primaryText)

            Spacer(minLength: 16)

            Toggle("", isOn: $isOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
                .tint(.accentColor)
        }
        .frame(minHeight: 44)
    }
}
