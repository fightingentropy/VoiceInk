import SwiftUI

enum MonochromeStyle {
    // Opaque equivalents of the Spotify mobile app's monochrome tokens.
    static let canvas = Color(red: 16 / 255, green: 16 / 255, blue: 18 / 255)
    static let sidebar = Color(red: 20 / 255, green: 20 / 255, blue: 22 / 255)
    static let subtleFill = Color(red: 26 / 255, green: 26 / 255, blue: 29 / 255)
    static let raisedFill = Color(red: 35 / 255, green: 35 / 255, blue: 38 / 255)
    static let selectedFill = Color(red: 44 / 255, green: 44 / 255, blue: 48 / 255)
    static let hairline = Color(red: 51 / 255, green: 51 / 255, blue: 56 / 255)
    static let primaryText = Color(red: 242 / 255, green: 242 / 255, blue: 242 / 255)
    static let secondaryText = Color(red: 153 / 255, green: 153 / 255, blue: 153 / 255)
    static let dimText = Color(red: 102 / 255, green: 102 / 255, blue: 102 / 255)
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
                Circle()
                    .fill(configuration.isPressed ? MonochromeStyle.selectedFill : MonochromeStyle.canvas)
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
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold))
                .tracking(1.1)
                .foregroundStyle(MonochromeStyle.secondaryText)
                .padding(.leading, 3)

            VStack(alignment: .leading, spacing: 0) {
                content()
            }
            .padding(.horizontal, 13)
            .monochromeSurface(cornerRadius: 13)
        }
    }
}

struct MonochromeToggleRow<Label: View>: View {
    @Binding var isOn: Bool
    @ViewBuilder let label: () -> Label

    var body: some View {
        HStack(spacing: 12) {
            label()
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(MonochromeStyle.primaryText)

            Spacer(minLength: 16)

            Toggle("", isOn: $isOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.mini)
                .tint(.white)
        }
        .frame(minHeight: 38)
    }
}
