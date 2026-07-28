import SwiftUI

struct StyleConstants {
    static let cornerRadius: CGFloat = 12
}

struct CardBackground: View {
    var isSelected: Bool
    var cornerRadius: CGFloat = StyleConstants.cornerRadius
    var useAccentGradientWhenSelected: Bool = false

    private var shape: RoundedRectangle {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
    }

    var body: some View {
        shape
            .fill(isSelected ? MonochromeStyle.selectedFill : MonochromeStyle.subtleFill)
            .monochromeSurface(
                cornerRadius: cornerRadius,
                tint: isSelected ? MonochromeStyle.selectedFill : MonochromeStyle.subtleFill
            )
            .overlay(
                shape.strokeBorder(
                    MonochromeStyle.hairline,
                    lineWidth: 0.75
                )
            )
    }
}
