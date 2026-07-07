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
            .fill(.thinMaterial)
            .overlay {
                if isSelected && useAccentGradientWhenSelected {
                    shape.fill(Color.accentColor.opacity(0.12))
                }
            }
            .overlay(
                shape.strokeBorder(
                    isSelected ? Color.accentColor.opacity(0.5) : Color.primary.opacity(0.07),
                    lineWidth: 1
                )
            )
            .shadow(color: Color.black.opacity(0.05), radius: 2, x: 0, y: 1)
    }
}
