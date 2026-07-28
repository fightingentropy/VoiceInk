import SwiftUI

struct CompactHeroSection: View {
    let icon: String
    let title: String
    let description: String
    var maxDescriptionWidth: CGFloat? = nil

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 20, weight: .medium))
                .foregroundStyle(MonochromeStyle.primaryText)
                .symbolRenderingMode(.monochrome)

            VStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 18, weight: .semibold))
                Text(description)
                    .font(.system(size: 12))
                    .foregroundStyle(MonochromeStyle.secondaryText)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: maxDescriptionWidth)
            }
        }
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity)
    }
}
