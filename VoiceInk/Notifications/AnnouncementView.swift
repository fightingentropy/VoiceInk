import SwiftUI

struct AnnouncementView: View {
    let title: String
    let description: String
    let onClose: () -> Void
    let onLearnMore: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(MonochromeStyle.primaryText)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)

                Spacer()

                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(MonochromeStyle.secondaryText)
                }
                .buttonStyle(PlainButtonStyle())
            }

            if !description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                ScrollView {
                    Text(description)
                        .font(.system(size: 12))
                        .foregroundColor(MonochromeStyle.primaryText)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 120)
            }

            HStack(spacing: 8) {
                Button(action: onLearnMore) {
                    Text("Learn more")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.black)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(MonochromeStyle.primaryText)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(PlainButtonStyle())

                Button(action: onClose) {
                    Text("Dismiss")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(MonochromeStyle.primaryText)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                }
                .buttonStyle(PlainButtonStyle())
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(minWidth: 360, idealWidth: 420)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(MonochromeStyle.subtleFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(MonochromeStyle.hairline, lineWidth: 0.75)
        )
    }
}


 
