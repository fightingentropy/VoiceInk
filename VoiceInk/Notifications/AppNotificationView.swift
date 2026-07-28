import SwiftUI

struct AppNotificationView: View {
    let title: String
    let type: NotificationType
    let duration: TimeInterval
    let onClose: () -> Void
    let onTap: (() -> Void)?
    
    @State private var progress: Double = 1.0
    @State private var progressTask: Task<Void, Never>?

    enum NotificationType {
        case error
        case warning
        case info
        case success

        var iconName: String {
            switch self {
            case .error: return "xmark.octagon.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .info: return "info.circle.fill"
            case .success: return "checkmark.circle.fill"
            }
        }

        var iconColor: Color {
            MonochromeStyle.primaryText
        }
    }

    var body: some View {
        ZStack {
            HStack(alignment: .center, spacing: 12) {
                // Type icon
                Image(systemName: type.iconName)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(type.iconColor)
                    .frame(width: 20, height: 20)

                // Single message text
                Text(title)
                    .font(.system(size: 12))
                    .fontWeight(.medium)
                    .foregroundColor(MonochromeStyle.primaryText)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                
                Spacer()
                
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(MonochromeStyle.secondaryText)
                }
                .buttonStyle(PlainButtonStyle())
                .frame(width: 16, height: 16)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .frame(minWidth: 220, maxWidth: 750, minHeight: 44)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(MonochromeStyle.subtleFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(MonochromeStyle.hairline, lineWidth: 0.75)
        )
        .overlay(
            VStack {
                Spacer()
                GeometryReader { geometry in
                    Rectangle()
                        .fill(MonochromeStyle.primaryText)
                        .frame(width: geometry.size.width * max(0, progress), height: 2)
                        .animation(.linear(duration: 0.1), value: progress)
                }
                .frame(height: 2)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        )
        .onAppear {
            startProgressTimer()
        }
        .onDisappear {
            progressTask?.cancel()
            progressTask = nil
        }
        .onTapGesture {
            if let onTap = onTap {
                onTap()
                onClose()
            }
        }
    }
    
    private func startProgressTimer() {
        let updateInterval: TimeInterval = 0.1
        let totalSteps = duration / updateInterval
        let stepDecrement = 1.0 / totalSteps

        progressTask?.cancel()
        progressTask = Task { @MainActor in
            while !Task.isCancelled && progress > 0 {
                try? await Task.sleep(nanoseconds: UInt64(updateInterval * 1_000_000_000))
                guard !Task.isCancelled else { break }
                progress = max(0, progress - stepDecrement)
            }
        }
    }
}
