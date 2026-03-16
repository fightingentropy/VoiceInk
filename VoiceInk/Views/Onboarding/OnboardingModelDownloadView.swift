import SwiftUI

struct OnboardingModelDownloadView: View {
    @Binding var hasCompletedOnboarding: Bool

    var body: some View {
        OnboardingTutorialView(hasCompletedOnboarding: $hasCompletedOnboarding)
    }
}
