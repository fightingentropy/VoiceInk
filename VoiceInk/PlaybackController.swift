import AppKit
import Combine
import Foundation
import SwiftUI
import MediaRemoteAdapter

/// Pauses media when recording starts, resumes when recording stops
class PlaybackController: ObservableObject {
    static let shared = PlaybackController()
    private var mediaController: MediaRemoteAdapter.MediaController
    private var wasPlayingWhenRecordingStarted = false
    private var isMediaPlaying = false
    private var lastKnownTrackInfo: TrackInfo?
    private var originalMediaAppBundleId: String?
    
    @Published var isPauseMediaEnabled: Bool = UserDefaults.standard.bool(forKey: "isPauseMediaEnabled") {
        didSet {
            UserDefaults.standard.set(isPauseMediaEnabled, forKey: "isPauseMediaEnabled")
        }
    }
    
    private init() {
        mediaController = MediaRemoteAdapter.MediaController()
        
        if !UserDefaults.standard.contains(key: "isPauseMediaEnabled") {
            UserDefaults.standard.set(true, forKey: "isPauseMediaEnabled")
        }
        
        mediaController.startListening()
        mediaController.onTrackInfoReceived = { [weak self] trackInfo in
            self?.isMediaPlaying = trackInfo.payload.isPlaying ?? false
            self?.lastKnownTrackInfo = trackInfo
        }
        
        mediaController.onListenerTerminated = {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.mediaController.startListening()
            }
        }
    }
    
    func pauseMedia() async {
        guard isPauseMediaEnabled else { return }

        if isMediaPlaying {
            wasPlayingWhenRecordingStarted = true
            originalMediaAppBundleId = lastKnownTrackInfo?.payload.bundleIdentifier
            mediaController.pause()
        } else {
            wasPlayingWhenRecordingStarted = false
            originalMediaAppBundleId = nil
        }
    }

    func resumeMedia() async {
        guard isPauseMediaEnabled, wasPlayingWhenRecordingStarted else { return }
        
        if let bundleId = originalMediaAppBundleId, isAppStillRunning(bundleId: bundleId) {
            mediaController.play()
        }
        
        originalMediaAppBundleId = nil
    }
    
    /// Checks if an app with the given bundle identifier is currently running
    private func isAppStillRunning(bundleId: String) -> Bool {
        let runningApps = NSWorkspace.shared.runningApplications
        return runningApps.contains { $0.bundleIdentifier == bundleId }
    }
}

extension UserDefaults {
    var isPauseMediaEnabled: Bool {
        get { bool(forKey: "isPauseMediaEnabled") }
        set { set(newValue, forKey: "isPauseMediaEnabled") }
    }
} 