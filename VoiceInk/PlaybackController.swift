import AppKit
import Combine
import Foundation
import SwiftUI
import MediaRemoteAdapter
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
            UserDefaults.standard.set(false, forKey: "isPauseMediaEnabled")
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
        wasPlayingWhenRecordingStarted = false
        originalMediaAppBundleId = nil
        
        guard isPauseMediaEnabled, 
              isMediaPlaying,
              lastKnownTrackInfo?.payload.isPlaying == true,
              let bundleId = lastKnownTrackInfo?.payload.bundleIdentifier else {
            return
        }
        
        wasPlayingWhenRecordingStarted = true
        originalMediaAppBundleId = bundleId
        mediaController.pause()
    }

    func resumeMedia() async {
        defer {
            wasPlayingWhenRecordingStarted = false
            originalMediaAppBundleId = nil
        }
        
        guard isPauseMediaEnabled,
              wasPlayingWhenRecordingStarted,
              let bundleId = originalMediaAppBundleId else {
            return
        }
        
        guard isAppStillRunning(bundleId: bundleId) else {
            return
        }
        
        guard let currentTrackInfo = lastKnownTrackInfo,
              let currentBundleId = currentTrackInfo.payload.bundleIdentifier,
              currentBundleId == bundleId,
              currentTrackInfo.payload.isPlaying == false else {
            return
        }
        
        mediaController.play()
    }
    
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