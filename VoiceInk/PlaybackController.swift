import AppKit
import Combine
import Foundation
import SwiftUI
import MediaRemoteAdapter

/// Pauses media when recording starts, resumes when recording stops
class PlaybackController: ObservableObject {
    static let shared = PlaybackController()
    private var mediaController: MediaRemoteAdapter.MediaController
    private var didPauseMedia = false
    
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
        
        mediaController.onListenerTerminated = {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.mediaController.startListening()
            }
        }
    }
    
    func pauseMedia() async -> Bool {
        guard isPauseMediaEnabled else { return false }
        
        mediaController.pause()
        didPauseMedia = true
        return true
    }
    
    func resumeMedia() async {
        guard isPauseMediaEnabled && didPauseMedia else { return }
        
        mediaController.play()
        didPauseMedia = false
    }
}

extension UserDefaults {
    var isPauseMediaEnabled: Bool {
        get { bool(forKey: "isPauseMediaEnabled") }
        set { set(newValue, forKey: "isPauseMediaEnabled") }
    }
} 