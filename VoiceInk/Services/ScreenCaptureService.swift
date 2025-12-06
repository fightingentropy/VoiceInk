import Foundation
import AppKit
import Vision
import os
import ScreenCaptureKit

@MainActor
class ScreenCaptureService: ObservableObject {
    @Published var isCapturing = false
    @Published var lastCapturedText: String?
    
    private let logger = Logger(
        subsystem: "com.prakashjoshipax.voiceink",
        category: "aienhancement"
    )
    
    private struct WindowCandidate {
        let title: String
        let ownerName: String
        let windowID: CGWindowID
        let ownerPID: pid_t
        let layer: Int32
        let bounds: CGRect
    }

    private func getActiveWindowInfo() async -> (title: String, ownerName: String, windowID: CGWindowID)? {
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let frontmostPID = await NSWorkspace.shared.frontmostApplication?.processIdentifier
        
        // Move expensive window list retrieval off the main thread
        let candidates = await Task.detached(priority: .userInitiated) { () -> [WindowCandidate] in
            let windowListInfo = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? []
            
            return windowListInfo.compactMap { info -> WindowCandidate? in
                guard let windowID = info[kCGWindowNumber as String] as? CGWindowID,
                      let ownerName = info[kCGWindowOwnerName as String] as? String,
                      let ownerPIDNumber = info[kCGWindowOwnerPID as String] as? NSNumber,
                      let layer = info[kCGWindowLayer as String] as? Int32,
                      let boundsDict = info[kCGWindowBounds as String] as? [String: Any],
                      let width = boundsDict["Width"] as? CGFloat,
                      let height = boundsDict["Height"] as? CGFloat else {
                    return nil
                }

                let rawTitle = (info[kCGWindowName as String] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                let resolvedTitle = rawTitle?.isEmpty == false ? rawTitle! : ownerName
                let bounds = CGRect(
                    x: boundsDict["X"] as? CGFloat ?? 0,
                    y: boundsDict["Y"] as? CGFloat ?? 0,
                    width: width,
                    height: height
                )

                return WindowCandidate(
                    title: resolvedTitle,
                    ownerName: ownerName,
                    windowID: windowID,
                    ownerPID: ownerPIDNumber.int32Value,
                    layer: layer,
                    bounds: bounds
                )
            }
        }.value

        func isEligible(_ candidate: WindowCandidate) -> Bool {
            guard candidate.layer == 0 else { return false }
            guard candidate.ownerPID != currentPID else { return false }
            return candidate.bounds.width >= 120 && candidate.bounds.height >= 120
        }

        if let frontmostPID = frontmostPID,
           let focusedWindow = candidates.first(where: { isEligible($0) && $0.ownerPID == frontmostPID }) {
            return (title: focusedWindow.title, ownerName: focusedWindow.ownerName, windowID: focusedWindow.windowID)
        }

        if let fallbackWindow = candidates.first(where: isEligible) {
            return (title: fallbackWindow.title, ownerName: fallbackWindow.ownerName, windowID: fallbackWindow.windowID)
        }

        return nil
    }
    
    func captureActiveWindow() async -> NSImage? {
        guard let windowInfo = await getActiveWindowInfo() else {
            return nil
        }
        
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            
            guard let targetWindow = content.windows.first(where: { $0.windowID == windowInfo.windowID }) else {
                return nil
            }
            
            let filter = SCContentFilter(desktopIndependentWindow: targetWindow)
            
            let configuration = SCStreamConfiguration()
            configuration.width = Int(targetWindow.frame.width) * 2
            configuration.height = Int(targetWindow.frame.height) * 2
            
            let cgImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: configuration)
            
            return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
            
        } catch {
            logger.notice("📸 Screen capture failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }
    
    private func extractText(from image: NSImage) async -> String? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        
        let result: Result<String?, Error> = await Task.detached(priority: .userInitiated) {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            request.automaticallyDetectsLanguage = true
            
            let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            
            do {
                try requestHandler.perform([request])
                guard let observations = request.results as? [VNRecognizedTextObservation] else {
                    return .success(nil)
                }
                
                let text = observations
                    .compactMap { $0.topCandidates(1).first?.string }
                    .joined(separator: "\n")
                
                return .success(text.isEmpty ? nil : text)
            } catch {
                return .failure(error)
            }
        }.value
        
        switch result {
        case .success(let text):
            return text
        case .failure(let error):
            logger.notice("📸 Text recognition failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }
    
    func captureAndExtractText() async -> String? {
        guard !isCapturing else { 
            return nil 
        }
        
        isCapturing = true
        defer { 
            DispatchQueue.main.async {
                self.isCapturing = false
            }
        }

        guard let windowInfo = await getActiveWindowInfo() else {
            logger.notice("📸 No active window found")
            return nil
        }
        
        logger.notice("📸 Capturing: \(windowInfo.title, privacy: .public) (\(windowInfo.ownerName, privacy: .public))")

        var contextText = """
        Active Window: \(windowInfo.title)
        Application: \(windowInfo.ownerName)
        
        """

        if let capturedImage = await captureActiveWindow() {
            let extractedText = await extractText(from: capturedImage)
            
            if let extractedText = extractedText, !extractedText.isEmpty {
                contextText += "Window Content:\n\(extractedText)"
                let preview = String(extractedText.prefix(100))
                logger.notice("📸 Text extracted: \(preview, privacy: .public)\(extractedText.count > 100 ? "..." : "")")
            } else {
                contextText += "Window Content:\nNo text detected via OCR"
                logger.notice("📸 No text extracted from window")
            }
            
            await MainActor.run {
                self.lastCapturedText = contextText
            }
            
            return contextText
        }
        
        logger.notice("📸 Window capture failed")
        return nil
    }
} 
