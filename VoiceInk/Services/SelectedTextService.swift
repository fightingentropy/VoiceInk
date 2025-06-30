import Foundation
import AppKit

class SelectedTextService {
    static func fetchSelectedText() -> String? {
        // Do not check for selected text within VoiceInk itself.
        guard let frontmostApp = NSWorkspace.shared.frontmostApplication,
              frontmostApp.bundleIdentifier != "com.prakashjoshipax.VoiceInk" else {
            return nil
        }

        // Get the currently focused UI element system-wide.
        let systemWideElement = AXUIElementCreateSystemWide()
        var focusedElement: CFTypeRef?
        guard AXUIElementCopyAttributeValue(systemWideElement, kAXFocusedUIElementAttribute as CFString, &focusedElement) == .success, focusedElement != nil else {
            return nil
        }
        let element = focusedElement as! AXUIElement

        // First, try the standard attribute, which is the most reliable method.
        var selectedTextValue: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXSelectedTextAttribute as CFString, &selectedTextValue) == .success,
           let selectedText = selectedTextValue as? String, !selectedText.isEmpty {
            return selectedText
        }

        // If the standard attribute fails, fall back to checking the selection range.
        // This correctly handles apps that don't support the standard attribute.
        var selectedRangeValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &selectedRangeValue) == .success, selectedRangeValue != nil else {
            return nil
        }
        let axRangeValue = selectedRangeValue as! AXValue

        var selectedRange: CFRange = .init()
        AXValueGetValue(axRangeValue, .cfRange, &selectedRange)

        // An actual selection must have a length greater than zero.
        guard selectedRange.length > 0 else {
            return nil
        }
        
        var fullTextValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &fullTextValue) == .success,
              let fullText = fullTextValue as? String,
              let subrange = Range(NSRange(location: selectedRange.location, length: selectedRange.length), in: fullText) else {
            return nil
        }
        
        return String(fullText[subrange])
    }
} 