import Foundation
import AppKit
import ApplicationServices

class SelectedTextService {
    static func fetchSelectedText() -> String? {
        guard ensureAccessibilityPermission() else {
            return nil
        }

        let strategies: [() -> String?] = [
            getSelectedTextViaAccessibility,
            getSelectedTextViaKeyboardCopy
        ]

        for fetch in strategies {
            if let text = fetch(), !text.isEmpty {
                return text
            }
        }

        return nil
    }

    private static func getSelectedTextViaAccessibility() -> String? {
        let systemWideElement = AXUIElementCreateSystemWide()

        var focusedElement: AnyObject?
        let focusedResult = AXUIElementCopyAttributeValue(
            systemWideElement,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElement
        )

        guard focusedResult == .success, let focusedElement else {
            return nil
        }

        let element = focusedElement as! AXUIElement

        if let selectedText = stringAttribute(kAXSelectedTextAttribute as CFString, of: element) {
            return selectedText
        }

        guard
            let selectedRange = rangeAttribute(kAXSelectedTextRangeAttribute as CFString, of: element),
            let value = stringAttribute(kAXValueAttribute as CFString, of: element),
            let textRange = Range(selectedRange, in: value)
        else {
            return nil
        }

        let substring = String(value[textRange])
        return substring.isEmpty ? nil : substring
    }

    private static func getSelectedTextViaKeyboardCopy() -> String? {
        return readSelectedTextUsingCopyAction {
            let source = CGEventSource(stateID: .hidSystemState)
            let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: true)
            cmdDown?.flags = .maskCommand
            let cDown = CGEvent(keyboardEventSource: source, virtualKey: 0x08, keyDown: true)
            cDown?.flags = .maskCommand
            let cUp = CGEvent(keyboardEventSource: source, virtualKey: 0x08, keyDown: false)
            cUp?.flags = .maskCommand
            let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: false)

            cmdDown?.post(tap: .cghidEventTap)
            cDown?.post(tap: .cghidEventTap)
            cUp?.post(tap: .cghidEventTap)
            cmdUp?.post(tap: .cghidEventTap)

            return true
        }
    }

    private static func readSelectedTextUsingCopyAction(_ copyAction: () -> Bool) -> String? {

        let pasteboard = NSPasteboard.general
        let originalClipboardText = pasteboard.string(forType: .string)
        let originalPasteboardItems = pasteboard.pasteboardItems?.map { item in
            item.types.reduce(into: [NSPasteboard.PasteboardType: Data]()) { acc, type in
                if let data = item.data(forType: type) {
                    acc[type] = data
                }
            }
        }

        defer {
            restorePasteboard(
                pasteboard,
                items: originalPasteboardItems,
                string: originalClipboardText
            )
        }

        // Clear clipboard to prepare for selection detection
        pasteboard.clearContents()

        guard copyAction() else { return nil }

        // Wait for copy operation to complete
        Thread.sleep(forTimeInterval: 0.1)

        // Read the copied text
        return pasteboard.string(forType: .string)
    }

    private static func restorePasteboard(
        _ pasteboard: NSPasteboard,
        items: [[NSPasteboard.PasteboardType: Data]]?,
        string: String?
    ) {
        pasteboard.clearContents()

        if let items, !items.isEmpty {
            let restoredItems: [NSPasteboardItem] = items.compactMap { dataMap in
                guard !dataMap.isEmpty else { return nil }
                let item = NSPasteboardItem()
                for (type, data) in dataMap {
                    item.setData(data, forType: type)
                }
                return item
            }

            if !restoredItems.isEmpty {
                pasteboard.writeObjects(restoredItems)
                return
            }
        }

        if let string {
            _ = pasteboard.setString(string, forType: .string)
        }
    }

    private static func stringAttribute(_ attribute: CFString, of element: AXUIElement) -> String? {
        var value: AnyObject?
        let result = AXUIElementCopyAttributeValue(element, attribute, &value)

        guard result == .success, let value else {
            return nil
        }

        if let text = value as? String {
            return text
        }

        if let attributed = value as? NSAttributedString {
            return attributed.string
        }

        return nil
    }

    private static func rangeAttribute(_ attribute: CFString, of element: AXUIElement) -> NSRange? {
        var value: AnyObject?
        let result = AXUIElementCopyAttributeValue(element, attribute, &value)

        guard result == .success, let value else {
            return nil
        }

        guard CFGetTypeID(value) == AXValueGetTypeID() else {
            return nil
        }

        let rangeValue = value as! AXValue
        guard AXValueGetType(rangeValue) == .cfRange else {
            return nil
        }

        var range = CFRange()
        let success = AXValueGetValue(rangeValue, .cfRange, &range)

        guard success else {
            return nil
        }

        return NSRange(location: range.location, length: range.length)
    }

    private static func ensureAccessibilityPermission() -> Bool {
        AXIsProcessTrustedWithOptions(nil)
    }
}
