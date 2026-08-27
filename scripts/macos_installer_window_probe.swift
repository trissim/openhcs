import ApplicationServices
import CoreGraphics
import Foundation

private enum InstallerWindowOperation: String {
    case inspect
    case pressPrimary = "press-primary"

    func perform(processIdentifier: pid_t, windowTitle: String) throws {
        switch self {
        case .inspect:
            return
        case .pressPrimary:
            try InstallerDefaultControl(
                processIdentifier: processIdentifier,
                windowTitle: windowTitle
            ).press()
        }
    }
}

private struct InstallerDefaultControl {
    let processIdentifier: pid_t
    let windowTitle: String

    func press() throws {
        let application = AXUIElementCreateApplication(processIdentifier)
        let windows = try elements(
            of: application,
            attribute: kAXWindowsAttribute
        )
        let matchingWindows = try windows.filter {
            try text(of: $0, attribute: kAXTitleAttribute) == windowTitle
        }
        guard matchingWindows.count == 1 else {
            throw InstallerControlError(
                message: "macOS did not expose one exact installer window."
            )
        }
        let button = try element(
            of: matchingWindows[0],
            attribute: kAXDefaultButtonAttribute
        )
        guard
            try text(of: button, attribute: kAXRoleAttribute)
                == kAXButtonRole as String,
            try flag(of: button, attribute: kAXEnabledAttribute)
        else {
            throw InstallerControlError(
                message: "The installer default control is not an enabled button."
            )
        }
        let result = AXUIElementPerformAction(button, kAXPressAction as CFString)
        guard result == .success else {
            throw InstallerControlError(
                message: "macOS could not press the installer default control: \(result.rawValue)."
            )
        }
    }

    private func elements(
        of owner: AXUIElement,
        attribute: CFString
    ) throws -> [AXUIElement] {
        let value = try attributeValue(of: owner, attribute: attribute)
        guard let elements = value as? [AXUIElement] else {
            throw InstallerControlError(
                message: "macOS returned a non-element accessibility collection."
            )
        }
        return elements
    }

    private func element(
        of owner: AXUIElement,
        attribute: CFString
    ) throws -> AXUIElement {
        let value = try attributeValue(of: owner, attribute: attribute)
        guard CFGetTypeID(value) == AXUIElementGetTypeID() else {
            throw InstallerControlError(
                message: "macOS returned a non-element accessibility value."
            )
        }
        return value as! AXUIElement
    }

    private func text(
        of owner: AXUIElement,
        attribute: CFString
    ) throws -> String {
        let value = try attributeValue(of: owner, attribute: attribute)
        guard let text = value as? String else {
            throw InstallerControlError(
                message: "macOS returned a non-text accessibility value."
            )
        }
        return text
    }

    private func flag(
        of owner: AXUIElement,
        attribute: CFString
    ) throws -> Bool {
        let value = try attributeValue(of: owner, attribute: attribute)
        guard let flag = value as? Bool else {
            throw InstallerControlError(
                message: "macOS returned a non-Boolean accessibility value."
            )
        }
        return flag
    }

    private func attributeValue(
        of owner: AXUIElement,
        attribute: CFString
    ) throws -> CFTypeRef {
        var value: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            owner,
            attribute,
            &value
        )
        guard result == .success, let value else {
            throw InstallerControlError(
                message: "macOS could not read installer accessibility state: \(result.rawValue)."
            )
        }
        return value
    }
}

private struct InstallerControlError: LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

guard CommandLine.arguments.count == 4,
      let processIdentifier = Int32(CommandLine.arguments[1]),
      let operation = InstallerWindowOperation(rawValue: CommandLine.arguments[3])
else {
    FileHandle.standardError.write(
        Data(
            "usage: macos_installer_window_probe PID EXPECTED_TITLE "
                .appending("{inspect|press-primary}\n").utf8
        )
    )
    exit(2)
}

let expectedTitle = CommandLine.arguments[2]
let windowRecords = CGWindowListCopyWindowInfo(
    [.optionOnScreenOnly, .excludeDesktopElements],
    kCGNullWindowID
) as? [[String: Any]] ?? []

let matchingWindows = windowRecords.compactMap { record -> [String: Any]? in
    guard let ownerPID = record[kCGWindowOwnerPID as String] as? Int32,
          ownerPID == processIdentifier,
          let layer = record[kCGWindowLayer as String] as? Int,
          layer == 0,
          let title = record[kCGWindowName as String] as? String,
          title == expectedTitle,
          let boundsPayload = record[kCGWindowBounds as String] as? [String: Any],
          let bounds = CGRect(
              dictionaryRepresentation: boundsPayload as CFDictionary
          ),
          bounds.width >= 600,
          bounds.height >= 500,
          let windowID = record[kCGWindowNumber as String] as? Int else {
        return nil
    }
    return [
        "process_id": processIdentifier,
        "title": title,
        "window_id": windowID,
        "left": bounds.origin.x,
        "top": bounds.origin.y,
        "width": bounds.width,
        "height": bounds.height,
    ]
}

guard matchingWindows.count == 1 else {
    exit(1)
}

let payload = try JSONSerialization.data(
    withJSONObject: matchingWindows[0],
    options: [.prettyPrinted, .sortedKeys]
)
FileHandle.standardOutput.write(payload)
FileHandle.standardOutput.write(Data("\n".utf8))

do {
    try operation.perform(
        processIdentifier: processIdentifier,
        windowTitle: expectedTitle
    )
} catch {
    FileHandle.standardError.write(
        Data("\(error.localizedDescription)\n".utf8)
    )
    exit(2)
}
