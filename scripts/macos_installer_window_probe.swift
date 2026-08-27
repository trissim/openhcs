import CoreGraphics
import Foundation

private enum InstallerWindowOperation: String {
    case inspect
    case pressPrimary = "press-primary"

    func perform(processIdentifier: pid_t) throws {
        switch self {
        case .inspect:
            return
        case .pressPrimary:
            let returnKeyCode = CGKeyCode(36)
            guard
                let keyDown = CGEvent(
                    keyboardEventSource: nil,
                    virtualKey: returnKeyCode,
                    keyDown: true
                ),
                let keyUp = CGEvent(
                    keyboardEventSource: nil,
                    virtualKey: returnKeyCode,
                    keyDown: false
            )
            else {
                throw KeyboardEventUnavailableError()
            }
            keyDown.postToPid(processIdentifier)
            keyUp.postToPid(processIdentifier)
        }
    }
}

private struct KeyboardEventUnavailableError: LocalizedError {
    let errorDescription: String? =
        "macOS could not create the installer keyboard event."
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
    try operation.perform(processIdentifier: processIdentifier)
} catch {
    FileHandle.standardError.write(
        Data("\(error.localizedDescription)\n".utf8)
    )
    exit(2)
}
