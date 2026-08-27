import CoreGraphics
import Foundation

guard CommandLine.arguments.count == 3,
      let processIdentifier = Int32(CommandLine.arguments[1]) else {
    FileHandle.standardError.write(
        Data("usage: macos_installer_window_probe PID EXPECTED_TITLE\n".utf8)
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
          let boundsPayload = record[kCGWindowBounds as String] as? CFDictionary,
          let bounds = CGRect(dictionaryRepresentation: boundsPayload),
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
