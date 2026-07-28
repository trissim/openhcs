import AppKit
import Foundation

private enum InstallerScreen {
    case welcome
    case installing
    case cancelling
    case finished
    case failed
    case cancelled
}

private struct InstallerResources {
    let contractURL: URL
    let bootstrapURL: URL
    let productName: String

    static func load() throws -> InstallerResources {
        guard
            let resourcesURL = Bundle.main.resourceURL,
            let contractURL = Bundle.main.url(
                forResource: "installer_contract",
                withExtension: "json"
            )
        else {
            throw InstallerStartupError(
                message: "The shared installer contract is missing."
            )
        }

        let bootstrapURL = resourcesURL.appendingPathComponent("install-openhcs.sh")
        guard FileManager.default.isExecutableFile(atPath: bootstrapURL.path) else {
            throw InstallerStartupError(
                message: "The installation worker is missing or is not executable."
            )
        }

        let data = try Data(contentsOf: contractURL)
        guard
            let contract = try JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let productName = contract["product_name"] as? String,
            !productName.isEmpty,
            productName.count <= 80,
            productName.rangeOfCharacter(from: .controlCharacters) == nil
        else {
            throw InstallerStartupError(
                message: "The shared installer contract has no safe product name."
            )
        }

        return InstallerResources(
            contractURL: contractURL,
            bootstrapURL: bootstrapURL,
            productName: productName
        )
    }
}

private struct InstallerStartupError: LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

private final class InstallerController: NSObject, NSApplicationDelegate,
    NSWindowDelegate
{
    private var screen: InstallerScreen = .welcome
    private var resources: InstallerResources?
    private var worker: Process?
    private var workerPipe: Pipe?
    private var progressTimer: Timer?
    private var stateDirectoryURL: URL?
    private var workerOutput = Data()
    private var cancellationRequested = false

    private let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 590, height: 390),
        styleMask: [.titled, .closable],
        backing: .buffered,
        defer: false
    )
    private let iconView = NSImageView()
    private let titleLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(wrappingLabelWithString: "")
    private let statusLabel = NSTextField(wrappingLabelWithString: "")
    private let progressIndicator = NSProgressIndicator()
    private let connectAgentsCheckbox = NSButton(
        checkboxWithTitle:
            "Connect OpenHCS to ChatGPT, Codex, and local AI agent apps",
        target: nil,
        action: nil
    )
    private let launchCheckbox = NSButton(
        checkboxWithTitle: "Launch when the installer closes",
        target: nil,
        action: nil
    )
    private let showLogButton = NSButton(
        title: "Show Log",
        target: nil,
        action: nil
    )
    private let secondaryButton = NSButton(
        title: "Cancel",
        target: nil,
        action: nil
    )
    private let primaryButton = NSButton(
        title: "Continue",
        target: nil,
        action: nil
    )

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            resources = try InstallerResources.load()
        } catch {
            let alert = NSAlert()
            alert.alertStyle = .critical
            alert.messageText = "Installer cannot start"
            alert.informativeText = error.localizedDescription
            alert.runModal()
            NSApplication.shared.terminate(nil)
            return
        }

        configureWindow()
        apply(screen: .welcome)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(
        _ sender: NSApplication
    ) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        progressTimer?.invalidate()
        if let stateDirectoryURL {
            try? FileManager.default.removeItem(at: stateDirectoryURL)
        }
    }

    func applicationShouldTerminate(
        _ sender: NSApplication
    ) -> NSApplication.TerminateReply {
        if screen == .installing || screen == .cancelling {
            requestCancellation()
            return .terminateCancel
        }
        return .terminateNow
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if screen == .installing || screen == .cancelling {
            requestCancellation()
            return false
        }
        return true
    }

    private func configureWindow() {
        guard let resources else {
            return
        }

        window.title = "\(resources.productName) Installer"
        window.isReleasedWhenClosed = false
        window.delegate = self

        iconView.image = NSImage(named: NSImage.applicationIconName)
        iconView.imageScaling = .scaleProportionallyUpOrDown
        iconView.translatesAutoresizingMaskIntoConstraints = false

        titleLabel.font = NSFont.systemFont(ofSize: 26, weight: .semibold)
        titleLabel.maximumNumberOfLines = 2
        titleLabel.lineBreakMode = .byWordWrapping
        titleLabel.translatesAutoresizingMaskIntoConstraints = false

        detailLabel.font = NSFont.systemFont(ofSize: 14)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.maximumNumberOfLines = 4
        detailLabel.translatesAutoresizingMaskIntoConstraints = false

        statusLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        statusLabel.maximumNumberOfLines = 2
        statusLabel.translatesAutoresizingMaskIntoConstraints = false

        progressIndicator.style = .spinning
        progressIndicator.controlSize = .regular
        progressIndicator.isIndeterminate = true
        progressIndicator.translatesAutoresizingMaskIntoConstraints = false

        connectAgentsCheckbox.state = .on
        connectAgentsCheckbox.translatesAutoresizingMaskIntoConstraints = false

        launchCheckbox.state = .on
        launchCheckbox.translatesAutoresizingMaskIntoConstraints = false

        showLogButton.bezelStyle = .rounded
        showLogButton.target = self
        showLogButton.action = #selector(showLog(_:))
        showLogButton.translatesAutoresizingMaskIntoConstraints = false

        secondaryButton.bezelStyle = .rounded
        secondaryButton.target = self
        secondaryButton.action = #selector(secondaryAction(_:))
        secondaryButton.translatesAutoresizingMaskIntoConstraints = false

        primaryButton.bezelStyle = .rounded
        primaryButton.keyEquivalent = "\r"
        primaryButton.target = self
        primaryButton.action = #selector(primaryAction(_:))
        primaryButton.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView()
        content.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = content

        for view in [
            iconView,
            titleLabel,
            detailLabel,
            statusLabel,
            progressIndicator,
            connectAgentsCheckbox,
            launchCheckbox,
            showLogButton,
            secondaryButton,
            primaryButton,
        ] {
            content.addSubview(view)
        }

        NSLayoutConstraint.activate([
            iconView.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 30),
            iconView.topAnchor.constraint(equalTo: content.topAnchor, constant: 30),
            iconView.widthAnchor.constraint(equalToConstant: 64),
            iconView.heightAnchor.constraint(equalToConstant: 64),

            titleLabel.leadingAnchor.constraint(equalTo: iconView.trailingAnchor, constant: 22),
            titleLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -30),
            titleLabel.topAnchor.constraint(equalTo: content.topAnchor, constant: 32),

            detailLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            detailLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
            detailLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 12),

            progressIndicator.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            progressIndicator.topAnchor.constraint(equalTo: detailLabel.bottomAnchor, constant: 30),
            progressIndicator.widthAnchor.constraint(equalToConstant: 24),
            progressIndicator.heightAnchor.constraint(equalToConstant: 24),

            statusLabel.leadingAnchor.constraint(
                equalTo: progressIndicator.trailingAnchor,
                constant: 12
            ),
            statusLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
            statusLabel.centerYAnchor.constraint(equalTo: progressIndicator.centerYAnchor),

            connectAgentsCheckbox.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            connectAgentsCheckbox.topAnchor.constraint(
                equalTo: detailLabel.bottomAnchor,
                constant: 30
            ),

            launchCheckbox.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            launchCheckbox.topAnchor.constraint(equalTo: detailLabel.bottomAnchor, constant: 30),

            showLogButton.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 30),
            showLogButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -24),

            primaryButton.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            primaryButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -24),
            primaryButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 92),

            secondaryButton.trailingAnchor.constraint(
                equalTo: primaryButton.leadingAnchor,
                constant: -10
            ),
            secondaryButton.centerYAnchor.constraint(equalTo: primaryButton.centerYAnchor),
            secondaryButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 92),
        ])
    }

    private func apply(screen newScreen: InstallerScreen) {
        guard let resources else {
            return
        }

        screen = newScreen
        progressIndicator.stopAnimation(nil)
        progressIndicator.isHidden = true
        statusLabel.isHidden = true
        connectAgentsCheckbox.isHidden = true
        launchCheckbox.isHidden = true
        showLogButton.isHidden = true
        primaryButton.isHidden = false
        primaryButton.isEnabled = true
        secondaryButton.isHidden = false
        secondaryButton.isEnabled = true

        switch newScreen {
        case .welcome:
            titleLabel.stringValue = "Welcome to the \(resources.productName) Installer"
            detailLabel.stringValue =
                "This installer sets up everything needed in a private environment "
                + "for your macOS account. No existing Python, Terminal commands, "
                + "or administrator password is required."
            connectAgentsCheckbox.isHidden = false
            primaryButton.title = "Continue"
            secondaryButton.title = "Cancel"
        case .installing:
            titleLabel.stringValue = "Installing \(resources.productName)"
            detailLabel.stringValue =
                "The installer is downloading and preparing the application. "
                + "This can take several minutes."
            statusLabel.stringValue = "Preparing installation…"
            statusLabel.isHidden = false
            progressIndicator.isHidden = false
            progressIndicator.startAnimation(nil)
            primaryButton.isHidden = true
            secondaryButton.title = "Cancel"
        case .cancelling:
            titleLabel.stringValue = "Cancelling safely"
            detailLabel.stringValue =
                "The installer is cleaning up its unfinished environment. "
                + "Any previously installed version remains available."
            statusLabel.stringValue = "Waiting for cleanup to finish…"
            statusLabel.isHidden = false
            progressIndicator.isHidden = false
            progressIndicator.startAnimation(nil)
            primaryButton.isHidden = true
            secondaryButton.title = "Cancelling…"
            secondaryButton.isEnabled = false
        case .finished:
            titleLabel.stringValue = "\(resources.productName) is ready"
            if installerStateValue(named: "agent-registration-status") == "connected" {
                let connectedClients =
                    installerStateValue(named: "agent-registration-summary")
                    ?? "ChatGPT desktop, Codex, and detected local agent apps"
                detailLabel.stringValue =
                    "OpenHCS is connected to \(connectedClients). "
                    + "Restart ChatGPT desktop, Codex, and other listed apps, "
                    + "then ask them to use OpenHCS."
            } else if installerStateValue(
                named: "agent-registration-status"
            ) == "warning" {
                detailLabel.stringValue =
                    "OpenHCS is installed, but one or more agent connections need "
                    + "attention. Open the installer log for details."
                showLogButton.isHidden = installerLogURL() == nil
            } else {
                detailLabel.stringValue =
                    "The application is available in Applications. A Desktop shortcut "
                    + "was also added when that location was available."
            }
            launchCheckbox.isHidden = false
            primaryButton.title = "Finish"
            secondaryButton.isHidden = true
        case .failed:
            titleLabel.stringValue = "Installation could not be completed"
            let logIsAvailable = installerLogURL() != nil
            detailLabel.stringValue = logIsAvailable
                ? "Your previous installation, if any, is still available. "
                    + "Open the installer log for details, then run this installer again."
                : "Your previous installation, if any, is still available. "
                    + "The installer stopped before its durable log could be created."
            showLogButton.isHidden = !logIsAvailable
            primaryButton.title = "Close"
            secondaryButton.isHidden = true
        case .cancelled:
            titleLabel.stringValue = "Installation cancelled"
            detailLabel.stringValue =
                "No unfinished environment was activated. You can close this "
                + "installer or run it again whenever you are ready."
            showLogButton.isHidden = installerLogURL() == nil
            primaryButton.title = "Close"
            secondaryButton.isHidden = true
        }
    }

    @objc private func primaryAction(_ sender: Any?) {
        switch screen {
        case .welcome:
            beginInstallation()
        case .finished:
            if launchCheckbox.state == .on {
                guard launchInstalledApplication() else {
                    return
                }
            }
            NSApplication.shared.terminate(nil)
        case .failed, .cancelled:
            NSApplication.shared.terminate(nil)
        case .installing, .cancelling:
            break
        }
    }

    @objc private func secondaryAction(_ sender: Any?) {
        switch screen {
        case .welcome:
            NSApplication.shared.terminate(nil)
        case .installing:
            confirmCancellation()
        case .cancelling, .finished, .failed, .cancelled:
            break
        }
    }

    private func beginInstallation() {
        guard let resources else {
            return
        }

        do {
            let stateDirectoryURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "openhcs-installer-\(UUID().uuidString)",
                    isDirectory: true
                )
            try FileManager.default.createDirectory(
                at: stateDirectoryURL,
                withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700]
            )
            self.stateDirectoryURL = stateDirectoryURL

            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = [resources.bootstrapURL.path, resources.contractURL.path]
            var environment = ProcessInfo.processInfo.environment
            environment["OPENHCS_INSTALLER_STATE_DIRECTORY"] = stateDirectoryURL.path
            environment["OPENHCS_INSTALLER_REGISTER_MCP_CLIENTS"] =
                connectAgentsCheckbox.state == .on ? "1" : "0"
            process.environment = environment

            let pipe = Pipe()
            workerPipe = pipe
            process.standardOutput = pipe
            process.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = {
                [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty else {
                    handle.readabilityHandler = nil
                    return
                }
                DispatchQueue.main.async {
                    self?.workerOutput.append(data)
                }
            }

            process.terminationHandler = { [weak self] process in
                DispatchQueue.main.async {
                    self?.workerDidTerminate(status: process.terminationStatus)
                }
            }

            cancellationRequested = false
            apply(screen: .installing)
            startProgressPolling()
            worker = process
            try process.run()
        } catch {
            stopProgressPolling()
            worker = nil
            apply(screen: .failed)
            presentError(
                title: "Installer could not start",
                message: error.localizedDescription
            )
        }
    }

    private func startProgressPolling() {
        let timer = Timer(timeInterval: 0.25, repeats: true) {
            [weak self] _ in
            guard let self else {
                return
            }
            if let value = self.installerStateValue(named: "progress"),
                !value.isEmpty
            {
                self.statusLabel.stringValue = value
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        progressTimer = timer
    }

    private func stopProgressPolling() {
        progressTimer?.invalidate()
        progressTimer = nil
        workerPipe?.fileHandleForReading.readabilityHandler = nil
        workerPipe = nil
    }

    private func installerStateValue(named name: String) -> String? {
        guard let stateDirectoryURL else {
            return nil
        }
        let url = stateDirectoryURL.appendingPathComponent(name)
        guard
            let value = try? String(contentsOf: url, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !value.isEmpty
        else {
            return nil
        }
        return value
    }

    private func installerLogURL() -> URL? {
        guard let path = installerStateValue(named: "log-path") else {
            return nil
        }
        let url = URL(fileURLWithPath: path)
        guard
            let values = try? url.resourceValues(
                forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
            ),
            values.isRegularFile == true,
            values.isSymbolicLink != true
        else {
            return nil
        }
        return url
    }

    private func confirmCancellation() {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Cancel this installation?"
        alert.informativeText =
            "The unfinished environment will be removed. Any previous "
            + "installation remains available."
        alert.addButton(withTitle: "Cancel Installation")
        alert.addButton(withTitle: "Keep Installing")
        alert.beginSheetModal(for: window) { [weak self] response in
            if response == .alertFirstButtonReturn {
                self?.requestCancellation()
            }
        }
    }

    private func requestCancellation() {
        guard screen == .installing, let worker, worker.isRunning else {
            return
        }
        cancellationRequested = true
        apply(screen: .cancelling)
        worker.terminate()
    }

    private func workerDidTerminate(status: Int32) {
        stopProgressPolling()
        worker = nil

        if cancellationRequested
            && installerStateValue(named: "launcher-path") != nil
        {
            apply(screen: .finished)
        } else if cancellationRequested {
            apply(screen: .cancelled)
        } else if status == 0 {
            apply(screen: .finished)
        } else {
            apply(screen: .failed)
            if installerLogURL() == nil {
                presentError(
                    title: "Installation failed before logging started",
                    message: capturedWorkerMessage()
                )
            }
        }
    }

    @objc private func showLog(_ sender: Any?) {
        guard let logURL = installerLogURL() else {
            presentError(
                title: "Installer log is unavailable",
                message: capturedWorkerMessage()
            )
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([logURL])
    }

    private func launchInstalledApplication() -> Bool {
        guard let path = installerStateValue(named: "launcher-path") else {
            presentError(
                title: "Application launcher is unavailable",
                message: "The installation completed, but its launcher was not reported."
            )
            return false
        }
        if !NSWorkspace.shared.open(URL(fileURLWithPath: path)) {
            presentError(
                title: "Application could not be opened",
                message: "You can open it manually from your Applications folder."
            )
            return false
        }
        return true
    }

    private func capturedWorkerMessage() -> String {
        let text = String(data: workerOutput, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let text, !text.isEmpty else {
            return "No additional error details were reported."
        }
        return text
    }

    private func presentError(title: String, message: String) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = title
        alert.informativeText = message
        alert.beginSheetModal(for: window)
    }

}

let application = NSApplication.shared
private let controller = InstallerController()
application.setActivationPolicy(.regular)
application.delegate = controller
application.run()
