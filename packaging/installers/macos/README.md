# macOS installer application

The release workflow compiles `OpenHCSInstaller.swift` into a universal native
AppKit application and embeds `install-openhcs.sh` plus a release-pinned copy of
the shared installer contract. Build it on macOS with:

```bash
packaging/installers/macos/build-installer.sh \
  packaging/installers/installer_contract.json \
  "dist/OpenHCS Installer.app"
```

The application presents a persistent Welcome, Progress, and Finish flow. The
shell worker runs without opening Terminal, reports live progress back to the
application, and retains the existing verified-candidate update transaction.
Cancel leaves any current installation in place. Failures offer the durable log,
and Finish can launch the installed application.

The installed Applications launcher and Desktop shortcut delegate to the
contract's installed entry point. The environment, managed Python, and durable
log remain under the current user's Library directories. The installer never
asks for administrator privileges or uses the system Python.
