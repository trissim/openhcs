# macOS installer adapter

The release workflow compiles `Install-OpenHCS.applescript` into an application
bundle and embeds `install-openhcs.sh` plus a release-pinned copy of the shared
installer contract. Build it on macOS with:

```bash
packaging/installers/macos/build-installer.sh \
  packaging/installers/installer_contract.json \
  "dist/OpenHCS Installer.app"
```

The installed application and Desktop shortcut delegate to the contract's
installed console entry point. The environment, managed Python, and durable log
remain under the current user's Library directories. The adapter never asks for
administrator privileges or uses the system Python.
