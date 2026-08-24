# macOS installer application

The release workflow compiles `OpenHCSInstaller.swift` into a universal native
AppKit application and embeds `install-openhcs.sh` plus a release-pinned copy of
the shared installer contract. Build it on macOS with:

```bash
RELEASE_VERSION=$(python -c 'from scripts.sync_mcp_release_metadata import read_package_version; print(read_package_version())')
python -m scripts.render_installer_contract \
  --version "$RELEASE_VERSION" \
  --output /tmp/openhcs-installer-contract.json
packaging/installers/macos/build-installer.sh \
  /tmp/openhcs-installer-contract.json \
  "dist/OpenHCS Installer.app"
```

The application presents a persistent Welcome, Progress, and Finish flow. Its
scrollable output pane streams the real combined standard output and error from
the shell worker while that same stream is appended to the durable installer
log. The application does not invent or mirror installation stages: the shell
still owns progress messages, logging, and the verified-candidate update
transaction. Cancel leaves any current installation in place. The durable log
remains available after success, cancellation, or failure, and Finish can launch
the installed application.

The installed Applications launcher and Desktop shortcut are projected by the
installed package's desktop-deployment authority. Native setup and in-app
updates therefore rebuild the same app bundle, icon, environment launcher, and
Desktop link from one implementation. The environment, managed Python, and
durable log remain under the current user's Library directories. The installer
never asks for administrator privileges or uses the system Python.

Local builds from ``build-installer.sh`` are intentionally unsigned. Until an
Apple Developer ID certificate is configured, release workflows also publish
the unsigned disk image with an explicit trust warning. Once the certificate
is configured, the workflow signs the resulting app with a Developer ID
Application identity and hardened runtime before placing it in the disk image.
It then signs the disk image, submits that exact image through ``notarytool
--wait``, requires an ``Accepted`` response, staples and validates the ticket,
and runs the Gatekeeper assessment before upload. Selecting that signed path is
fail-closed: incomplete credentials or any failed trust operation stop the
release instead of silently downgrading it to unsigned output.

If macOS blocks an official OpenHCS bootstrap because it is unsigned and not
notarised, first try to open **OpenHCS Installer.app** so macOS records the
attempt. Then open **System Settings > Privacy & Security**, scroll to
**Security**, click **Open Anyway**, authenticate, and confirm **Open**. Only
override this protection for the disk image downloaded from the official
OpenHCS GitHub release. Apple makes **Open Anyway** available for about an hour
after the blocked launch attempt; see
[Apple's Gatekeeper instructions](https://support.apple.com/guide/mac-help/open-an-app-by-overriding-security-settings-mh40617/mac).
