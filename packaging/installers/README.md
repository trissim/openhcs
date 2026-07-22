# OpenHCS desktop installers

These installers are thin, user-scoped adapters over existing authorities:

1. `installer_contract.json` selects a supported Python minor, the published
   OpenHCS requirement, the installed GUI entry point, and official uv
   bootstrap URLs.
2. uv installs its standalone executable without requiring Python, installs or
   locates the selected Python, creates the dedicated virtual environment, and
   installs the contract's PyPI requirement.
3. The platform adapter creates a desktop launcher for the installed
   `openhcs-gui` console script with `OPENHCS_CPU_ONLY=true`.

The platform scripts do not carry dependency lists, Python download tables, or
an alternate OpenHCS startup implementation. PyPI metadata and the installed
entry point remain authoritative.

## Installation model

- Installation is per-user and does not modify system Python or require an
  administrator account.
- Re-running an installer updates/reinstalls the same isolated environment.
- The default is the CPU-safe `openhcs[gui]` surface. GPU and optional viewer
  stacks remain explicit post-install choices because their system requirements
  cannot be inferred safely.
- A durable log is kept in the platform's OpenHCS user log directory and is
  shown when installation fails.
- The source-tree contract installs the latest published compatible OpenHCS.
  Tag builds render a copy pinned to that release before packaging the assets.

## Source validation

From the repository environment:

```bash
python -m pytest -q tests/installer
python scripts/render_installer_contract.py \
  --version 0.5.22 \
  --output /tmp/openhcs-installer-contract.json
```

Platform-specific source, build, and launch instructions are documented beside
each adapter.

## Release assets

Tag publication renders a contract pinned to the tag version and attaches two
archives to the GitHub release:

- `OpenHCS-Windows-Installer.zip` contains the contract plus
  `Install-OpenHCS.cmd` and its PowerShell implementation. Extract the archive
  and double-click the CMD file.
- `OpenHCS-macOS-Installer.zip` contains a compiled `OpenHCS Installer.app`
  with the bootstrap and pinned contract embedded as application resources.

Pull-request CI parses the Windows PowerShell source on Windows and compiles the
AppleScript application on macOS. The tag workflow repeats those gates before
making either archive a release asset.

## Current distribution boundary

The first release assets are intentionally simple native bootstrap packages,
not MSI/PKG system installers. They are not code-signed or notarized yet, so
Windows SmartScreen or macOS Gatekeeper may require the user to confirm that
the downloaded asset is trusted. Signing/notarization can be added without
changing the install contract or OpenHCS environment layout.
