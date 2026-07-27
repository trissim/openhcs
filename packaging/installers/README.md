# OpenHCS desktop installers

These installers are thin, user-scoped adapters over existing authorities:

1. `installer_contract.json` selects a supported Python minor, the published
   OpenHCS requirement, the installed GUI entry point, and official uv
   bootstrap URLs.
2. uv installs its standalone executable without requiring Python, installs or
   locates the selected Python, creates the dedicated virtual environment, and
   installs the contract's PyPI requirement.
3. The platform adapter creates a desktop launcher for the installed canonical
   `openhcs` console script with `OPENHCS_CPU_ONLY=true`; that dispatcher opens
   the GUI by default.
4. With the checked-by-default agent connection option, the installer projects
   that same stable launcher into supported local MCP clients. Client
   configuration never points at the version-stamped environment that an update
   replaces.

The platform scripts do not carry dependency lists, Python download tables, or
an alternate OpenHCS startup implementation. PyPI metadata and the installed
entry point remain authoritative.

## Installation model

- Installation is per-user and does not modify system Python or require an
  administrator account.
- Re-running an installer updates/reinstalls the same isolated environment.
- Existing unrelated MCP client configuration is preserved. Setup owns only the
  local server entry named `openhcs`, and keeps a recoverable backup when it
  changes an existing client configuration.
- The default is the CPU-safe
  `openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]` desktop surface: the Qt
  application, Napari, Fiji/PyImageJ, Bio-Formats, supported CellProfiler
  compatibility libraries, and the local MCP server. GPU dependencies remain
  an explicit post-install choice because the host's CUDA environment cannot
  be inferred safely.
- PyImageJ resolves and caches the Fiji/Bio-Formats Java distribution on first
  Fiji or Bio-Formats use; the installer does not embed a standalone `Fiji.app`.
- A durable log is kept in the platform's OpenHCS user log directory and is
  shown when installation fails.
- The source-tree contract installs the latest published compatible OpenHCS.
  Tag builds render a copy pinned to that release before packaging the assets.

## What users see

The release assets are intended for users who do not want to work in a
terminal. No archive extraction is required: Windows users run one executable,
and macOS users open one disk image. Installation stays inside a small native
window:

- Windows presents Welcome, installation-folder, progress, and Finish pages.
  Its checked agent option connects the Codex app/CLI/IDE and detected supported
  local clients. The final page reports whether those connections succeeded
  and can launch OpenHCS immediately.
- macOS presents Welcome, progress, and Finish pages in
  ``OpenHCS Installer.app``. Its equivalent checked agent option is shown on the
  Welcome page, and the final page can launch OpenHCS immediately.

Neither path opens a command window or asks the user to install Python, uv, or
individual OpenHCS dependencies. Advanced output remains available through the
durable installer log when troubleshooting is needed.

After a successful agent connection, restart the local client and ask it to use
OpenHCS. The Codex app, Codex CLI, and the Codex IDE extension share one
registration. In the current unified ChatGPT desktop app, this is the distinct
Codex view; select Codex before asking it to use OpenHCS. Claude Desktop,
Cursor, Gemini CLI, and Windsurf are configured when detected; VS Code is
registered through its supported command-line interface when available. A
client may still show its normal first-use trust or tool-approval prompt. This
local Codex setup does not use ChatGPT Developer Mode.

The Chat and Work views in ChatGPT do not directly start this local stdio
process or read Codex's local configuration. Those views require a remote HTTPS
MCP app or an OpenAI Secure MCP Tunnel, with availability and approval
controlled by the user's ChatGPT plan and workspace. The official MCP Registry
record provides distribution discovery and verification, not a
browser-to-local bridge.

## Source validation

From the repository environment:

```bash
python -m pytest -q tests/installer
python scripts/render_installer_contract.py \
  --version 0.6.4 \
  --output /tmp/openhcs-installer-contract.json
```

Platform-specific source, build, and launch instructions are documented beside
each adapter.

## Release assets

Tag publication renders a contract pinned to the tag version and attaches two
directly usable files to the GitHub release:

- `OpenHCS-Windows-Installer.exe` is a small GUI-subsystem executable with its
  PowerShell worker and pinned contract embedded. Double-click the downloaded
  file.
- `OpenHCS-macOS-Installer.dmg` contains the compiled
  `OpenHCS Installer.app`, whose bootstrap and pinned contract are embedded as
  application resources. Open the downloaded disk image, then open the
  application.

Pull-request CI parses the Windows PowerShell source and compiles the
GUI-subsystem launcher on Windows, and compiles the universal Swift/AppKit
application on macOS. It also executes both native installers,
checks every selected dependency against installed OpenHCS metadata, launches
the canonical desktop command, and drives the installed MCP server through a
real stdio session. It then runs ``openhcs-mcp-demo`` from that installed wheel:
MCP generates a two-channel synthetic plate, the packaged neurite preset runs
through the real execution server, Napari receives the result, MCP validates
mounted nonzero viewer payloads, and the smoke shuts down only its dynamically
allocated TCP runtime/viewer endpoints. The tag workflow repeats the source
gates before making either file a release asset.

Users can run the same portable acceptance after installation:

```bash
openhcs-mcp-demo --json
```

## Current distribution boundary

The release assets are intentionally simple native bootstrap packages, not
MSI/PKG system installers. Production tag publication now fails closed unless:

- the Windows executable has a valid SHA-256 Authenticode signature and RFC
  3161 timestamp that pass the default Authenticode verification policy; and
- the macOS application and disk image have Developer ID Application
  signatures, the application uses hardened runtime, Apple accepts the disk
  image through ``notarytool``, the ticket is stapled, and both stapling and
  Gatekeeper validation succeed.

Pull-request and local builds remain unsigned so contributors do not need
publisher credentials. Existing release assets created before this trust
workflow remain unsigned; adding signatures does not change the install
contract or OpenHCS environment layout.

The private keys, certificates, passwords, and Apple API key are GitHub Actions
secrets. See ``docs/source/development/mcp_release.rst`` for the exact release
credential contract. No signing credential belongs in this directory or in a
generated installer artifact.
