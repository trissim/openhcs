# OpenHCS desktop installers

These installers are thin, user-scoped adapters over existing authorities:

1. `DESKTOP_INSTALL_PROFILE` in `openhcs.desktop_installation` owns only
   native-install policy: the selected Python minor, desktop extras,
   binary-wheel constraints, and one reviewed uv release.
   `render_installer_contract.py` combines that policy with the project entry
   points and brand declaration to produce the native
   `installer_contract.json` projection.
2. uv installs its standalone executable without requiring Python, installs or
   locates the selected Python, creates the dedicated virtual environment, and
   installs the contract's PyPI requirement.
3. The installed `openhcs.desktop_deployment` authority projects the platform
   launcher, shortcut, and icon from the package's declared entry points and
   brand assets. Both native setup and the in-application updater invoke that
   same owner.
4. With the checked-by-default agent connection option, the installer projects
   that same stable launcher into supported local MCP clients. Client
   configuration never points at the version-stamped environment that an update
   replaces.

The platform scripts do not carry dependency lists, Python download tables,
launcher templates, shortcut construction, or an alternate OpenHCS startup
implementation. PyPI metadata, packaged brand assets, and the installed entry
points remain authoritative.

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
- Every native build renders an explicit version-pinned contract. Tag builds
  derive that version from the verified release tag before packaging assets.
- The uv bootstrap version is pinned in the shared contract. OpenHCS upgrades
  it deliberately instead of executing whichever uv release happened to become
  latest after an installer was published. The installer never creates
  antivirus exclusions. If endpoint security quarantines the official uv
  executable, installation stops with a targeted diagnostic rather than asking
  the user to disable protection or exclude the OpenHCS installation folder.

## What users see

The release assets are intended for users who do not want to work in a
terminal. No archive extraction is required: Windows users run one executable,
and macOS users open one disk image. Installation stays inside a small native
window:

- Windows presents Welcome, installation-folder, progress, and Finish pages.
  Its checked agent option connects ChatGPT desktop, Codex app/CLI/IDE, and
  detected supported local clients. The final page reports whether those
  connections succeeded and can launch OpenHCS immediately.
- macOS presents Welcome, progress, a scrollable live transcript, and Finish
  pages in ``OpenHCS Installer.app``. The transcript is the shell worker's real
  combined output, which the shell also appends to its durable log. Its
  equivalent checked agent option is shown on the Welcome page, and the final
  page can launch OpenHCS immediately.

Neither path opens a command window or asks the user to install Python, uv, or
individual OpenHCS dependencies. On macOS, detailed output is visible during
installation and remains available through the durable installer log.

After a successful agent connection, restart the local client and ask it to use
OpenHCS. ChatGPT desktop, the Codex app and CLI, and the Codex IDE extension
share one registration. In ChatGPT desktop, use **Settings > MCP servers** to
inspect the connection and ``/mcp`` after restarting to list connected servers.
Claude Desktop, Cursor, Gemini CLI, and Windsurf are configured when detected;
VS Code is registered through its supported command-line interface when
available. A client may still show its normal first-use trust or tool-approval
prompt.

## Source validation

From the repository environment:

```bash
python -m pytest -q tests/installer
RELEASE_VERSION=$(python -c 'from scripts.sync_mcp_release_metadata import read_package_version; print(read_package_version())')
python scripts/render_installer_contract.py \
  --version "$RELEASE_VERSION" \
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
application on macOS. It executes the Windows installer twice, including an
update from an environment containing a package path beyond the traditional
Windows `MAX_PATH` boundary, and executes the native macOS installer. It also
checks every selected dependency against installed OpenHCS metadata, launches
the canonical desktop command, and drives the installed MCP server through a
real stdio session. It then runs ``openhcs-mcp-demo`` from that installed wheel:
MCP generates a two-channel synthetic plate, the packaged neurite preset runs
through the real execution server, Napari receives the result, MCP validates
mounted nonzero viewer payloads, and the smoke shuts down only its dynamically
allocated TCP runtime/viewer endpoints. Before making either file a release
asset, the tag workflow resolves one annotated release tag to an exact commit
and requires successful Integration Tests and Documentation runs for that
commit. Normal and recovery installer builds then check out that same commit
before rendering the pinned contract.

Users can run the same portable acceptance after installation:

```bash
openhcs-mcp-demo --json
```

## Current distribution boundary

The release assets are intentionally simple native bootstrap packages, not
MSI/PKG system installers. When publisher credentials are configured, the
release workflow requires:

- the Windows executable has a valid SHA-256 Authenticode signature and RFC
  3161 timestamp that pass the default Authenticode verification policy; and
- the macOS application and disk image have Developer ID Application
  signatures, the application uses hardened runtime, Apple accepts the disk
  image through ``notarytool``, the ticket is stapled, and both stapling and
  Gatekeeper validation succeed.

Until those publisher credentials are available, tag and installer-recovery
workflows publish unsigned native assets and disclose that trust mode in their
workflow summaries and GitHub Release text. Providing the Windows certificate
thumbprint or macOS signing certificate selects the signed path; after that
selection, incomplete credentials or failed verification stop publication
instead of falling back to unsigned output.

Pull-request and local builds also remain unsigned so contributors do not need
publisher credentials. Adding signatures later does not change the install
contract or OpenHCS environment layout.

The Windows private key remains non-exportable in its certificate provider;
the workflow receives only its public thumbprint through a repository
variable. Apple certificate and notary credentials remain GitHub Actions
secrets. See ``docs/source/development/mcp_release.rst`` for the exact release
trust contract. No signing credential belongs in this directory or in a
generated installer artifact.
