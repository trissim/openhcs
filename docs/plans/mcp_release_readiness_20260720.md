# OpenHCS MCP release readiness

Status: **prepared, not ready to publish**.

This plan separates repository work that is complete from publisher-controlled
steps and deployment work that still block a public release. No package,
registry entry, plugin, connector, or GitHub Release was published during this
preparation.

## Implemented release surfaces

- The ordinary OpenHCS wheel owns the engine, local MCP server, and optional
  PyQt UI; client wrappers install `openhcs[gui,mcp]` rather than another
  implementation.
- `openhcs.__version__` is projected by
  `scripts/sync_mcp_release_metadata.py` into the Codex plugin, Claude MCPB,
  and MCP Registry metadata.
- The Codex local plugin, `use-openhcs` skill, MCPB manifest, and `server.json`
  validate against their current tooling/schema.
- The wheel build projects the manifest-declared knowledge corpus into package
  resources without checking in a second document tree.
- The existing Python 3.11/3.12, Linux/macOS/Windows wheel matrix installs the
  combined GUI/MCP product and runs the real stdio protocol outside the source
  checkout.
- Hosted Streamable HTTP is a separate, fail-closed OAuth resource-server
  boundary. Remote exposure comes from nominal capability inheritance, not a
  copied tool list.
- The browser plugin is generated only after a public HTTPS endpoint is known.

## Required extracted-package release train

The current OpenHCS wheel cannot install purely from PyPI until eight candidate
packages are published. The safe dependency order is:

1. `metaclass-registry 0.1.5`
2. `arraybridge 0.2.11`, then `PolyStore 0.1.10`
3. `python-introspect 0.1.5`, then `ObjectState 1.0.18`, then
   `pyqt-reactive 0.1.22`
4. `zmqruntime 0.1.9` and `pycodify 0.1.3` at any point before OpenHCS
5. OpenHCS last

`pycodify 0.1.3` is required because OpenHCS consumes the post-0.1.2 immutable
render-context extension API; the OpenHCS floor now names that candidate.
`scripts/validate_local_release_floors.py` discovers package names,
versions, and dependencies from PEP 621 metadata and rejects stale OpenHCS
floors or unsatisfied local dependency edges. It contains no API or feature
mirror.

The first clean PyPI-only failure was `python-introspect 0.1.4`, which lacks the
current `signature_analysis_target` export. A one-shot install using the eight
local candidate wheels passed with 117 compatible packages, MCP health `ok`,
all 43 knowledge documents, PyQt/QScintilla offscreen import, and all six
console entrypoints.

## Final local-package release gate

For each extracted package:

1. Review and commit that package's own dirty tree independently.
2. Run its native unit/build checks.
3. Publish the version above to PyPI.
4. Verify its wheel metadata and import surface from a clean environment.
5. Advance only after downstream candidate installation resolves from PyPI.

After all eight are available, run the existing OpenHCS cross-platform matrix
without local wheel overlays. A pure-PyPI `wheel[gui,mcp]` smoke on every
supported Python/OS cell is the dependency-release acceptance test.

## OpenHCS publication gate

1. Replace `0.5.22.dev0` with the intended final PEP 440 version.
2. Run `scripts/sync_mcp_release_metadata.py` and its `--check` mode.
3. Run the dependency-floor preflight, documentation validator, focused MCP
   suites, and the complete existing integration matrix.
4. Build sdist and wheel, install `wheel[gui,mcp]` outside the checkout, and run
   `scripts/smoke_installed_mcp.py`.
5. Validate the Codex plugin and skill, validate/pack/sign MCPB, and validate
   `server.json`.
6. Create the release tag only after every candidate gate is green. The publish
   workflow rejects a tag/version projection mismatch and repeats the installed
   wheel smoke before upload.

The MCP SDK remains pinned to the stable `mcp>=1.28,<2` line. MCP SDK v2 should
be a deliberate compatibility migration after its stable release, not an
unbounded dependency update in this release train.

## External blockers

- PyPI publication authority for the eight extracted packages and OpenHCS.
- A production MCPB signing certificate and signed Claude artifact.
- Codex/plugin marketplace and Claude directory submission/approval.
- Official MCP Registry authentication and publication after the PyPI wheel is
  live.
- A hosted domain, TLS certificate, OAuth issuer/introspection client, secret
  store, tenant workspace provisioning, reverse-proxy limits, durable audit
  sink, privacy/legal URLs, and end-to-end tenant-isolation/load tests.
- Browser plugin generation and submission using the final hosted HTTPS URL.

## Residual risks to track

- Windows descriptor security currently relies on inherited user-directory
  ACLs; explicit ACL ownership verification is not implemented.
- Linux can discover bridge descriptors through `/proc`; Windows requires the
  shared/configured descriptor directory or an explicit descriptor path.
- `pyqt-reactive` makes PyQt6 transitive in the base dependency graph. MCP cold
  startup is headless and imports no PyQt module, but reducing base installation
  weight requires a future dependency split.
- Native Windows behavior is covered by the existing CI matrix but was not
  executed on the Linux development host during this preparation.
