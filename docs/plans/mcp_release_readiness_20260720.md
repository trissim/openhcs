# OpenHCS MCP release readiness

Status: **repository gate prepared; final PyPI-only acceptance verified**.

This plan separates repository work that is complete from publisher-controlled
steps and deployment work that still block a public release. The extracted
owner packages were published in dependency order for clean-wheel acceptance;
no OpenHCS package, MCP Registry entry, plugin, connector, or OpenHCS GitHub
Release was published during this preparation.

## Implemented release surfaces

- The ordinary OpenHCS wheel owns the engine, local MCP server, optional PyQt
  UI, and viewer runtimes. The dependency-free prebuild gate checks only static
  package/version structure. After installing the built wheel, the full gate
  derives desktop extras from the actual selected capability registry and
  projects `openhcs[gui,mcp,viz]` without a second capability or extra lattice.
- `openhcs.__version__` is projected by
  `scripts/sync_mcp_release_metadata.py` into the Codex plugin, Claude MCPB,
  and MCP Registry metadata.
- The Codex local plugin, `use-openhcs` skill, MCPB manifest, and `server.json`
  validate against their current tooling/schema.
- The wheel build projects the manifest-declared knowledge corpus into package
  resources without checking in a second document tree.
- The existing Linux/macOS/Windows integration matrix covers Python 3.11-3.13.
  Its Python 3.11/3.12 PyPI installed-wheel cells install the combined GUI/MCP
  product and run the real stdio protocol outside the source checkout; the
  documented OMERO subset remains limited to 3.11/3.12 by ZeroC Ice wheels.
- Hosted Streamable HTTP is a separate, fail-closed access boundary. The
  universal plugin mode is anonymous and private deployments may enable
  subject-isolated OAuth introspection; both reject any non-read-only hosted
  declaration. Remote exposure comes from nominal capability inheritance, not
  a copied tool list.
- The browser plugin is generated only after a public HTTPS endpoint is known.

## Dependency isolation and extracted-package release train

Published `openhcs==0.5.21` exact-pins the previous eight owner releases, so
publishing the candidate owner versions cannot change existing or fresh 0.5.21
installs.  The 0.5.22 candidate uses explicit lower bounds on the new versions;
only the new OpenHCS release can select them.

All eight owner versions below are now available from PyPI. They were published
in this dependency order:

1. `metaclass-registry 0.1.5`
2. `python-introspect 0.1.5`, then `ObjectState 1.0.18`
3. `arraybridge 0.2.11`
4. `zmqruntime 0.1.10`, then `PolyStore 0.1.12`
5. `pycodify 0.1.3`
6. `pyqt-reactive 0.1.22` after its ObjectState and python-introspect owners
7. OpenHCS last

`pycodify 0.1.3` is required because OpenHCS consumes the post-0.1.2 immutable
render-context extension API; the OpenHCS floor now names that candidate.
`scripts/validate_local_release_floors.py` discovers package names,
versions, and dependencies from PEP 621 metadata and rejects stale OpenHCS
floors or unsatisfied local dependency edges. It contains no API or feature
mirror.

The first diagnostic PyPI-only failure was `python-introspect 0.1.4`, which
lacks the current `signature_analysis_target` export. A local-candidate overlay
proved the dependency graph and installed MCP path before publication, but it is
not release evidence. Final acceptance subsequently resolved every owner from
PyPI, installed the `0.5.22` wheel outside the checkout, and passed the real
stdio MCP protocol and packaged-resource smoke. The smoke reports its
manifest-derived knowledge-document and distribution-derived console-entrypoint
counts rather than copying those authorities into this plan.

## Final local-package release gate

For each extracted package:

1. Review and commit that package's own dirty tree independently.
2. Run its native unit/build checks.
3. Publish the version above to PyPI.
4. Verify its wheel metadata and import surface from a clean environment.
5. Advance only after downstream candidate installation resolves from PyPI.

All eight are now available. Run the existing OpenHCS cross-platform matrix
without local wheel overlays as the ordinary pre-merge CI gate. The final local
acceptance installed the synchronized desktop wheel extras from PyPI-only
dependencies and exercised
the installed stdio server outside the checkout.

The final transport/storage owner refresh passed 41 ZMQRuntime tests and 251
PolyStore tests (with 13 declared Zarr capability skips). Clean PyPI installs
resolved `zmqruntime==0.1.10` and `polystore==0.1.12`; the installed transport
exposed the viewer data/control socket and response-serialization APIs consumed
by OpenHCS. OpenHCS 0.5.21 remains isolated because its published metadata
exact-pins the earlier owner train.

## Pre-merge OpenHCS publication gate

1. Confirm all eight exact candidate versions are downloadable from PyPI in a
   clean environment, with no local wheel overlay or editable submodule.
2. Confirm the package authority is the intended final PEP 440 version
   (`0.5.22` for this release).
3. Run `scripts/sync_mcp_release_metadata.py` and its static `--check` mode so the
   Codex plugin, MCPB wrapper, and `server.json` are committed at that same final
   version before merge.
4. Run the dependency-floor preflight, documentation validator, focused MCP
   suites, and the complete existing integration matrix.
5. Build sdist and wheel from the merge candidate, install the extras printed
   by `--print-desktop-extras` outside the checkout, run the full
   `--check --capability-requirements` gate, and then run
   `scripts/smoke_installed_mcp.py`. The smoke derives installed entry points
   from wheel metadata and reads every manifest-declared knowledge document.
6. Validate the Codex plugin and skill, validate/pack/sign MCPB, and validate
   `server.json` against its declared official MCP Registry schema.
7. Merge only after the final version projection and clean-wheel gate are green.

The package authority and every generated MCP distribution projection agree on
final version `0.5.22`. Clean PyPI-only owner resolution and the built-wheel
protocol smoke are complete; the existing cross-platform workflow remains the
normal merge gate.

## Post-merge publication action

When the pre-merge gate is complete, the normal annotated tag is the only
repository action required to publish OpenHCS:

```bash
git tag -a v0.5.22 -m "Release OpenHCS 0.5.22"
git push origin v0.5.22
```

The tag workflow's build job independently checks that the tag, package
authority, Codex plugin, MCPB wrapper, and `server.json` agree; validates owner
floors and the official Registry schema; rebuilds the artifacts; installs the
synchronized desktop extras, validates them against the installed capability
registry, and runs the real stdio MCP protocol smoke;
and only then uploads to PyPI and creates the GitHub Release. A dependent
Registry-only job rechecks the exact tag and generated metadata, polls the
exact-version PyPI JSON endpoint every five seconds for up to 15 minutes until
it exposes a downloadable release file, downloads and verifies a pinned
`mcp-publisher`, runs its live validator, authenticates with GitHub Actions
OIDC, and publishes the synchronized Registry record as its final action. The
build/upload job has no OIDC permission; read-only repository access and
`id-token: write` are scoped to the dependent Registry job. No separate MCP
Registry secret or normal post-tag command is required.

The Registry record must follow the wheel because it pins the exact PyPI
version; it is not a substitute for the wheel release gate. Registry versions
are immutable, so the publisher step intentionally has no later workflow action
that could fail after a successful official publication. Interactive Registry
login and publication remain a recovery path only if the tag workflow cannot be
rerun after an external service failure.

The MCP SDK remains pinned to the stable `mcp>=1.28,<2` line. MCP SDK v2 should
be a deliberate compatibility migration after its stable release, not an
unbounded dependency update in this release train.

## External blockers

- OpenHCS PyPI, GitHub Release, and official MCP Registry publication through
  the post-merge tag workflow.
- A production MCPB signing certificate and signed Claude artifact.
- Codex/plugin marketplace and Claude directory submission/approval.
- External availability of PyPI, GitHub OIDC, and the official MCP Registry
  during the tag workflow.
- A hosted domain, TLS certificate, reverse-proxy limits, durable audit sink,
  public privacy/legal/support URLs, and end-to-end protocol/load tests.
- Private hosted operation additionally requires an OAuth issuer/introspection
  client, secret store, tenant workspace provisioning, and tenant-isolation
  tests; those are not prerequisites for the public read-only plugin.
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
