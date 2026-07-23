Preparing an MCP release
========================

An MCP release is accepted only when the ordinary OpenHCS wheel, local stdio
server, client wrappers, and generated metadata all identify the same version.
Publishing is the final step, not the validation mechanism.

The current readiness checklist and extracted-package release order are tracked
in ``docs/plans/mcp_release_readiness_20260720.md``.

Version projection
------------------

Set the intended PEP 440 version in ``openhcs/__init__.py``, then synchronize
the install-surface projections:

.. code-block:: bash

   RELEASE_VERSION=0.6.1
   python scripts/sync_mcp_release_metadata.py
   python scripts/sync_mcp_release_metadata.py --check
   python scripts/sync_mcp_release_metadata.py --check --expected-version "$RELEASE_VERSION"

The script reads the literal assignment with Python's AST and updates structured
JSON/TOML metadata. Do not edit the same version independently in the Codex
plugin, MCPB manifest, or ``server.json``.

Local validation
----------------

Run focused protocol and metadata tests, then build the actual wheel:

.. code-block:: bash

   python -m pytest tests/unit/agent/test_mcp_server.py
   python -m pytest tests/unit/test_cli.py tests/unit/test_sync_mcp_release_metadata.py
   python -m build

Install ``dist/*.whl[gui,mcp]`` into a disposable environment and run
``scripts/smoke_installed_mcp.py`` from outside the checkout. The smoke test
asserts that:

* ``openhcs.__file__`` belongs to the installed environment;
* every console script declared by the installed distribution exists and loads
  its callable entry point;
* the combined client environment contains the PyQt UI dependency;
* health succeeds over a real stdio session and reports the wheel's version;
* the health resource projection reports no missing package resource;
* every document in the packaged knowledge catalog can be read; and
* no knowledge path resolves back into the source checkout.

Client artifacts
----------------

Validate the Codex plugin and its workflow skill with the current Codex plugin
validators. Validate and pack the Claude artifact with the official ``mcpb``
CLI:

.. code-block:: bash

   MCPB_ROOT=packaging/mcpb/openhcs
   mcpb validate "$MCPB_ROOT/manifest.json"
   mcpb pack "$MCPB_ROOT" dist/openhcs.mcpb

Production MCPB artifacts require the project release certificate. A development
self-signed artifact is not a public release.

Registry preparation
--------------------

The PyPI README contains the MCP Registry ownership marker. The tag workflow's
build job validates ``server.json`` against its declared schema before
publishing the matching wheel. A dependent Registry-only job rechecks the exact
tag and generated metadata, then polls the exact-version PyPI JSON endpoint
every five seconds for up to 15 minutes until the release exposes at least one
downloadable file. Only after that signal succeeds does it download and verify
the pinned ``mcp-publisher``, run the publisher's live validator, authenticate
through GitHub Actions OIDC, and publish the metadata. No MCP Registry secret or
second release action is required. The Registry points to PyPI; it does not host
OpenHCS code.

The build/upload job has no OIDC permission. ``id-token: write`` and read-only
repository access are scoped to the dependent Registry publication job.

Registry versions are immutable. Official publication is therefore the final
tag-workflow action. If the Registry is unavailable after PyPI succeeds,
dispatch the same publish workflow with the already-published
``release_version``; its build/upload jobs remain skipped and only the
Registry job runs. Use interactive
``mcp-publisher login github`` followed by ``mcp-publisher publish`` only as a
manual recovery path, not as the normal release procedure.

CI and publication gates
------------------------

Extend the existing cross-platform and cross-Python wheel matrix rather than
creating a second release-validation system. Each matrix installation runs from
outside the checkout and tests the installed wheel. The tag-triggered publish
workflow repeats the installed-wheel smoke before upload and rejects a tag whose
version differs from the package authority or generated release metadata. Its
least-privilege dependent job completes official Registry publication after
PyPI confirms the exact release. Treat the full matrix as the release candidate
gate before creating the tag.

Tag and publish
---------------

After the release-candidate matrix is green and the release commit is on
``main``, run the repository release entry point from a clean checkout:

.. code-block:: bash

   python scripts/release.py

The script reads the version from ``openhcs/__init__.py``, requires it to be
newer than the version currently published on PyPI, rechecks all generated MCP
metadata against that version, asks for confirmation, and pushes one annotated
``v<version>`` tag. Do not create a second MCP-specific or installer-specific
tag.

That tag starts ``.github/workflows/publish.yml``. The workflow builds and
validates the Windows and macOS installer archives first, then builds and smoke
tests the OpenHCS wheel outside the checkout. After publishing the wheel and
source distribution to PyPI, it creates one GitHub Release containing those
Python artifacts plus ``OpenHCS-Windows-Installer.zip`` and
``OpenHCS-macOS-Installer.zip``. The dependent MCP Registry job waits until the
exact PyPI version is downloadable, validates the generated registry metadata,
and publishes it through GitHub OIDC.

Monitor the tag workflow at the Actions URL printed by ``scripts/release.py``.
The release is complete only when the installer-build, PyPI/GitHub Release, and
MCP Registry jobs have all succeeded; a pushed tag by itself is not completion.

External steps
--------------

Repository automation can build and validate all artifacts. These actions still
require publisher-controlled external state:

* PyPI and GitHub Release publication;
* production MCPB code signing;
* Codex/OpenAI and Claude directory submissions; and
* hosted domain, OAuth issuer, deployment credentials, and privacy/legal URLs.

Hosted Streamable HTTP staging
------------------------------

The hosted entry point is ``openhcs-mcp-http`` from the ``mcp-http`` extra.
It is a fail-closed OAuth resource server, not the local stdio server exposed
through a public port. Configure one subject-isolated instance with:

.. code-block:: text

   OPENHCS_MCP_HTTP_PUBLIC_URL=https://mcp.example.org/mcp
   OPENHCS_MCP_HTTP_ISSUER_URL=https://auth.example.org
   OPENHCS_MCP_HTTP_INTROSPECTION_URL=https://auth.example.org/oauth/introspect
   OPENHCS_MCP_HTTP_INTROSPECTION_CLIENT_ID=openhcs-resource-server
   OPENHCS_MCP_HTTP_INTROSPECTION_CLIENT_SECRET=...
   OPENHCS_MCP_HTTP_TENANT_SUBJECT=the-exact-token-subject
   OPENHCS_MCP_HTTP_REQUIRED_SCOPES=openhcs:use
   OPENHCS_MCP_HTTP_ALLOWED_HOSTS=mcp.example.org
   OPENHCS_MCP_HTTP_ALLOWED_ORIGINS=https://chatgpt.com
   OPENHCS_AGENT_READ_ROOTS=/srv/openhcs/tenant/input
   OPENHCS_AGENT_WRITE_ROOTS=/srv/openhcs/tenant/output

Token introspection must return an active token with the exact issuer, tenant
subject, resource-server audience, required scopes, client identifier, and a
future expiry. Requests fail closed if introspection is unavailable or any
claim differs. The MCP SDK serves protected-resource metadata from the issuer
and resource-server settings.

Each hosted capability invocation emits a token-free JSON audit event on the
``openhcs.mcp.audit`` logger with the configured tenant subject, nominal
capability identity, transport, outcome, and timestamp. Route that logger to a
durable restricted sink; never add bearer tokens or tool arguments to it.

Terminate TLS at a trusted reverse proxy and enforce request-size, concurrency,
rate, and idle-time limits there. Keep the Python process bound to a private
interface. Plain HTTP is accepted only for loopback development when
``OPENHCS_MCP_HTTP_ALLOW_INSECURE_LOOPBACK=1``; that switch is not a production
mode. Production deployment still requires the external authorization server,
domain, certificate, secrets, per-tenant workspace isolation, audit retention,
and connector registration.

Browser plugin artifact
-----------------------

After the public domain and OAuth deployment are live, generate the remote-only
plugin from the synchronized local release metadata:

.. code-block:: bash

   python scripts/build_hosted_mcp_plugin.py \
     --url https://mcp.example.org/mcp \
     --output-dir dist/openhcs-hosted-plugin
   python /path/to/validate_plugin.py dist/openhcs-hosted-plugin

The generated plugin contains only a remote MCP URL and read-only product
metadata. It does not bundle the local workflow skill, advertise write/UI
capabilities, or copy a tool-name allowlist. Submit this artifact for browser
distribution only after end-to-end OAuth and tenant-isolation tests pass.
