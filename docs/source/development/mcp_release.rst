Preparing an MCP release
========================

An MCP release is accepted only when the ordinary OpenHCS wheel, local stdio
server, client wrappers, and generated metadata all identify the same version.
Publishing is the final step, not the validation mechanism.

``docs/plans/mcp_release_readiness_20260720.md`` records the historical 0.5.22
release preparation. It is not the current checklist or a product-fact
authority; the procedure below and the referenced scripts/workflow define the
present release path.

Version projection
------------------

Set the intended PEP 440 version in ``openhcs/__init__.py``, then synchronize
the install-surface projections:

.. code-block:: bash

   RELEASE_VERSION=$(python -c 'from scripts.sync_mcp_release_metadata import read_package_version; print(read_package_version())')
   python scripts/sync_mcp_release_metadata.py
   python scripts/sync_mcp_release_metadata.py --check
   python scripts/sync_mcp_release_metadata.py --check --expected-version "$RELEASE_VERSION"

The script reads the literal assignment with Python's AST and updates structured
JSON/TOML metadata. This dependency-free phase validates versions, package
identity, declared extra names, and agreement across the three packaging
surfaces. Do not edit the same version independently in the Codex plugin, MCPB
manifest, or ``server.json``.

Local validation
----------------

Run focused protocol and metadata tests, then build the actual wheel:

.. code-block:: bash

   python -m pytest tests/unit/agent/test_mcp_server.py
   python -m pytest tests/unit/test_cli.py tests/unit/test_sync_mcp_release_metadata.py
   python -m build

Read the synchronized desktop extras, install them from the built wheel into a
disposable environment, and then run the full registry-derived metadata check:

.. code-block:: bash

   DESKTOP_EXTRAS=$(python scripts/sync_mcp_release_metadata.py --print-desktop-extras)
   pip install "${WHEEL}[${DESKTOP_EXTRAS}]"
   python scripts/sync_mcp_release_metadata.py --check --capability-requirements

Run ``scripts/smoke_installed_mcp.py`` from outside the checkout. The smoke test
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

The matrix has one dependency-readiness gate. It fetches the recorded submodule
release tags, validates the local dependency release floors, waits until the
exact wheels are visible through PyPI's installer-facing index, and projects
those exact requirements into downstream candidate installs. This gate prevents
a green source checkout from standing in for unpublished dependency releases.
The source gates also run the maintained PyQt workflow suite against the exact
pinned pyqt-reactive candidate. Published-dependency and installed-wheel jobs
remain the proof for the public package boundary.

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
validates the Windows and macOS installer assets first, then builds and smoke
tests the OpenHCS wheel outside the checkout. After publishing the wheel and
source distribution to PyPI, it creates one GitHub Release containing those
Python artifacts plus the directly runnable
``OpenHCS-Windows-Installer.exe`` and the single-file
``OpenHCS-macOS-Installer.dmg``. The dependent MCP Registry job waits until the
exact PyPI version is downloadable, validates the generated registry metadata,
and publishes it through GitHub OIDC.

Monitor the tag workflow at the Actions URL printed by ``scripts/release.py``.
The release is complete only when the installer-build, PyPI/GitHub Release, and
MCP Registry jobs have all succeeded; a pushed tag by itself is not completion.

Recovery publication
--------------------

The manual workflow can publish Python/MCP distributions or attach desktop
installers to an existing version tag. To publish the Python and MCP
distributions:

.. code-block:: bash

   RELEASE_VERSION=$(python -c 'from scripts.sync_mcp_release_metadata import read_package_version; print(read_package_version())')
   gh workflow run publish.yml \
     --field release_version="$RELEASE_VERSION" \
     --field publish_python_package=true

This explicit manual path validates and smoke-tests the built wheel, publishes
the wheel and source distribution to PyPI, then publishes the matching official
MCP Registry entry. It does not create a tag, GitHub Release, or native
installer. PyPI upload remains idempotent through ``--skip-existing``.

To build native installers from the current release workflow and attach them
to an existing version tag without republishing PyPI or the MCP Registry:

.. code-block:: bash

   gh workflow run publish.yml \
     --field release_version="$RELEASE_VERSION" \
     --field publish_python_package=false \
     --field publish_desktop_installers=true

This recovery path is useful when package publication completed before native
assets. It requires the version tag to exist and renders the installer contract
for that exact version. The workflow verifies the remote
``refs/tags/v<release_version>`` before any recovery build, so a manual dispatch
cannot manufacture release assets for an untagged version.

Native installer signing
------------------------

Pull-request and local installer builds are intentionally unsigned. Production
tag and installer-recovery builds also publish unsigned native assets while
publisher credentials are absent, and disclose that mode in the workflow
summary and GitHub Release text. This temporary bootstrap policy keeps the
native installers available before certificate enrollment is complete.

Signing remains fail-closed after it is selected. A configured Windows
certificate thumbprint routes the job to the signing runner. A configured
macOS certificate selects Developer ID signing and notarization. In either
case, an inaccessible key, incomplete remaining configuration, expired
certificate, or failed validation stops the release rather than silently
downgrading to unsigned output.

Windows uses the same low-cost certificate-store route as Fiji's Jaunch
launchers. The production private key is non-exportable and remains in Certum
SimplySign or another Windows-compatible hardware/cloud provider. SignTool
selects the corresponding certificate from the current Windows user's
``My`` store by its exact SHA-1 thumbprint. Configure that public thumbprint as
one repository variable, not a secret:

.. code-block:: text

   OPENHCS_WINDOWS_SIGNING_CERTIFICATE_THUMBPRINT

When the thumbprint is configured, the Windows job targets a self-hosted runner
with all of these labels:

.. code-block:: text

   self-hosted
   windows
   x64
   openhcs-signing

Without the thumbprint, the same installer source is built on
``windows-latest`` and uploaded unsigned with a workflow warning.

Prepare that runner under the same interactive Windows user that owns the
certificate-store projection:

1. Install Windows SDK Signing Tools, Certum SimplySign Desktop (or the chosen
   provider's equivalent), and the GitHub Actions runner.
2. Connect SimplySign from the desktop session and confirm that the selected
   certificate and private key appear in ``Cert:\CurrentUser\My``.
3. Assign the runner's custom ``openhcs-signing`` label and start the runner
   interactively from that connected session. Do not run it as another user or
   a background service that cannot access the virtual card or PIN dialog.
4. Push the release tag. A PIN-backed card may prompt during SignTool; a
   pinless card signs immediately. The signing step has a bounded timeout and
   the release fails if access is not authorized.

Certum documents SimplySign Desktop plus its mobile application as required to
link the cloud certificate to the signing computer. It documents SignTool
selection by ``/sha1 <thumbprint>`` and states that PIN-backed cards show a PIN
dialog while pinless cards sign without that prompt. It does not document a
safe unattended GitHub-hosted-runner login flow, so OpenHCS does not claim or
configure one.

The helper validates the exact certificate-store object, accessible private
key, validity period, and code-signing EKU before signing the existing
``OpenHCS-Windows-Installer.exe``. It uses SHA-256 with Certum's RFC 3161
timestamp service, runs SignTool with the default Authenticode policy, and then
requires PowerShell's native signature object to report ``Valid``, the exact
configured signer thumbprint, and a timestamp-authority certificate.

The same helper can be invoked outside Actions after SimplySign is connected:

.. code-block:: powershell

   $env:OPENHCS_WINDOWS_SIGNING_CERTIFICATE_THUMBPRINT = "<40-character thumbprint>"
   .\packaging\installers\windows\Sign-Installer.ps1 `
       -ArtifactPath .\OpenHCS-Windows-Installer.exe

The macOS job requires a Developer ID Application certificate, not a Developer
ID Installer certificate, because the shipped objects are an application and
disk image rather than a flat installer package. It also requires a team App
Store Connect API key authorized for the Apple notary service; Apple does not
permit individual App Store Connect API keys to use ``notarytool``. Store:

.. code-block:: text

   OPENHCS_MACOS_SIGNING_CERTIFICATE_BASE64
   OPENHCS_MACOS_SIGNING_CERTIFICATE_PASSWORD
   OPENHCS_MACOS_DEVELOPER_IDENTITY
   OPENHCS_MACOS_NOTARY_KEY_BASE64
   OPENHCS_MACOS_NOTARY_KEY_ID
   OPENHCS_MACOS_NOTARY_ISSUER_ID

The certificate secret is the base64 encoding of the P12 containing the
Developer ID Application certificate and private key. The identity secret is
the full identity label shown by ``security find-identity -v -p codesigning``.
The notary-key secret is the base64 encoding of the App Store Connect ``.p8``
key; the key ID and issuer ID are the corresponding App Store Connect values.

The workflow imports the certificate into an ephemeral keychain, signs the app
with hardened runtime and a secure timestamp, verifies that signature, builds
and signs the DMG, and submits the exact DMG with ``notarytool --wait``. It
requires the returned status to be ``Accepted``, retrieves the authoritative
log for that exact submission with the same credentials, emits the complete
log so Apple warnings remain visible, and independently requires the log status
to be ``Accepted``. Only then does it staple and validate the ticket and run a
Gatekeeper assessment before upload. Temporary keychains, certificate files,
notarization logs, and API-key files are removed by the trust helper.

The DMG is the trust acceptance boundary because it is the exact outermost
file distributed to users. Apple directs custom distributions to sign nested
code from the inside out, notarize only the outermost supported container, and
staple that distributed container. Submitting a DMG also generates tickets for
its nested code. The workflow therefore verifies the enclosed app before
building the DMG, notarizes and staples the signed DMG, rechecks the final DMG
signature and structure after stapling, and assesses that DMG with ``spctl
--type open --context context:primary-signature``. ``spctl`` uses the same
security-assessment policy subsystem as Gatekeeper.

Publisher setup should follow the current primary platform guidance:

* `Microsoft Authenticode timestamping and SignTool
  <https://learn.microsoft.com/en-us/windows/win32/seccrypto/time-stamping-authenticode-signatures>`_.
* `Microsoft PowerShell Authenticode signature inspection
  <https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/get-authenticodesignature>`_.
* `Certum Open Source Code Signing in the Cloud
  <https://shop.certum.eu/open-source-code-signing-on-simplysign.html>`_.
* `Certum SignTool cloud-signing instructions
  <https://support.certum.eu/en/signing-the-code-using-tools-like-signtool-and-jarsigner-instruction/>`_.
* `Jaunch's Fiji-compatible Windows signing guide
  <https://github.com/apposed/jaunch/blob/5dbbcb8b865aaeb4f0a1c508d8bfc73f3ff0d0cf/doc/WINDOWS.md>`_.
* `GitHub self-hosted runner labels
  <https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow>`_.
* `Apple Developer ID
  <https://developer.apple.com/developer-id/>`_.
* `Apple custom notarization workflow
  <https://developer.apple.com/documentation/security/customizing-the-notarization-workflow>`_.
* `Apple App Store Connect API key types
  <https://developer.apple.com/documentation/appstoreconnectapi/creating-api-keys-for-app-store-connect-api>`_.
* `Apple packaging guidance for nested distributions
  <https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution>`_.
* `Apple Gatekeeper assessment with ``spctl``
  <https://developer.apple.com/library/archive/technotes/tn2206/_index.html>`_.

Do not test the production path with self-signed credentials: that would prove
only file mutation, not the user-facing Windows or macOS trust chain. After
preparing the real Windows signing host and Apple secrets, validate them with a
release candidate tag only after the normal integration matrix is green.

External steps
--------------

Repository automation can build and validate all artifacts. These actions still
require publisher-controlled external state:

* PyPI and GitHub Release publication;
* production MCPB code signing;
* Codex and Claude directory submissions.
