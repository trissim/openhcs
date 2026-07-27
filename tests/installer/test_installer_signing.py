"""Fail-closed trust gates for production native installer artifacts."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"
INTEGRATION_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
)
WINDOWS_SIGNING_PATH = (
    REPOSITORY_ROOT / "packaging" / "installers" / "windows" / "Sign-Installer.ps1"
)
MACOS_SIGNING_PATH = (
    REPOSITORY_ROOT / "packaging" / "installers" / "macos" / "Sign-Installer.sh"
)


def test_windows_release_signs_and_verifies_the_constructed_executable() -> None:
    source = WINDOWS_SIGNING_PATH.read_text(encoding="utf-8")

    assert "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_BASE64" in source
    assert "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_PASSWORD" in source
    assert "[Convert]::FromBase64String($certificateBase64)" in source
    assert '"openhcs-authenticode-$([Guid]::NewGuid()' in source
    assert "Resolve-SignToolPath" in source
    assert '"Windows Kits"' in source
    assert '"signtool.exe"' in source

    sign = source.index("$signArguments = @(")
    invoke_sign = source.index("& $signToolPath @signArguments")
    verify = source.index('& $signToolPath "verify" "/pa" "/tw" "/v"')
    native_postcondition = source.index(
        "$authenticodeSignature = Get-AuthenticodeSignature"
    )
    timestamp_postcondition = source.index(
        "$null -eq $authenticodeSignature.TimeStamperCertificate"
    )
    cleanup = source.index("Remove-Item -LiteralPath $temporaryCertificatePath -Force")
    assert (
        sign
        < invoke_sign
        < verify
        < native_postcondition
        < timestamp_postcondition
        < cleanup
    )
    sign_block = source[sign:invoke_sign]
    assert sign_block.index('"SHA256"') < sign_block.index('"/tr"')
    assert sign_block.index('"/tr"') < sign_block.index('"/td"')
    assert sign_block.count('"SHA256"') == 2
    assert '"/f"' in sign_block
    assert '"/p"' in sign_block
    assert "if ($LASTEXITCODE -ne 0)" in source
    assert "[System.Management.Automation.SignatureStatus]::Valid" in source
    assert "$null -eq $authenticodeSignature.SignerCertificate" in source
    assert source.index("finally {") < cleanup

    # Secrets are consumed from the Actions environment and never printed.
    for secret_name in (
        "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_BASE64",
        "OPENHCS_WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
    ):
        assert f'Write-Host "${secret_name}' not in source
        assert f"Write-Output ${secret_name}" not in source


def test_macos_release_uses_developer_id_notarytool_and_stapling() -> None:
    source = MACOS_SIGNING_PATH.read_text(encoding="utf-8")

    for secret_name in (
        "OPENHCS_MACOS_SIGNING_CERTIFICATE_BASE64",
        "OPENHCS_MACOS_SIGNING_CERTIFICATE_PASSWORD",
        "OPENHCS_MACOS_DEVELOPER_IDENTITY",
        "OPENHCS_MACOS_NOTARY_KEY_BASE64",
        "OPENHCS_MACOS_NOTARY_KEY_ID",
        "OPENHCS_MACOS_NOTARY_ISSUER_ID",
    ):
        assert f"require_environment {secret_name}" in source

    assert "/usr/bin/security create-keychain" in source
    assert "/usr/bin/security import" in source
    assert "/usr/bin/security set-key-partition-list" in source
    assert "/usr/bin/security delete-keychain" in source
    assert "/usr/bin/codesign --verify --deep --strict" in source
    assert '"Developer ID Application: "*)' in source
    assert "altool" not in source

    app = source[
        source.index("sign_app() {") : source.index("sign_dmg_and_notarize() {")
    ]
    assert app.index("create_signing_keychain") < app.index("/usr/bin/codesign")
    assert "--options runtime" in app
    assert "--timestamp" in app
    assert app.index("/usr/bin/codesign") < app.index(
        'verify_timestamped_signature "$app_path"'
    )
    assert "flags=.*runtime" in app

    dmg = source[
        source.index("sign_dmg_and_notarize() {") : source.index(
            "require_command /usr/bin/base64"
        )
    ]
    codesign = dmg.index("/usr/bin/codesign")
    verify = dmg.index('verify_timestamped_signature "$dmg_path"')
    submit = dmg.index("/usr/bin/xcrun notarytool submit")
    accepted = dmg.index('if [[ "$notary_status" != Accepted ]]')
    staple = dmg.index("/usr/bin/xcrun stapler staple")
    validate = dmg.index("/usr/bin/xcrun stapler validate")
    final_signature = dmg.rindex('verify_timestamped_signature "$dmg_path"')
    final_integrity = dmg.rindex('/usr/bin/hdiutil verify "$dmg_path"')
    gatekeeper = dmg.index("/usr/sbin/spctl")
    assert (
        codesign
        < verify
        < submit
        < accepted
        < staple
        < validate
        < final_signature
        < final_integrity
        < gatekeeper
    )
    assert dmg.count('/usr/bin/hdiutil verify "$dmg_path"') == 2
    assert "--wait" in dmg
    assert "--output-format json" in dmg


def test_tag_workflow_cannot_upload_unsigned_native_artifacts() -> None:
    workflow = PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")
    windows_job = workflow[
        workflow.index("  build-windows-installer:") : workflow.index(
            "  build-macos-installer:"
        )
    ]
    macos_job = workflow[
        workflow.index("  build-macos-installer:") : workflow.index(
            "  build-and-publish:"
        )
    ]

    windows_build = windows_job.index(
        "      - name: Build release-pinned single-file Windows installer"
    )
    windows_sign = windows_job.index("      - name: Sign and verify Windows installer")
    windows_upload = windows_job.index("Upload single-file Windows installer")
    assert windows_build < windows_sign < windows_upload
    assert "${{ secrets.OPENHCS_WINDOWS_SIGNING_CERTIFICATE_BASE64 }}" in (windows_job)
    assert "${{ secrets.OPENHCS_WINDOWS_SIGNING_CERTIFICATE_PASSWORD }}" in (
        windows_job
    )

    app_build = macos_job.index(
        "      - name: Build release-pinned macOS installer application"
    )
    app_sign = macos_job.index(
        "      - name: Sign and verify macOS installer application"
    )
    dmg_build = macos_job.index(
        "      - name: Build release-pinned macOS installer disk image"
    )
    staged_app_verify = macos_job.index("codesign \\", dmg_build)
    dmg_create = macos_job.index("hdiutil create", dmg_build)
    dmg_verify = macos_job.index("hdiutil verify", dmg_create)
    dmg_trust = macos_job.index(
        "      - name: Sign, notarize, and verify macOS installer disk image"
    )
    dmg_upload = macos_job.index("Upload macOS installer disk image")
    assert (
        app_build
        < app_sign
        < dmg_build
        < staged_app_verify
        < dmg_create
        < dmg_verify
        < dmg_trust
        < dmg_upload
    )

    app_build_step = macos_job[app_build:app_sign]
    app_sign_step = macos_job[app_sign:dmg_build]
    dmg_build_step = macos_job[dmg_build:dmg_trust]
    dmg_trust_step = macos_job[dmg_trust:dmg_upload]
    assert "${{ secrets." not in app_build_step
    assert "OPENHCS_MACOS_NOTARY_" not in app_sign_step
    assert "${{ secrets." not in dmg_build_step
    assert "OPENHCS_MACOS_NOTARY_KEY_BASE64" in dmg_trust_step

    for secret_name in (
        "OPENHCS_MACOS_SIGNING_CERTIFICATE_BASE64",
        "OPENHCS_MACOS_SIGNING_CERTIFICATE_PASSWORD",
        "OPENHCS_MACOS_DEVELOPER_IDENTITY",
        "OPENHCS_MACOS_NOTARY_KEY_BASE64",
        "OPENHCS_MACOS_NOTARY_KEY_ID",
        "OPENHCS_MACOS_NOTARY_ISSUER_ID",
    ):
        assert f"${{{{ secrets.{secret_name} }}}}" in macos_job

    publish_job = workflow[
        workflow.index("  build-and-publish:") : workflow.index(
            "  publish-mcp-registry:"
        )
    ]
    assert "needs: [build-windows-installer, build-macos-installer]" in publish_job


def test_local_and_pull_request_installer_builds_only_parse_trust_helpers() -> None:
    integration_workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")

    windows_parse = integration_workflow[
        integration_workflow.index(
            "      - name: Parse Windows PowerShell installer sources"
        ) : integration_workflow.index(
            "      - name: Execute and verify Windows installer"
        )
    ]
    macos_parse = integration_workflow[
        integration_workflow.index(
            "      - name: Validate macOS installer sources"
        ) : integration_workflow.index(
            "      - name: Execute and verify macOS installer"
        )
    ]
    assert '"packaging/installers/windows/Sign-Installer.ps1"' in windows_parse
    assert "packaging/installers/macos/Sign-Installer.sh" in macos_parse
    assert "OPENHCS_WINDOWS_SIGNING_CERTIFICATE" not in integration_workflow
    assert "OPENHCS_MACOS_SIGNING_CERTIFICATE" not in integration_workflow
    assert "sign-app" not in integration_workflow
    assert "sign-dmg-and-notarize" not in integration_workflow
