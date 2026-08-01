"""Tests for the generated OpenHCS ChatGPT web-plugin review bundle."""

import json

import pytest

from scripts import build_hosted_mcp_plugin as plugin_builder
from scripts.build_hosted_mcp_plugin import (
    build_hosted_plugin,
    normalize_authentication_mode,
    normalize_registered_app_id,
    validate_plugin_relative_path,
    validate_public_https_url,
    validate_remote_mcp_url,
)


PUBLIC_URLS = {
    "remote_mcp_url": "https://mcp.openhcs.example/mcp",
    "registered_app_id": "plugin_asdk_app_0123456789abcdef",
    "authentication_mode": "public_read_only",
    "website_url": "https://openhcs.example/",
    "support_url": "https://openhcs.example/support.html",
    "privacy_policy_url": "https://openhcs.example/privacy.html",
    "terms_of_service_url": "https://openhcs.example/terms.html",
}


def _source_manifest(tmp_path, *, name="openhcs"):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.2.3",
                "description": "local",
                "author": {
                    "name": "OpenHCSDev",
                    "email": "maintainer@example.com",
                    "url": "https://github.com/OpenHCSDev",
                },
                "homepage": "https://local-only.example/",
                "repository": "https://github.com/OpenHCSDev/OpenHCS",
                "license": "MIT",
                "keywords": ["microscopy", "mcp"],
                "skills": "./skills/",
                "mcpServers": "./.mcp.json",
                "interface": {
                    "shortDescription": "local",
                    "longDescription": "local",
                    "capabilities": ["Read", "Write", "Interactive"],
                    "defaultPrompt": ["local"],
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def _source_asset(tmp_path):
    source = tmp_path / "openhcs.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64"/></svg>\n',
        encoding="utf-8",
    )
    return source


def test_default_hosted_plugin_asset_uses_package_brand_authority():
    assert plugin_builder.CHATGPT_PLUGIN_ASSET == (
        plugin_builder.REPO_ROOT
        / "openhcs"
        / "resources"
        / "assets"
        / "openhcs-icon-square.svg"
    )


def _build(tmp_path, *, output_name="plugin", **overrides):
    arguments = {
        **PUBLIC_URLS,
        "output_dir": tmp_path / output_name / "openhcs",
        "source_manifest": _source_manifest(tmp_path),
        "source_asset": _source_asset(tmp_path),
        **overrides,
    }
    return build_hosted_plugin(**arguments)


def test_hosted_plugin_is_registered_read_only_projection(tmp_path):
    manifest_path, app_path, submission_path, asset_path = _build(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    app = json.loads(app_path.read_text(encoding="utf-8"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))

    assert manifest["version"] == "1.2.3"
    assert manifest["author"]["name"] == "OpenHCSDev"
    assert manifest["repository"] == "https://github.com/OpenHCSDev/OpenHCS"
    assert manifest["homepage"] == "https://openhcs.example/"
    assert "skills" not in manifest
    assert "mcpServers" not in manifest
    assert manifest["apps"] == "./.app.json"
    assert manifest["interface"] == {
        "displayName": "OpenHCS",
        "shortDescription": "Explore OpenHCS workflows",
        "longDescription": (
            "Use OpenHCS's public read-only hosted service to understand "
            "pipeline architecture, discover microscopy processing functions, "
            "compare preprocessing approaches, and inspect configuration "
            "schemas. This web surface cannot access local files, run "
            "pipelines, or control desktop viewers."
        ),
        "developerName": "OpenHCSDev",
        "category": "Developer Tools",
        "capabilities": ["Read"],
        "websiteURL": "https://openhcs.example/",
        "privacyPolicyURL": "https://openhcs.example/privacy.html",
        "termsOfServiceURL": "https://openhcs.example/terms.html",
        "defaultPrompt": [
            "Explain how an OpenHCS pipeline represents image-processing work.",
            "Find OpenHCS functions for normalizing an image stack.",
            "Show me how to configure an OpenHCS pipeline for my microscopy assay.",
        ],
        "brandColor": "#00AAFF",
        "composerIcon": "./assets/openhcs.svg",
        "logo": "./assets/openhcs.svg",
    }
    assert app == {"apps": {"openhcs": {"id": "asdk_app_0123456789abcdef"}}}
    assert asset_path.read_text(encoding="utf-8").startswith("<svg ")

    assert submission["submission_type"] == "with_mcp"
    assert submission["local_package"] == {
        "registered_app_id": "asdk_app_0123456789abcdef",
        "purpose": (
            "Local plugin testing only. Do not enter this identity in the "
            "public submission portal."
        ),
    }
    assert submission["mcp"] == {
        "url_type": "universal",
        "server_url": "https://mcp.openhcs.example/mcp",
        "authentication": {
            "mode": "public_read_only",
            "type": "none",
            "user_login_required": False,
        },
        "access": "read_only",
        "tool_metadata_source": "production_server_scan",
    }
    assert submission["listing"]["support_url"] == (
        "https://openhcs.example/support.html"
    )
    assert len(submission["review_cases"]["positive"]) == 5
    assert len(submission["review_cases"]["negative"]) == 3
    assert "tools" not in submission
    assert "tool_names" not in submission


def test_hosted_plugin_build_is_deterministic(tmp_path):
    first_paths = _build(tmp_path, output_name="first")
    second_paths = _build(tmp_path, output_name="second")

    first = {
        path.relative_to(tmp_path / "first" / "openhcs"): path.read_bytes()
        for path in first_paths
    }
    second = {
        path.relative_to(tmp_path / "second" / "openhcs"): path.read_bytes()
        for path in second_paths
    }
    assert first == second


def test_private_hosted_plugin_truthfully_projects_oauth_mode(tmp_path):
    manifest_path, _, submission_path, _ = _build(
        tmp_path,
        authentication_mode="oauth_introspection",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert "authenticated, subject-isolated read-only" in manifest["description"]
    assert submission["mcp"]["authentication"] == {
        "mode": "oauth_introspection",
        "type": "oauth",
        "user_login_required": True,
    }


def test_plugin_build_requires_the_validated_hosted_registry(monkeypatch, tmp_path):
    def reject_hosted_surface():
        raise ValueError("hosted policy rejected")

    monkeypatch.setattr(
        plugin_builder,
        "hosted_capability_registry",
        reject_hosted_surface,
    )

    with pytest.raises(ValueError, match="hosted policy rejected"):
        _build(tmp_path)


def test_authentication_mode_uses_hosted_server_authority():
    assert normalize_authentication_mode("public_read_only").value == (
        "public_read_only"
    )
    assert normalize_authentication_mode("oauth_introspection").value == (
        "oauth_introspection"
    )
    with pytest.raises(ValueError, match="authentication mode"):
        normalize_authentication_mode("oauth")


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp.openhcs.example/mcp",
        "https://localhost/mcp",
        "https://service.local/mcp",
        "https://127.0.0.1/mcp",
        "https://[::1]/mcp",
        "https://user:secret@mcp.openhcs.example/mcp",
        "https://mcp.openhcs.example/mcp#fragment",
        "https://mcp.openhcs.example/mcp?token=secret",
        "https://mcp.openhcs.example/mcp\nheader",
    ],
)
def test_hosted_plugin_rejects_nonproduction_mcp_urls(url):
    with pytest.raises(ValueError):
        validate_remote_mcp_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://openhcs.example/privacy.html",
        "https://localhost/privacy.html",
        "https://user:secret@openhcs.example/privacy.html",
        "https://openhcs.example/privacy.html#section",
    ],
)
def test_public_listing_urls_are_https_public_and_unfragmented(url):
    with pytest.raises(ValueError):
        validate_public_https_url(url, label="Listing URL")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("asdk_app_abc123", "asdk_app_abc123"),
        ("plugin_asdk_app_abc123", "asdk_app_abc123"),
    ],
)
def test_registered_app_id_is_normalized_for_app_manifest(value, expected):
    assert normalize_registered_app_id(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "connector_abc123",
        "plugin_asdk_app_",
        "plugin_asdk_app_bad/value",
        " plugin_asdk_app_abc123",
    ],
)
def test_registered_app_id_rejects_missing_or_unrelated_identity(value):
    with pytest.raises(ValueError, match="app ID"):
        normalize_registered_app_id(value)


@pytest.mark.parametrize(
    "path",
    [
        ".app.json",
        "../.app.json",
        "./../.app.json",
        "/assets/icon.svg",
        "./assets\\icon.svg",
    ],
)
def test_plugin_manifest_paths_cannot_escape_package(path):
    with pytest.raises(ValueError):
        validate_plugin_relative_path(path, label="Manifest path")


def test_hosted_plugin_does_not_overwrite_existing_output(tmp_path):
    destination = tmp_path / "plugin" / "openhcs"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        _build(tmp_path)

    assert marker.read_text(encoding="utf-8") == "user data"


def test_hosted_plugin_rejects_existing_empty_output(tmp_path):
    destination = tmp_path / "plugin" / "openhcs"
    destination.mkdir(parents=True)

    with pytest.raises(ValueError, match="already exists"):
        _build(tmp_path)

    assert list(destination.iterdir()) == []


def test_invalid_external_input_leaves_no_partial_output(tmp_path):
    destination = tmp_path / "plugin" / "openhcs"

    with pytest.raises(ValueError, match="privacy-policy"):
        _build(
            tmp_path,
            privacy_policy_url="http://openhcs.example/privacy.html",
        )

    assert not destination.exists()


def test_missing_synchronized_metadata_leaves_no_partial_output(tmp_path):
    source_manifest = tmp_path / "incomplete.json"
    source_manifest.write_text('{"name": "openhcs"}\n', encoding="utf-8")
    destination = tmp_path / "plugin" / "openhcs"

    with pytest.raises(ValueError, match="author"):
        _build(
            tmp_path,
            source_manifest=source_manifest,
        )

    assert not destination.exists()


def test_registered_app_alias_projects_synchronized_plugin_identity(tmp_path):
    source_manifest = _source_manifest(tmp_path, name="openhcs-web")
    manifest_path, app_path, _, _ = build_hosted_plugin(
        **PUBLIC_URLS,
        output_dir=tmp_path / "openhcs-web",
        source_manifest=source_manifest,
        source_asset=_source_asset(tmp_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    app = json.loads(app_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "openhcs-web"
    assert app == {
        "apps": {
            "openhcs-web": {
                "id": "asdk_app_0123456789abcdef",
            }
        }
    }


def test_hosted_plugin_rejects_output_folder_that_differs_from_manifest_name(
    tmp_path,
):
    with pytest.raises(ValueError, match="output folder"):
        _build(
            tmp_path,
            output_dir=tmp_path / "wrong-name",
        )

    assert not (tmp_path / "wrong-name").exists()
