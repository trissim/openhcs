#!/usr/bin/env python3
"""Build the OpenHCS ChatGPT web-plugin package and review packet."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from openhcs.mcp.http import hosted_capability_registry
from openhcs.mcp.http_auth import McpHttpAuthenticationMode
from openhcs.resources.brand import BRAND_PRIMARY_COLOR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PLUGIN_ROOT = REPO_ROOT / "packaging" / "codex" / "openhcs"
LOCAL_PLUGIN_MANIFEST = LOCAL_PLUGIN_ROOT / ".codex-plugin" / "plugin.json"
CHATGPT_PLUGIN_ASSET = (
    REPO_ROOT / "openhcs" / "resources" / "assets" / "openhcs-icon-square.svg"
)

APP_MANIFEST_PATH = "./.app.json"
BRAND_ASSET_PATH = "./assets/openhcs.svg"
REGISTERED_APP_ID_PATTERN = re.compile(
    r"^(?:plugin_)?asdk_app_[A-Za-z0-9][A-Za-z0-9_-]*$"
)
LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})

PLUGIN_DESCRIPTION = (
    "Explore OpenHCS microscopy workflow architecture, processing functions, "
    "configuration schemas, and packaged guidance through its {access} hosted "
    "MCP service."
)
SHORT_DESCRIPTION = "Explore OpenHCS workflows"
LONG_DESCRIPTION = (
    "Use OpenHCS's {access} hosted service to understand "
    "pipeline architecture, discover microscopy processing functions, compare "
    "preprocessing approaches, and inspect configuration schemas. This web "
    "surface cannot access local files, run pipelines, or control desktop "
    "viewers."
)
STARTER_PROMPTS = (
    "Explain how an OpenHCS pipeline represents image-processing work.",
    "Find OpenHCS functions for normalizing an image stack.",
    "Show me how to configure an OpenHCS pipeline for my microscopy assay.",
)

POSITIVE_REVIEW_CASES = (
    {
        "id": "architecture-guidance",
        "prompt": "Explain the major parts of an OpenHCS pipeline to a new user.",
        "expected_behavior": (
            "Use packaged OpenHCS knowledge and architecture projections to "
            "explain the linear workflow and inherited configuration model."
        ),
        "expected_result_shape": (
            "A concise, structured explanation grounded in current OpenHCS "
            "declarations."
        ),
    },
    {
        "id": "function-discovery",
        "prompt": (
            "Find OpenHCS processing functions that can normalize a "
            "multi-plane fluorescence stack."
        ),
        "expected_behavior": (
            "Search declaration-owned function metadata and distinguish "
            "per-plane from whole-stack normalization."
        ),
        "expected_result_shape": (
            "Relevant functions with their declared semantics and validation "
            "considerations."
        ),
    },
    {
        "id": "preprocessing-guidance",
        "prompt": (
            "Compare background subtraction and illumination correction for "
            "uneven fluorescence images."
        ),
        "expected_behavior": (
            "Use packaged processing guidance to explain when each approach is "
            "appropriate and how to validate it."
        ),
        "expected_result_shape": (
            "A comparison with limitations, parameter considerations, and "
            "representative-image checks."
        ),
    },
    {
        "id": "configuration-schema",
        "prompt": (
            "Describe the OpenHCS pipeline configuration fields that control "
            "image grouping."
        ),
        "expected_behavior": (
            "Reflect the current configuration declarations rather than "
            "inventing or hand-copying a schema."
        ),
        "expected_result_shape": (
            "Field names, types, defaults or inheritance behavior, and concise "
            "descriptions."
        ),
    },
    {
        "id": "workflow-planning",
        "prompt": (
            "Help me outline an OpenHCS workflow for a two-channel nuclear and "
            "neurite assay."
        ),
        "expected_behavior": (
            "Combine function discovery and packaged domain guidance while "
            "asking for missing assay-specific information."
        ),
        "expected_result_shape": (
            "A bounded workflow outline with assumptions, candidate processing "
            "stages, and validation steps."
        ),
    },
)

NEGATIVE_REVIEW_CASES = (
    {
        "id": "local-file-access",
        "prompt": "Open the microscopy plate in C:\\Data\\experiment and inspect it.",
        "expected_behavior": (
            "Explain that the hosted read-only surface cannot access local "
            "files and direct the user to the local OpenHCS installation."
        ),
        "reason": "Browser-hosted OpenHCS has no client-machine filesystem access.",
    },
    {
        "id": "pipeline-execution",
        "prompt": "Run this OpenHCS pipeline on my images and save the outputs.",
        "expected_behavior": (
            "Do not claim to execute or write outputs; explain that execution "
            "belongs to the local OpenHCS MCP and desktop runtime."
        ),
        "reason": "The submitted hosted surface is intentionally read-only.",
    },
    {
        "id": "viewer-control",
        "prompt": "Show the latest result in my running Napari viewer.",
        "expected_behavior": (
            "Explain that the hosted service cannot discover or control local "
            "GUI/viewer processes."
        ),
        "reason": "Local UI bridges and viewers are excluded from hosted transport.",
    },
)


def _required_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _required_text(
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string.")
    return value


def validate_public_https_url(
    url: str,
    *,
    label: str,
    allow_query: bool = True,
) -> str:
    """Return a public HTTPS URL without embedded credentials or fragments."""
    value = _required_text(url, label=label)
    if any(character.isspace() or not character.isprintable() for character in value):
        raise ValueError(f"{label} must not contain whitespace or control characters.")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.hostname is None:
        raise ValueError(f"{label} must be an absolute HTTPS URL: {value}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not embed credentials: {value}")
    if parsed.fragment:
        raise ValueError(f"{label} must not include a fragment: {value}")
    if parsed.query and not allow_query:
        raise ValueError(f"{label} must not include a query: {value}")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} has an invalid port: {value}") from exc

    hostname = parsed.hostname.rstrip(".").lower()
    if (
        hostname in LOCAL_HOSTNAMES
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
    ):
        raise ValueError(f"{label} must use a public host: {value}")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError(f"{label} must use a public host: {value}")
    return value


def validate_remote_mcp_url(url: str) -> str:
    """Return a production-safe universal MCP endpoint URL."""
    return validate_public_https_url(
        url,
        label="Hosted MCP server URL",
        allow_query=False,
    )


def normalize_registered_app_id(app_id: str) -> str:
    """Return the canonical ID accepted by a plugin's ``.app.json``."""
    value = _required_text(app_id, label="Registered ChatGPT app ID")
    if REGISTERED_APP_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "Registered ChatGPT app ID must use asdk_app_... or "
            "plugin_asdk_app_... format."
        )
    if value.startswith("plugin_"):
        return value.removeprefix("plugin_")
    return value


def normalize_authentication_mode(
    authentication_mode: str | McpHttpAuthenticationMode,
) -> McpHttpAuthenticationMode:
    """Resolve the package mode through the hosted-server mode authority."""
    try:
        return McpHttpAuthenticationMode(authentication_mode)
    except ValueError as exc:
        valid_modes = ", ".join(mode.value for mode in McpHttpAuthenticationMode)
        raise ValueError(
            f"Hosted MCP authentication mode must be one of: {valid_modes}"
        ) from exc


def validate_plugin_relative_path(path: str, *, label: str) -> str:
    """Validate one manifest path relative to the plugin root."""
    value = _required_text(path, label=label)
    if not value.startswith("./") or "\\" in value:
        raise ValueError(f"{label} must start with './' and use '/' separators.")
    relative = PurePosixPath(value.removeprefix("./"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{label} must stay inside the plugin package.")
    return value


def _source_release_metadata(source_manifest: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read synchronized plugin manifest: {source_manifest}"
        ) from exc
    source = _required_mapping(decoded, label="Synchronized plugin manifest")
    author = _required_mapping(
        source.get("author"),
        label="Synchronized plugin manifest author",
    )
    _required_text(source.get("name"), label="Synchronized plugin name")
    _required_text(source.get("version"), label="Synchronized plugin version")
    developer_name = _required_text(
        author.get("name"),
        label="Synchronized plugin author name",
    )

    projected_keys = (
        "name",
        "version",
        "author",
        "repository",
        "license",
        "keywords",
    )
    release_metadata = {key: source[key] for key in projected_keys if key in source}
    release_metadata["developer_name"] = developer_name
    return release_metadata


def _plugin_manifest(
    *,
    release_metadata: Mapping[str, Any],
    authentication_mode: McpHttpAuthenticationMode,
    website_url: str,
    privacy_policy_url: str,
    terms_of_service_url: str,
) -> dict[str, Any]:
    developer_name = release_metadata["developer_name"]
    hosted_capability_registry()
    access_description = f"{authentication_mode.access_qualifier} read-only"
    manifest = {
        key: value for key, value in release_metadata.items() if key != "developer_name"
    }
    manifest.update(
        {
            "description": PLUGIN_DESCRIPTION.format(access=access_description),
            "homepage": website_url,
            "apps": validate_plugin_relative_path(
                APP_MANIFEST_PATH,
                label="Plugin app-manifest path",
            ),
            "interface": {
                "displayName": "OpenHCS",
                "shortDescription": SHORT_DESCRIPTION,
                "longDescription": LONG_DESCRIPTION.format(access=access_description),
                "developerName": developer_name,
                "category": "Developer Tools",
                "capabilities": ["Read"],
                "websiteURL": website_url,
                "privacyPolicyURL": privacy_policy_url,
                "termsOfServiceURL": terms_of_service_url,
                "defaultPrompt": list(STARTER_PROMPTS),
                "brandColor": BRAND_PRIMARY_COLOR,
                "composerIcon": validate_plugin_relative_path(
                    BRAND_ASSET_PATH,
                    label="Plugin composer-icon path",
                ),
                "logo": validate_plugin_relative_path(
                    BRAND_ASSET_PATH,
                    label="Plugin logo path",
                ),
            },
        }
    )
    return manifest


def _submission_packet(
    *,
    manifest: Mapping[str, Any],
    remote_mcp_url: str,
    registered_app_id: str,
    authentication_mode: McpHttpAuthenticationMode,
    website_url: str,
    support_url: str,
    privacy_policy_url: str,
    terms_of_service_url: str,
) -> dict[str, Any]:
    interface = _required_mapping(
        manifest.get("interface"),
        label="Generated plugin interface",
    )
    authentication = _submission_authentication(authentication_mode)
    return {
        "schema_version": "openhcs.chatgpt-plugin-submission.v1",
        "purpose": (
            "Deterministic reviewer-input packet. Public submission must use "
            "the OpenAI Platform With MCP flow and scan the production server "
            "directly; this file is not an uploaded runtime manifest."
        ),
        "submission_type": "with_mcp",
        "release": {
            "name": manifest["name"],
            "version": manifest["version"],
        },
        "local_package": {
            "registered_app_id": registered_app_id,
            "purpose": (
                "Local plugin testing only. Do not enter this identity in the "
                "public submission portal."
            ),
        },
        "mcp": {
            "url_type": "universal",
            "server_url": remote_mcp_url,
            "authentication": authentication,
            "access": "read_only",
            "tool_metadata_source": "production_server_scan",
        },
        "listing": {
            "display_name": interface["displayName"],
            "short_description": interface["shortDescription"],
            "long_description": interface["longDescription"],
            "developer_name": interface["developerName"],
            "website_url": website_url,
            "support_url": support_url,
            "privacy_policy_url": privacy_policy_url,
            "terms_of_service_url": terms_of_service_url,
            "starter_prompts": list(STARTER_PROMPTS),
        },
        "review_cases": {
            "positive": list(POSITIVE_REVIEW_CASES),
            "negative": list(NEGATIVE_REVIEW_CASES),
        },
        "intentional_exclusions": [
            "Client-machine files and directories",
            "Pipeline compilation or execution",
            "Writes, job submission, and output creation",
            "Local OpenHCS GUI, Napari, Fiji, and viewer control",
            "Arbitrary code execution",
        ],
        "release_notes": (
            f"OpenHCS {manifest['version']} read-only hosted MCP plugin for "
            "architecture guidance, function discovery, preprocessing "
            "guidance, and configuration-schema reflection."
        ),
    }


def _submission_authentication(
    authentication_mode: McpHttpAuthenticationMode,
) -> dict[str, object]:
    """Project the hosted auth owner into OpenAI reviewer metadata once."""
    return {
        "mode": authentication_mode.value,
        "type": "oauth" if authentication_mode.requires_oauth else "none",
        "user_login_required": authentication_mode.requires_oauth,
    }


def build_hosted_plugin(
    *,
    remote_mcp_url: str,
    registered_app_id: str,
    authentication_mode: str | McpHttpAuthenticationMode,
    website_url: str,
    support_url: str,
    privacy_policy_url: str,
    terms_of_service_url: str,
    output_dir: Path,
    source_manifest: Path = LOCAL_PLUGIN_MANIFEST,
    source_asset: Path = CHATGPT_PLUGIN_ASSET,
) -> tuple[Path, Path, Path, Path]:
    """Project release metadata into a registered web-plugin review bundle."""
    endpoint = validate_remote_mcp_url(remote_mcp_url)
    app_id = normalize_registered_app_id(registered_app_id)
    auth_mode = normalize_authentication_mode(authentication_mode)
    website = validate_public_https_url(website_url, label="Public website URL")
    support = validate_public_https_url(support_url, label="Public support URL")
    privacy = validate_public_https_url(
        privacy_policy_url,
        label="Public privacy-policy URL",
    )
    terms = validate_public_https_url(
        terms_of_service_url,
        label="Public terms-of-service URL",
    )
    release_metadata = _source_release_metadata(source_manifest)
    if not source_asset.is_file():
        raise ValueError(f"Plugin brand asset is unavailable: {source_asset}")

    destination = output_dir.resolve()
    if destination.name != release_metadata["name"]:
        raise ValueError(
            "Hosted plugin output folder must match the synchronized plugin name "
            f"{release_metadata['name']!r}: {destination}"
        )
    if destination.exists():
        raise ValueError(f"Hosted plugin output path already exists: {destination}")

    manifest = _plugin_manifest(
        release_metadata=release_metadata,
        authentication_mode=auth_mode,
        website_url=website,
        privacy_policy_url=privacy,
        terms_of_service_url=terms,
    )
    app_manifest = {"apps": {manifest["name"]: {"id": app_id}}}
    submission = _submission_packet(
        manifest=manifest,
        remote_mcp_url=endpoint,
        registered_app_id=app_id,
        authentication_mode=auth_mode,
        website_url=website,
        support_url=support,
        privacy_policy_url=privacy,
        terms_of_service_url=terms,
    )

    manifest_path = destination / ".codex-plugin" / "plugin.json"
    app_path = destination / ".app.json"
    submission_path = destination / "submission.json"
    asset_path = destination / BRAND_ASSET_PATH.removeprefix("./")
    manifest_path.parent.mkdir(parents=True)
    asset_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    app_path.write_text(
        json.dumps(app_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    submission_path.write_text(
        json.dumps(submission, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(source_asset, asset_path)
    return manifest_path, app_path, submission_path, asset_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Public production HTTPS MCP URL.")
    parser.add_argument(
        "--app-id",
        required=True,
        help="Registered ChatGPT app ID (plugin_asdk_app_... or asdk_app_...).",
    )
    parser.add_argument(
        "--authentication-mode",
        choices=[mode.value for mode in McpHttpAuthenticationMode],
        default=McpHttpAuthenticationMode.PUBLIC_READ_ONLY.value,
        help=(
            "Hosted server authentication contract. The initial public plugin "
            "uses public_read_only."
        ),
    )
    parser.add_argument("--website-url", required=True)
    parser.add_argument("--support-url", required=True)
    parser.add_argument("--privacy-policy-url", required=True)
    parser.add_argument("--terms-of-service-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in build_hosted_plugin(
        remote_mcp_url=args.url,
        registered_app_id=args.app_id,
        authentication_mode=args.authentication_mode,
        website_url=args.website_url,
        support_url=args.support_url,
        privacy_policy_url=args.privacy_policy_url,
        terms_of_service_url=args.terms_of_service_url,
        output_dir=args.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
