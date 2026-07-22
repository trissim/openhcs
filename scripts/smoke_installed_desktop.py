#!/usr/bin/env python3
"""Verify one native OpenHCS installer result outside the source checkout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import plistlib
import subprocess
from typing import Any

from packaging.requirements import Requirement

from scripts.render_installer_contract import SCHEMA_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INSTALLED_MCP_SMOKE_PATH = Path(__file__).with_name("smoke_installed_mcp.py")


@dataclass(frozen=True, slots=True)
class InstallerSmokeContract:
    """Validated values needed to inspect an installed desktop application."""

    product_name: str
    package_requirement: Requirement
    entry_point: str

    @classmethod
    def load(cls, path: Path) -> "InstallerSmokeContract":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError(
                "Smoke test received an unsupported installer contract"
            )
        return cls(
            product_name=_required_text(payload, "product_name"),
            package_requirement=Requirement(
                _required_text(payload, "package_requirement")
            ),
            entry_point=_required_text(payload, "entry_point"),
        )


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(f"Installer contract field {name!r} is not text")
    return value


def _only_environment(install_root: Path) -> Path:
    environments_root = install_root / "environments"
    environments = sorted(path for path in environments_root.iterdir() if path.is_dir())
    if len(environments) != 1:
        raise AssertionError(
            f"Expected one installed environment under {environments_root}, "
            f"found {len(environments)}"
        )
    return environments[0].resolve()


def _environment_paths(
    environment: Path,
    entry_point: str,
    platform_name: str,
) -> tuple[Path, Path]:
    if platform_name == "windows":
        return (
            environment / "Scripts" / "python.exe",
            environment / "Scripts" / f"{entry_point}.exe",
        )
    return environment / "bin" / "python", environment / "bin" / entry_point


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    timeout_seconds: float = 120,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Command failed with exit code {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _installed_distribution_probe(
    python_executable: Path,
    contract: InstallerSmokeContract,
    environment: Path,
) -> dict[str, Any]:
    probe = r"""
import json
from importlib.metadata import PackageNotFoundError, distribution
import sys

from packaging.requirements import Requirement
from packaging.version import Version

package_name, entry_name, *selected_extras = sys.argv[1:]
installed = distribution(package_name)
provided_extras = set(installed.metadata.get_all("Provides-Extra") or ())
missing_extras = sorted(set(selected_extras) - provided_extras)
if missing_extras:
    raise SystemExit(
        "installed distribution does not declare selected extras: "
        + ", ".join(missing_extras)
    )

resolved_requirements = {}
for requirement_text in installed.requires or ():
    requirement = Requirement(requirement_text)
    if requirement.marker is not None and not any(
        requirement.marker.evaluate({"extra": extra})
        for extra in selected_extras
    ):
        continue
    try:
        dependency = distribution(requirement.name)
    except PackageNotFoundError as exc:
        raise SystemExit(
            f"selected installer dependency is missing: {requirement}"
        ) from exc
    if requirement.specifier and Version(dependency.version) not in requirement.specifier:
        raise SystemExit(
            f"installed {requirement.name} {dependency.version} does not satisfy "
            f"{requirement.specifier}"
        )
    resolved_requirements[requirement.name] = dependency.version

entry_points = [
    {"name": item.name, "value": item.value}
    for item in installed.entry_points
    if item.group == "console_scripts" and item.name == entry_name
]
print(json.dumps({
    "version": installed.version,
    "location": str(installed.locate_file("")),
    "entry_points": entry_points,
    "selected_extras": sorted(selected_extras),
    "resolved_requirements": resolved_requirements,
}))
"""
    completed = _run_checked(
        [
            str(python_executable),
            "-I",
            "-c",
            probe,
            contract.package_requirement.name,
            contract.entry_point,
            *sorted(contract.package_requirement.extras),
        ],
        cwd=environment,
    )
    payload = json.loads(completed.stdout)
    version = payload.get("version")
    if (
        not isinstance(version, str)
        or version not in contract.package_requirement.specifier
    ):
        raise AssertionError(
            f"Installed version {version!r} does not satisfy "
            f"{contract.package_requirement}"
        )
    entry_points = payload.get("entry_points")
    if not isinstance(entry_points, list) or len(entry_points) != 1:
        raise AssertionError(
            f"Installed distribution does not expose exactly one "
            f"{contract.entry_point!r} console entry point"
        )
    distribution_root = Path(payload["location"]).resolve()
    if not distribution_root.is_relative_to(environment):
        raise AssertionError(
            f"Installed distribution escaped its environment: {distribution_root}"
        )
    return payload


def _smoke_entry_point(entry_executable: Path, install_root: Path) -> None:
    environment = os.environ.copy()
    environment["OPENHCS_CPU_ONLY"] = "true"
    completed = _run_checked(
        [str(entry_executable), "--help"],
        cwd=install_root,
        environment=environment,
    )
    if "High-Content Screening Platform" not in completed.stdout:
        raise AssertionError("Installed OpenHCS entry point did not render GUI help")


def _smoke_installed_mcp(
    python_executable: Path,
    install_root: Path,
) -> dict[str, Any]:
    """Exercise the installed MCP server over stdio from outside the checkout."""

    environment = os.environ.copy()
    environment["OPENHCS_CPU_ONLY"] = "true"
    environment["XDG_CACHE_HOME"] = str((install_root / "mcp-cache").resolve())
    completed = _run_checked(
        [
            str(python_executable),
            "-I",
            str(INSTALLED_MCP_SMOKE_PATH.resolve()),
            "--forbid-import-root",
            str(REPOSITORY_ROOT),
        ],
        cwd=install_root,
        environment=environment,
    )
    payload = json.loads(completed.stdout)
    if payload.get("health_status") != "ok":
        raise AssertionError(f"Installed MCP smoke did not report health: {payload}")
    return payload


def _smoke_installed_demo(
    python_executable: Path,
    install_root: Path,
) -> dict[str, Any]:
    """Execute the installed MCP/runtime/Napari demo outside the checkout."""

    demo_root = (install_root / "installer-smoke-demo").resolve()
    environment = os.environ.copy()
    environment["OPENHCS_CPU_ONLY"] = "true"
    environment["OPENHCS_AGENT_READ_ROOTS"] = str(demo_root)
    environment["OPENHCS_AGENT_WRITE_ROOTS"] = str(demo_root)
    environment["XDG_CACHE_HOME"] = str((install_root / "mcp-cache").resolve())
    completed = _run_checked(
        [
            str(python_executable),
            "-I",
            "-m",
            "openhcs.mcp.installed_demo",
            "--output-root",
            str(demo_root),
            "--forbid-import-root",
            str(REPOSITORY_ROOT),
            "--json",
        ],
        cwd=install_root,
        environment=environment,
        timeout_seconds=300,
    )
    payload = json.loads(completed.stdout)
    required_values = {
        "execution_status": "complete",
        "viewer_observed": True,
        "viewer_type": "napari",
    }
    mismatches = {
        name: {"expected": expected, "actual": payload.get(name)}
        for name, expected in required_values.items()
        if payload.get(name) != expected
    }
    if mismatches:
        raise AssertionError(
            f"Installed MCP/Napari demo did not satisfy its contract: {mismatches}"
        )
    if payload.get("viewer_layer_count", 0) < 1:
        raise AssertionError(f"Installed Napari demo mounted no layers: {payload}")
    if payload.get("viewer_nonzero_payload_count", 0) < 1:
        raise AssertionError(
            f"Installed Napari demo exposed no nonzero payloads: {payload}"
        )
    return payload


def _verify_windows_launcher(
    contract: InstallerSmokeContract,
    install_root: Path,
    environment: Path,
    entry_executable: Path,
    desktop_root: Path,
) -> dict[str, str]:
    launcher_name = f"Launch-{contract.product_name.replace(' ', '-')}.ps1"
    launcher_path = install_root / launcher_name
    shortcut_path = desktop_root / f"{contract.product_name}.lnk"
    if not launcher_path.is_file():
        raise AssertionError(f"Windows launch adapter is missing: {launcher_path}")
    if not shortcut_path.is_file():
        raise AssertionError(f"Windows desktop shortcut is missing: {shortcut_path}")

    launcher_source = launcher_path.read_text(encoding="utf-8-sig")
    relative_entry = entry_executable.relative_to(install_root)
    if str(relative_entry).replace("/", "\\") not in launcher_source:
        raise AssertionError(
            "Windows launch adapter does not target the installed entry"
        )
    if environment.name not in launcher_source:
        raise AssertionError(
            "Windows launch adapter does not retain environment identity"
        )
    return {
        "launcher_path": str(launcher_path.resolve()),
        "shortcut_path": str(shortcut_path.resolve()),
    }


def _verify_macos_launcher(
    contract: InstallerSmokeContract,
    install_root: Path,
    environment: Path,
    home_root: Path,
) -> dict[str, str]:
    current_environment = install_root / "current"
    if not current_environment.is_symlink():
        raise AssertionError(
            "macOS installer did not publish the current environment link"
        )
    if current_environment.resolve() != environment:
        raise AssertionError(
            "macOS current environment link targets the wrong environment"
        )

    launcher_app = home_root / "Applications" / f"{contract.product_name}.app"
    desktop_link = home_root / "Desktop" / f"{contract.product_name}.app"
    if not launcher_app.is_dir():
        raise AssertionError(f"macOS launcher app is missing: {launcher_app}")
    if (
        not desktop_link.is_symlink()
        or desktop_link.resolve() != launcher_app.resolve()
    ):
        raise AssertionError("macOS desktop launcher does not target the installed app")

    plist_path = launcher_app / "Contents" / "Info.plist"
    with plist_path.open("rb") as stream:
        info = plistlib.load(stream)
    executable_name = info.get("CFBundleExecutable")
    if not isinstance(executable_name, str) or not executable_name:
        raise AssertionError("macOS launcher app has no declared executable")
    launcher_executable = launcher_app / "Contents" / "MacOS" / executable_name
    if not os.access(launcher_executable, os.X_OK):
        raise AssertionError("macOS launcher executable is not executable")

    launcher_environment = os.environ.copy()
    launcher_environment["HOME"] = str(home_root)
    launcher_environment["OPENHCS_CPU_ONLY"] = "true"
    completed = _run_checked(
        [str(launcher_executable), "--help"],
        cwd=home_root,
        environment=launcher_environment,
    )
    if "High-Content Screening Platform" not in completed.stdout:
        raise AssertionError(
            "macOS launcher did not reach the installed OpenHCS GUI entry"
        )
    return {
        "launcher_path": str(launcher_app.resolve()),
        "shortcut_path": str(desktop_link),
    }


def smoke_installed_desktop(
    *,
    contract_path: Path,
    install_root: Path,
    platform_name: str,
    home_root: Path | None,
    desktop_root: Path | None,
) -> dict[str, Any]:
    contract = InstallerSmokeContract.load(contract_path)
    install_root = install_root.resolve()
    environment = _only_environment(install_root)
    python_executable, entry_executable = _environment_paths(
        environment,
        contract.entry_point,
        platform_name,
    )
    if not python_executable.is_file() or not entry_executable.is_file():
        raise AssertionError(
            "Installer did not publish Python and the declared entry point"
        )

    distribution = _installed_distribution_probe(
        python_executable,
        contract,
        environment,
    )
    _smoke_entry_point(entry_executable, install_root)
    mcp = _smoke_installed_mcp(python_executable, install_root)

    if platform_name == "windows":
        if desktop_root is None:
            raise AssertionError("Windows smoke requires the native Desktop folder")
        launcher = _verify_windows_launcher(
            contract,
            install_root,
            environment,
            entry_executable,
            desktop_root.resolve(),
        )
    else:
        if home_root is None:
            raise AssertionError("macOS smoke requires the isolated user home")
        launcher = _verify_macos_launcher(
            contract,
            install_root,
            environment,
            home_root.resolve(),
        )

    demo = _smoke_installed_demo(python_executable, install_root)

    return {
        "platform": platform_name,
        "version": distribution["version"],
        "environment": str(environment),
        "mcp": mcp,
        "demo": demo,
        **launcher,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows", "macos"), required=True)
    parser.add_argument("--home-root", type=Path)
    parser.add_argument("--desktop-root", type=Path)
    args = parser.parse_args()
    result = smoke_installed_desktop(
        contract_path=args.contract,
        install_root=args.install_root,
        platform_name=args.platform,
        home_root=args.home_root,
        desktop_root=args.desktop_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
