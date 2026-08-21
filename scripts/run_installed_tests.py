#!/usr/bin/env python3
"""Run pytest while proving that OpenHCS comes from the installed wheel."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _remove_checkout_import_paths(repo_root: Path) -> None:
    installed_roots = tuple(
        dict.fromkeys((Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()))
    )

    def _is_checkout_source_path(entry: str) -> bool:
        path = Path(entry)
        if any(_is_within(path, root) for root in installed_roots):
            return False
        return _is_within(path, repo_root)

    sys.path[:] = [
        entry
        for entry in sys.path
        if entry and not _is_checkout_source_path(entry)
    ]

    inherited_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    safe_entries = [
        entry
        for entry in inherited_entries
        if entry and not _is_checkout_source_path(entry)
    ]
    if safe_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join(safe_entries)
    else:
        os.environ.pop("PYTHONPATH", None)


def _resolved_package_path(package_name: str) -> Path:
    """Resolve a package without importing it before pytest-cov starts."""

    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Installed package {package_name!r} is not importable")
    return Path(spec.origin).resolve()


def _prepare_installed_test_runtime(repo_root: Path) -> Path:
    """Expose checkout tests after the installed OpenHCS import location."""

    _remove_checkout_import_paths(repo_root)
    sys.path.append(str(repo_root))
    return _resolved_package_path("openhcs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Collect coverage from the resolved installed OpenHCS package.",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("pytest_arguments", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()

    # Fixtures and pytest configuration remain checkout-owned, but the checkout
    # comes after installed site-packages so the tested OpenHCS import stays on
    # the candidate wheel. Resolving the spec does not execute OpenHCS before
    # pytest-cov starts tracing at session start.
    package_path = _prepare_installed_test_runtime(repo_root)
    source_root = repo_root / "openhcs"
    if package_path.is_relative_to(source_root):
        raise RuntimeError(
            "CI imported OpenHCS from the source checkout instead of its wheel: "
            f"{package_path}"
        )

    import pytest

    # Child Python processes must retain the same package boundary even when
    # their working directory is the source checkout.
    os.environ["PYTHONSAFEPATH"] = "1"
    pytest_arguments = args.pytest_arguments
    if pytest_arguments[:1] == ["--"]:
        pytest_arguments = pytest_arguments[1:]
    if args.coverage:
        pytest_arguments = [f"--cov={package_path.parent}", *pytest_arguments]
    return pytest.main(["--import-mode=importlib", *pytest_arguments])


if __name__ == "__main__":
    raise SystemExit(main())
