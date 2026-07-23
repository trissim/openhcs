"""Source-checkout dependency bootstrap for vendored OpenHCS externals."""

from __future__ import annotations

import configparser
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTERNAL_ROOT = _REPO_ROOT / "external"


def ensure_source_checkout_external_paths(
    repo_root: Path = _REPO_ROOT,
) -> tuple[Path, ...]:
    """Prefer local external checkouts when OpenHCS runs from source."""
    external_root = repo_root / "external"
    if not external_root.exists():
        return ()

    paths = tuple(_discover_external_import_roots(external_root))
    _reject_stale_loaded_externals(paths)
    for path in reversed(paths):
        _prepend_sys_path(path)
    return paths


def _discover_external_import_roots(external_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for repo_dir in sorted(external_root.iterdir()):
        if not repo_dir.is_dir():
            continue
        paths.extend(_discover_repo_import_roots(repo_dir))
    return tuple(_dedupe_existing_paths(paths))


def _discover_repo_import_roots(repo_dir: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    candidates.extend(_pyproject_import_roots(repo_dir))
    candidates.extend(_setup_cfg_import_roots(repo_dir))
    if not candidates:
        candidates.extend(_heuristic_import_roots(repo_dir))
    return tuple(_dedupe_existing_paths(candidates))


def _pyproject_import_roots(repo_dir: Path) -> tuple[Path, ...]:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.is_file():
        return ()

    import tomllib

    data = tomllib.loads(pyproject.read_text())
    setuptools_config = data.get("tool", {}).get("setuptools", {})
    candidates: list[Path] = []

    find_config = setuptools_config.get("packages", {}).get("find", {})
    where = find_config.get("where")
    if isinstance(where, list):
        candidates.extend(repo_dir / item for item in where)

    package_dir = setuptools_config.get("package-dir")
    if isinstance(package_dir, dict):
        base = package_dir.get("") or package_dir.get("root")
        if base:
            candidates.append(repo_dir / base)

    return tuple(candidates)


def _setup_cfg_import_roots(repo_dir: Path) -> tuple[Path, ...]:
    setup_cfg = repo_dir / "setup.cfg"
    if not setup_cfg.is_file():
        return ()

    parser = configparser.ConfigParser()
    parser.read(setup_cfg)
    candidates: list[Path] = []

    if parser.has_section("options.packages.find") and parser.has_option(
        "options.packages.find",
        "where",
    ):
        where = parser.get("options.packages.find", "where")
        candidates.extend(
            repo_dir / item.strip()
            for item in where.split(",")
            if item.strip()
        )

    if parser.has_section("options") and parser.has_option(
        "options",
        "package_dir",
    ):
        package_dir = parser.get("options", "package_dir").strip()
        if package_dir.startswith("="):
            base = package_dir.split("=", 1)[1].strip()
            if base:
                candidates.append(repo_dir / base)

    return tuple(candidates)


def _heuristic_import_roots(repo_dir: Path) -> tuple[Path, ...]:
    src_dir = repo_dir / "src"
    if _has_package_dir(src_dir):
        return (src_dir,)
    if _has_package_dir(repo_dir):
        return (repo_dir,)
    return ()


def _has_package_dir(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(
        child.is_dir() and (child / "__init__.py").is_file()
        for child in root.iterdir()
    )


def _dedupe_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _reject_stale_loaded_externals(import_roots: tuple[Path, ...]) -> None:
    roots_by_package = _external_roots_by_package(import_roots)
    for package_name, roots in roots_by_package.items():
        module = sys.modules.get(package_name)
        if module is None:
            continue
        origin = _module_origin(module)
        if origin is None:
            continue
        if any(_is_relative_to(origin, root) for root in roots):
            continue
        raise RuntimeError(
            f"External package {package_name!r} was imported from {origin} "
            "before OpenHCS source-checkout externals were activated. "
            "Start the process with the OpenHCS checkout on sys.path first, "
            f"or import OpenHCS before {package_name!r}. Expected one of: "
            f"{', '.join(str(root) for root in roots)}"
        )


def _external_roots_by_package(
    import_roots: tuple[Path, ...],
) -> dict[str, tuple[Path, ...]]:
    mutable: dict[str, list[Path]] = defaultdict(list)
    for root in import_roots:
        for package_name in _top_level_packages(root):
            mutable[package_name].append(root)
    return {name: tuple(paths) for name, paths in mutable.items()}


def _top_level_packages(import_root: Path) -> tuple[str, ...]:
    return tuple(
        child.name
        for child in import_root.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    )


def _module_origin(module: ModuleType) -> Path | None:
    origin = getattr(module, "__file__", None)
    if origin is None:
        return None
    return Path(origin).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

