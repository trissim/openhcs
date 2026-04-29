"""
OpenHCS: A library for stitching microscopy images.

This module provides the public API for OpenHCS.
It re-exports only the intended public symbols from openhcs.ez.api
and does NOT import from internal modules in a way that triggers
registrations or other side-effects.
"""

import logging
import os
import sys
import platform
from pathlib import Path

__version__ = "0.5.15"

# Configure polystore defaults for OpenHCS integration
os.environ.setdefault("POLYSTORE_METADATA_FILENAME", "openhcs_metadata.json")
os.environ.setdefault("POLYSTORE_ZMQ_APP_NAME", "openhcs")
os.environ.setdefault("POLYSTORE_ZMQ_IPC_PREFIX", "openhcs-zmq")
os.environ.setdefault("POLYSTORE_ZMQ_IPC_DIR", "ipc")
os.environ.setdefault("POLYSTORE_ZMQ_IPC_EXT", ".sock")
os.environ.setdefault("POLYSTORE_ZMQ_CONTROL_OFFSET", "1000")
os.environ.setdefault("POLYSTORE_ZMQ_DEFAULT_PORT", "7777")
os.environ.setdefault("POLYSTORE_ZMQ_ACK_PORT", "7555")
if os.getenv("OPENHCS_SUBPROCESS_NO_GPU") == "1":
    os.environ.setdefault("POLYSTORE_SUBPROCESS_NO_GPU", "1")

# Prefer local external package checkouts when running from source
_repo_root = Path(__file__).resolve().parent.parent
_external_root = _repo_root / "external"


def _has_package_dir(root: Path) -> bool:
    """Return True if root contains at least one Python package directory."""
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if child.is_dir() and (child / "__init__.py").is_file():
            return True
    return False


def _discover_external_paths(repo_dir: Path) -> list[Path]:
    """Discover import roots for an external repo without hardcoding layout."""
    candidates: list[Path] = []

    # 1) Try pyproject.toml (setuptools where/package-dir hints)
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib

            data = tomllib.loads(pyproject.read_text())
            find_cfg = (
                data.get("tool", {})
                .get("setuptools", {})
                .get("packages", {})
                .get("find", {})
            )
            where = find_cfg.get("where")
            if isinstance(where, list):
                candidates.extend(repo_dir / w for w in where)

            package_dir = data.get("tool", {}).get("setuptools", {}).get("package-dir")
            if isinstance(package_dir, dict):
                base = package_dir.get("") or package_dir.get("root")
                if base:
                    candidates.append(repo_dir / base)
        except Exception:
            pass

    # 2) Try setup.cfg (setuptools where/package_dir)
    setup_cfg = repo_dir / "setup.cfg"
    if setup_cfg.is_file():
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read(setup_cfg)
            if cfg.has_section("options.packages.find") and cfg.has_option("options.packages.find", "where"):
                where = cfg.get("options.packages.find", "where")
                for w in [p.strip() for p in where.split(",") if p.strip()]:
                    candidates.append(repo_dir / w)
            if cfg.has_section("options") and cfg.has_option("options", "package_dir"):
                pkg_dir = cfg.get("options", "package_dir").strip()
                if pkg_dir.startswith("="):
                    base = pkg_dir.split("=", 1)[1].strip()
                    if base:
                        candidates.append(repo_dir / base)
        except Exception:
            pass

    # 3) Heuristics (src/ or repo root packages)
    if not candidates:
        src_dir = repo_dir / "src"
        if _has_package_dir(src_dir):
            candidates.append(src_dir)
        elif _has_package_dir(repo_dir):
            candidates.append(repo_dir)

    # Filter and de-dupe
    seen = set()
    result: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if path.is_dir() and resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


if _external_root.exists():
    for _repo in sorted(_external_root.iterdir()):
        if not _repo.is_dir():
            continue
        for _path in _discover_external_paths(_repo):
            _path_str = str(_path)
            if _path_str not in sys.path:
                sys.path.insert(0, _path_str)

# Force UTF-8 encoding for stdout/stderr on Windows
# This ensures emoji and Unicode characters work in console output
if platform.system() == 'Windows':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Monkey patch logging.FileHandler to default to UTF-8 encoding
# This ensures all log files support emojis and Unicode characters
_original_file_handler_init = logging.FileHandler.__init__

def _utf8_file_handler_init(self, filename, mode='a', encoding='utf-8', delay=False, errors=None):
    """FileHandler.__init__ with UTF-8 encoding as default."""
    return _original_file_handler_init(self, filename, mode, encoding, delay, errors)

logging.FileHandler.__init__ = _utf8_file_handler_init

# Set up basic logging configuration if none exists
# This ensures INFO level logging works when testing outside the TUI
def _ensure_basic_logging():
    """Ensure basic logging is configured if no configuration exists."""
    root_logger = logging.getLogger()

    # Only configure if no handlers exist and level is too high
    if not root_logger.handlers and root_logger.level > logging.INFO:
        # Set up basic console logging at INFO level
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

# Configure basic logging on import
_ensure_basic_logging()

# Re-export public API
#from openhcs.ez.api import (
#    # Core functions
#    initialize,
#    create_config,
#    run_pipeline,
#    stitch_images,
#
#    # Key types
#    PipelineConfig,
#    BackendConfig,
#    MISTConfig,
#    VirtualPath,
#    PhysicalPath,
#)
#
__all__ = [
    # Core functions
    "initialize",
    "create_config",
    "run_pipeline",
    "stitch_images",

    # Key types
    "PipelineConfig",
    "BackendConfig",
    "MISTConfig",
    "VirtualPath",
    "PhysicalPath",
]
