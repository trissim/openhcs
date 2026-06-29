"""
Dynamic setup.py for openhcs.

This setup.py allows for different dependency sources:
- PyPI releases: Use pip versions from PyPI
- Development mode: Use local external modules from external/ directory

Usage:
    # Production/PyPI build (uses pip versions)
    pip install .
    python -m build

    # Development install (uses local external modules)
    pip install -e .
"""

import os
from pathlib import Path
from setuptools import setup

# External module versions for PyPI releases
PYPI_DEPENDENCIES = [
    "zmqruntime>=0.1.8",
    "pycodify>=0.1.0",
    "objectstate>=1.0.17",
    "python-introspect>=0.1.1",
    "metaclass-registry>=0.1.4",
    "arraybridge>=0.2.10",
    "polystore>=0.1.9",
    "pyqt-reactive>=0.1.21",
]

# Local external modules for development
# These are dynamically generated with absolute paths at runtime
def get_local_external_dependencies():
    """Generate local external dependencies with absolute paths."""
    project_root = Path(__file__).parent.resolve()
    return [
        f"ObjectState @ file://{project_root}/external/ObjectState",
        f"python-introspect @ file://{project_root}/external/python-introspect",
        f"metaclass-registry @ file://{project_root}/external/metaclass-registry",
        f"arraybridge @ file://{project_root}/external/arraybridge",
        f"polystore @ file://{project_root}/external/PolyStore",
        f"pyqt-reactive @ file://{project_root}/external/pyqt-reactive",
        f"zmqruntime @ file://{project_root}/external/zmqruntime",
        f"pycodify @ file://{project_root}/external/pycodify",
    ]


def is_development_mode():
    """
    Determine if we're in development mode.

    Development mode is detected when:
    1. The external/ directory exists with git submodules
    2. OPENHCS_DEV_MODE environment variable is set to "1", "true", or "yes"
    """
    # Check for external directory
    project_root = Path(__file__).parent
    external_dir = project_root / "external"
    has_external = external_dir.exists()

    # Check for explicit dev mode flag
    dev_mode_env = os.environ.get("OPENHCS_DEV_MODE", "").lower() in ("1", "true", "yes")

    result = has_external or dev_mode_env
    if result:
        print("openhcs: Installing in DEVELOPMENT mode (using local external modules)")
    else:
        print("openhcs: Installing in PRODUCTION mode (using PyPI versions)")
    return result


def get_external_dependencies():
    """
    Get external dependencies based on install mode.

    Returns:
        list: List of dependency specifications
    """
    if is_development_mode():
        return get_local_external_dependencies()
    else:
        return PYPI_DEPENDENCIES


# Call setup without install_requires (dependencies are in pyproject.toml)
# External modules are in pyproject.toml for production builds
# Local external modules are used only in development mode via pyproject.toml overrides
setup()
