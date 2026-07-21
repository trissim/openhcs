"""Setuptools compatibility entry point for OpenHCS.

Dependencies come from ``pyproject.toml``. The legacy dependency-selection
helpers below are retained for inspection but are not passed to ``setup()``;
integrated development environments install the packages under ``external/``
explicitly before installing OpenHCS.
"""

import os
import runpy
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

_KNOWLEDGE_BUILD_HELPERS = runpy.run_path(
    str(Path(__file__).resolve().parent / "scripts/build_mcp_knowledge_assets.py")
)
KNOWLEDGE_MANIFEST_RELATIVE_PATH = _KNOWLEDGE_BUILD_HELPERS[
    "KNOWLEDGE_MANIFEST_RELATIVE_PATH"
]
PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH = _KNOWLEDGE_BUILD_HELPERS[
    "PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH"
]
project_knowledge_assets = _KNOWLEDGE_BUILD_HELPERS["project_knowledge_assets"]

# External module versions for PyPI releases
PYPI_DEPENDENCIES = [
    "zmqruntime>=0.1.0",
    "pycodify>=0.1.0",
    "objectstate>=0.1.0",
    "python-introspect>=0.1.1",
    "metaclass-registry>=0.1.0",
    "arraybridge>=0.1.0",
    "polystore>=0.1.0",
    "pyqt-reactive>=0.1.0",
]

# Local external modules for development
# These are dynamically generated with absolute paths at runtime
def get_local_external_dependencies():
    """Return legacy local dependency specs; the active setup call does not use them."""
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
    Return the legacy development-mode heuristic.

    This helper does not affect ``setup()``. It remains only until the obsolete
    dependency-selection implementation is removed.
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
    Return legacy dependency specs based on the inactive mode heuristic.

    The active ``setup()`` call does not consume this result.
    """
    if is_development_mode():
        return get_local_external_dependencies()
    else:
        return PYPI_DEPENDENCIES


class BuildPyWithMcpKnowledge(_build_py):
    """Project canonical MCP knowledge sources directly into wheel build output."""

    def run(self):
        super().run()
        project_root = Path(__file__).resolve().parent
        if (project_root / KNOWLEDGE_MANIFEST_RELATIVE_PATH).is_file():
            project_knowledge_assets(
                project_root,
                Path(self.build_lib) / PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH,
            )


class SdistWithMcpKnowledge(_sdist):
    """Include projected MCP knowledge resources in source distributions."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        project_knowledge_assets(
            Path(__file__).resolve().parent,
            Path(base_dir) / PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH,
        )


# Dependencies are declared in pyproject.toml. Local packages must be installed
# explicitly for integrated development; no dependency override occurs here.
setup(
    cmdclass={
        "build_py": BuildPyWithMcpKnowledge,
        "sdist": SdistWithMcpKnowledge,
    }
)
