"""Setuptools build hooks for projecting OpenHCS MCP knowledge resources.

Dependency and package metadata comes exclusively from ``pyproject.toml``.
"""

import runpy
import shutil
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


class BuildPyWithMcpKnowledge(_build_py):
    """Project a fresh OpenHCS package and its canonical MCP knowledge assets."""

    def run(self):
        package_build_root = Path(self.build_lib) / "openhcs"
        if package_build_root.exists():
            shutil.rmtree(package_build_root)
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


setup(
    cmdclass={
        "build_py": BuildPyWithMcpKnowledge,
        "sdist": SdistWithMcpKnowledge,
    }
)
