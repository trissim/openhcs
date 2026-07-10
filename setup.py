"""Compatibility entry point; package metadata lives in pyproject.toml.

Use ``python scripts/dev_install.py`` to install OpenHCS and its pinned
submodules as editable projects. A bare ``pip install -e .`` uses the
production dependency metadata from pyproject.toml.
"""

from setuptools import setup


setup()
