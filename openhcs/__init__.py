"""
OpenHCS: A library for stitching microscopy images.

This module provides the public API for OpenHCS.
It re-exports only the intended public symbols from openhcs.ez.api
and does NOT import from internal modules in a way that triggers
registrations or other side-effects.
"""

import os
import platform
import sys

from openhcs._source_dependencies import ensure_source_checkout_external_paths
from openhcs.utils.environment import OpenHCSProcessEnvironment

__version__ = "0.7.26"

# Configure polystore defaults for OpenHCS integration
os.environ.setdefault("POLYSTORE_METADATA_FILENAME", "openhcs_metadata.json")
OpenHCSProcessEnvironment.project_dependency_gpu_import_policy()

ensure_source_checkout_external_paths()

# Force UTF-8 encoding for stdout/stderr on Windows
# This ensures emoji and Unicode characters work in console output
if platform.system() == "Windows":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Re-export public API
# from openhcs.ez.api import (
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
# )
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
