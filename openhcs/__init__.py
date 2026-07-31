"""
OpenHCS: A library for stitching microscopy images.

This module provides the public API for OpenHCS.
It re-exports only the intended public symbols from openhcs.ez.api
and does NOT import from internal modules in a way that triggers
registrations or other side-effects.
"""

import logging
import os
import platform
import sys

from openhcs._source_dependencies import ensure_source_checkout_external_paths

__version__ = "0.7.2"

# Configure polystore defaults for OpenHCS integration
os.environ.setdefault("POLYSTORE_METADATA_FILENAME", "openhcs_metadata.json")
if os.getenv("OPENHCS_SUBPROCESS_NO_GPU") == "1":
    os.environ.setdefault("POLYSTORE_SUBPROCESS_NO_GPU", "1")

ensure_source_checkout_external_paths()

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
    configured_level_name = os.environ.get("OPENHCS_LOG_LEVEL", "INFO").upper()
    configured_level = getattr(logging, configured_level_name, None)
    if not isinstance(configured_level, int):
        raise ValueError(f"Unknown OPENHCS_LOG_LEVEL: {configured_level_name!r}")

    # Only configure if no handlers exist and level is too high
    if not root_logger.handlers and root_logger.level > configured_level:
        logging.basicConfig(
            level=configured_level,
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
