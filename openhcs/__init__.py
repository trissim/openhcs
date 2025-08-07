"""
OpenHCS: A library for stitching microscopy images.

This module provides the public API for OpenHCS.
It re-exports only the intended public symbols from openhcs.ez.api
and does NOT import from internal modules in a way that triggers
registrations or other side-effects.
"""

import logging

__version__ = "0.1.0"

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
