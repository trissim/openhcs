"""
OpenHCS: A library for stitching microscopy images.

This module provides the public API for OpenHCS.
It re-exports only the intended public symbols from openhcs.ez.api
and does NOT import from internal modules in a way that triggers
registrations or other side-effects.
"""

import logging

__version__ = "0.1.0"

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
