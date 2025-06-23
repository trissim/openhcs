"""
Microscope interfaces for openhcs.

This module provides abstract base classes for handling microscope-specific
functionality, including filename parsing and metadata handling.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.microscopes.imagexpress import ImageXpressHandler
from openhcs.microscopes.microscope_base import (MICROSCOPE_HANDLERS,
                                                    MicroscopeHandler)
from openhcs.microscopes.opera_phenix import OperaPhenixHandler

logger = logging.getLogger(__name__)

# DEPRECATED: Manual registration removed - handlers are now auto-registered via metaclass
# The imports below trigger automatic registration through MicroscopeHandlerMeta


# DEPRECATED: Old factory function removed - use microscope_base.create_microscope_handler
# The canonical factory function is now in microscope_base.py and uses handler-based auto-detection
# via the metaclass registration system instead of hardcoded filename checks.
