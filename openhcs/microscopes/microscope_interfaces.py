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

# Register the handlers
MICROSCOPE_HANDLERS['opera_phenix'] = OperaPhenixHandler
MICROSCOPE_HANDLERS['imagexpress'] = ImageXpressHandler


# Factory function
def create_microscope_handler(plate_folder: Union[str, Path],
                              filemanager: FileManager,
                             microscope_type: str = 'auto',
                             pattern_format: Optional[str] = None) -> MicroscopeHandler:
    """
    Factory function to create a microscope handler.

    This function enforces explicit dependency injection by requiring a FileManager
    instance to be provided. This ensures that all components requiring file operations
    receive their dependencies explicitly, eliminating runtime fallbacks and enforcing
    declarative configuration.

    Args:
        microscope_type: 'auto', 'imagexpress', 'opera_phenix'.
        plate_folder: Required for 'auto' detection.
        filemanager: FileManager instance. Must be provided.
        pattern_format: Name of the pattern format to use.

    Returns:
        An initialized MicroscopeHandler instance.

    Raises:
        ValueError: If filemanager is None or if microscope_type cannot be determined.
    """
    if filemanager is None:
        raise ValueError(
            "FileManager must be provided to create_microscope_handler. "
            "Default fallback has been removed."
        )

    logger.info("Using provided FileManager for microscope handler.")

    # Auto-detect microscope type if needed
    if microscope_type == 'auto':
        microscope_type = _auto_detect_microscope_type(plate_folder, filemanager)
        logger.info("Auto-detected microscope type: %s", microscope_type)

    # Get the appropriate handler class from the constant mapping
    # No dynamic imports or fallbacks (Clause 77: Rot Intolerance)
    handler_class = MICROSCOPE_HANDLERS.get(microscope_type.lower())
    if not handler_class:
        raise ValueError(
            f"Unsupported microscope type: {microscope_type}. "
            f"Supported types: {list(MICROSCOPE_HANDLERS.keys())}"
        )

    # Create and configure the handler
    logger.info("Creating %s", handler_class.__name__)
    handler = handler_class(filemanager, pattern_format=pattern_format)

    return handler


def _auto_detect_microscope_type(plate_folder: Path, filemanager: FileManager) -> str:
    """
    Auto-detect microscope type based on files in the plate folder.

    Args:
        plate_folder: Path to the plate folder
        filemanager: FileManager instance

    Returns:
        Detected microscope type as string

    Raises:
        ValueError: If microscope type cannot be determined
    """
    try:
        # Check for Opera Phenix (Index.xml)
        # Use list_files with pattern instead of find_file_recursive
        if filemanager.list_files(
            plate_folder, Backend.DISK.value, pattern="Index.xml", recursive=True
        ):
            logger.info("Auto-detected Opera Phenix microscope type.")
            # Use consistent key from MICROSCOPE_HANDLERS (Clause 77: Rot Intolerance)
            return 'opera_phenix'

        # Check for ImageXpress (.htd files)
        if filemanager.list_files(
            plate_folder, Backend.DISK.value, extensions={'.htd','.HTD'}, recursive=True
        ):
            logger.info("Auto-detected ImageXpress microscope type.")
            # Use consistent key from MICROSCOPE_HANDLERS (Clause 77: Rot Intolerance)
            return 'imagexpress'

        # No known microscope type detected - fail deterministically (Clause 12: Smell Intolerance)
        supported_types = list(MICROSCOPE_HANDLERS.keys())
        msg = (f"Could not auto-detect microscope type in {plate_folder}. "
               f"Supported types: {supported_types}")
        logger.error(msg)
        raise ValueError(msg)

    except Exception as e:
        # Wrap exception with clear context (Clause 12: Smell Intolerance)
        raise ValueError(
            f"Error during microscope type auto-detection in {plate_folder}: {e}"
        ) from e
