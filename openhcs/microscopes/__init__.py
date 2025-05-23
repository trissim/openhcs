"""
Microscope-specific implementations for openhcs.

This package contains modules for different microscope types, each providing
concrete implementations of FilenameParser and MetadataHandler interfaces.
"""

# Import microscope handlers for easier access
from openhcs.microscopes.imagexpress import (ImageXpressFilenameParser,
                                                ImageXpressMetadataHandler)
from openhcs.microscopes.opera_phenix import (OperaPhenixFilenameParser,
                                                 OperaPhenixMetadataHandler)

__all__ = [
    'ImageXpressFilenameParser',
    'ImageXpressMetadataHandler',
    'OperaPhenixFilenameParser',
    'OperaPhenixMetadataHandler'
]
