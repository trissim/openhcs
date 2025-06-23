"""
Microscope-specific implementations for openhcs.

This package contains modules for different microscope types, each providing
concrete implementations of FilenameParser and MetadataHandler interfaces.
"""

# Import base components needed for registration
from openhcs.microscopes.microscope_base import MICROSCOPE_HANDLERS, METADATA_HANDLERS, create_microscope_handler

# Import concrete MicroscopeHandler implementations
# These imports trigger automatic registration via metaclass
from openhcs.microscopes.imagexpress import ImageXpressHandler
from openhcs.microscopes.opera_phenix import OperaPhenixHandler
from openhcs.microscopes.openhcs import OpenHCSMicroscopeHandler

# Import parsers and metadata handlers that might be useful to export,
# though direct use is often through the MicroscopeHandler
from openhcs.microscopes.imagexpress import (ImageXpressFilenameParser,
                                                ImageXpressMetadataHandler)
from openhcs.microscopes.opera_phenix import (OperaPhenixFilenameParser,
                                                 OperaPhenixMetadataHandler)
from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

# Note: No manual registration needed - handlers are automatically registered via metaclass

__all__ = [
    # Factory function
    'create_microscope_handler',
    # Handlers
    'ImageXpressHandler',
    'OperaPhenixHandler',
    'OpenHCSMicroscopeHandler',
    # Individual parsers and metadata handlers (optional, for direct use if needed)
    'ImageXpressFilenameParser',
    'ImageXpressMetadataHandler',
    'OperaPhenixFilenameParser',
    'OperaPhenixMetadataHandler',
    'OpenHCSMetadataHandler',
    # The MICROSCOPE_HANDLERS dict itself is not typically part of __all__
    # as it's used internally by the factory.
]
