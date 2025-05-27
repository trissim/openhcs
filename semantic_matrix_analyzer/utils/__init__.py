"""
Utility functions for the Semantic Matrix Analyzer.

This package provides various utility functions used throughout the SMA codebase.
"""

from .import_utils import (
    optional_import,
    create_placeholder_class,
    get_missing_libraries,
    clear_import_cache
)

__all__ = [
    'optional_import',
    'create_placeholder_class',
    'get_missing_libraries',
    'clear_import_cache'
]
