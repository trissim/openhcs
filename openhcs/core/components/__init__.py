"""
Generic component framework for OpenHCS.

This module provides a generic, configurable system for handling variable components
in OpenHCS pipelines, replacing the hardcoded Well-based multiprocessing and fixed
4-component assumptions.
"""

# Import from the new location (framework.py moved to openhcs/components/)
from openhcs.components.framework import ComponentConfiguration, ComponentConfigurationFactory
from .multiprocessing import MultiprocessingCoordinator
from .validation import GenericValidator
from .parser_metaprogramming import FilenameParseResult, GenericFilenameParser

__all__ = [
    'ComponentConfiguration',
    'ComponentConfigurationFactory',
    'MultiprocessingCoordinator',
    'GenericValidator',
    'GenericFilenameParser',
    'FilenameParseResult',
]
