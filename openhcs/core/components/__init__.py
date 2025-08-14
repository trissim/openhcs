"""
Generic component framework for OpenHCS.

This module provides a generic, configurable system for handling variable components
in OpenHCS pipelines, replacing the hardcoded Well-based multiprocessing and fixed
4-component assumptions.
"""

from .framework import ComponentConfiguration, ComponentConfigurationFactory
from .multiprocessing import MultiprocessingCoordinator
from .validation import GenericValidator
from .patterns import GenericPatternEngine

__all__ = [
    'ComponentConfiguration',
    'ComponentConfigurationFactory', 
    'MultiprocessingCoordinator',
    'GenericValidator',
    'GenericPatternEngine'
]
