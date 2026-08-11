"""
Experimental analysis backend system for OpenHCS.

This module provides the standalone engine for processing experimental-analysis
data from the workbook-declared CX5 or MetaXpress result format.

Result-format behavior is selected through the scope-keyed strategies in
``openhcs.formats.experimental_result_formats``. The workbook scope remains the
single declaration of its input format.
"""

from .unified_analysis_engine import DataProcessingError, ExperimentalAnalysisEngine

__all__ = [
    "DataProcessingError",
    "ExperimentalAnalysisEngine",
]
