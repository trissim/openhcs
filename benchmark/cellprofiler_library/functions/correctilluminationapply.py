"""
Converted from CellProfiler: CorrectIlluminationApply.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.illumination import (
    DivideIlluminationCorrectionStrategy,
    IlluminationCorrectionInputStack,
    IlluminationCorrectionMethod,
    IlluminationCorrectionSettingSequence,
    IlluminationCorrectionStrategy,
    SubtractIlluminationCorrectionStrategy,
    correct_illumination_apply,
)

__all__ = [
    "DivideIlluminationCorrectionStrategy",
    "IlluminationCorrectionInputStack",
    "IlluminationCorrectionMethod",
    "IlluminationCorrectionSettingSequence",
    "IlluminationCorrectionStrategy",
    "SubtractIlluminationCorrectionStrategy",
    "correct_illumination_apply",
]
