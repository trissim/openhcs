"""
Converted from CellProfiler: CorrectIlluminationApply.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.illumination import (
    DivideIlluminationCorrectionStrategy,
    IlluminationCorrection,
    IlluminationCorrectionInputStack,
    IlluminationCorrectionMethod,
    IlluminationCorrectionRequest,
    IlluminationCorrectionSettingSequence,
    IlluminationCorrectionStrategy,
    SubtractIlluminationCorrectionStrategy,
    correct_illumination_apply,
)

__all__ = [
    "DivideIlluminationCorrectionStrategy",
    "IlluminationCorrection",
    "IlluminationCorrectionInputStack",
    "IlluminationCorrectionMethod",
    "IlluminationCorrectionRequest",
    "IlluminationCorrectionSettingSequence",
    "IlluminationCorrectionStrategy",
    "SubtractIlluminationCorrectionStrategy",
    "correct_illumination_apply",
]
