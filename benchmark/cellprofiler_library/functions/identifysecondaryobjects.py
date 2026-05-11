"""
Converted from CellProfiler: IdentifySecondaryObjects
Original: IdentifySecondaryObjects.run

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.secondary import (
    THRESHOLD_CALCULATOR_DECLARATIONS,
    DistanceMaskedSegmentationStrategy,
    DistanceOnlySegmentationStrategy,
    GradientWatershedSegmentationStrategy,
    ImageWatershedSegmentationStrategy,
    PropagationSegmentationStrategy,
    SecondaryImageInputs,
    SecondaryMethod,
    SecondaryObjectLabels,
    SecondaryObjectStats,
    SecondarySegmentationRequest,
    SecondarySegmentationStrategy,
    SecondaryThresholdRequest,
    SecondaryThresholdResult,
    ThresholdCalculator,
    ThresholdCalculatorDeclaration,
    ThresholdMethod,
    identify_secondary_objects,
)

__all__ = [
    "DistanceMaskedSegmentationStrategy",
    "DistanceOnlySegmentationStrategy",
    "GradientWatershedSegmentationStrategy",
    "ImageWatershedSegmentationStrategy",
    "PropagationSegmentationStrategy",
    "SecondaryImageInputs",
    "SecondaryMethod",
    "SecondaryObjectLabels",
    "SecondaryObjectStats",
    "SecondarySegmentationRequest",
    "SecondarySegmentationStrategy",
    "SecondaryThresholdRequest",
    "SecondaryThresholdResult",
    "THRESHOLD_CALCULATOR_DECLARATIONS",
    "ThresholdCalculator",
    "ThresholdCalculatorDeclaration",
    "ThresholdMethod",
    "identify_secondary_objects",
]
