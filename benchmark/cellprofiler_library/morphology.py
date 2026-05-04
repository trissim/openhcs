"""Compatibility import for the canonical OpenHCS CellProfiler backend seam."""

from openhcs.processing.backends.cellprofiler.morphology import (
    CentrosomeNumpyMorphologyBackendStrategy,
    HolePredicate,
    MorphologyBackendStrategy,
    NumpyMorphologyBackendStrategy,
)

__all__ = [
    "CentrosomeNumpyMorphologyBackendStrategy",
    "HolePredicate",
    "MorphologyBackendStrategy",
    "NumpyMorphologyBackendStrategy",
]
