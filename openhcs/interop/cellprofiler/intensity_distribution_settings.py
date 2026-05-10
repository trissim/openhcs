"""CellProfiler MeasureObjectIntensityDistribution setting semantics."""

from __future__ import annotations

from enum import Enum

from .settings_binder import coerce_cellprofiler_enum


class IntensityDistributionCenterChoice(Enum):
    """Nominal CP center choices for radial intensity distribution."""

    def __new__(
        cls,
        absorbed_value: str,
        *cellprofiler_literals: str,
    ) -> "IntensityDistributionCenterChoice":
        obj = object.__new__(cls)
        obj._value_ = absorbed_value
        obj.cellprofiler_literals = (absorbed_value, *cellprofiler_literals)
        return obj

    SELF = ("self", "These objects")
    CENTERS_OF_OTHER = (
        "centers_of_other",
        "Centers of other objects",
    )
    EDGES_OF_OTHER = (
        "edges_of_other",
        "Edges of other objects",
    )


class IntensityDistributionZernikeMode(Enum):
    """Nominal CP Zernike output modes for intensity distribution."""

    def __new__(
        cls,
        absorbed_value: str,
        *cellprofiler_literals: str,
    ) -> "IntensityDistributionZernikeMode":
        obj = object.__new__(cls)
        obj._value_ = absorbed_value
        obj.cellprofiler_literals = (absorbed_value, *cellprofiler_literals)
        return obj

    NONE = ("none",)
    MAGNITUDES = ("magnitudes", "Magnitudes only")
    MAGNITUDES_AND_PHASE = ("magnitudes_and_phase", "Magnitudes and phase")


def parse_intensity_distribution_zernike_mode(value: str) -> str:
    """Return the absorbed-function Zernike mode literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionZernikeMode, value).value


def parse_intensity_distribution_center_choice(value: str) -> str:
    """Return the absorbed-function center-choice literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionCenterChoice, value).value
