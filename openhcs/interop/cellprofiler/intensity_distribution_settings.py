"""CellProfiler MeasureObjectIntensityDistribution setting semantics."""

from __future__ import annotations

from enum import Enum

from .settings_binder import coerce_cellprofiler_enum, normalize_cellprofiler_setting_name


class IntensityDistributionCenterChoice(Enum):
    """Nominal CP center choices for radial intensity distribution."""

    SELF = "self"
    CENTERS_OF_OTHER = "centers_of_other"
    EDGES_OF_OTHER = "edges_of_other"


class IntensityDistributionZernikeMode(Enum):
    """Nominal CP Zernike output modes for intensity distribution."""

    NONE = "none"
    MAGNITUDES = "magnitudes"
    MAGNITUDES_AND_PHASE = "magnitudes_and_phase"


def parse_intensity_distribution_zernike_mode(value: str) -> str:
    """Return the absorbed-function Zernike mode literal for a CP setting."""
    normalized = normalize_cellprofiler_setting_name(value)
    if normalized == "magnitudes_only":
        return IntensityDistributionZernikeMode.MAGNITUDES.value
    return coerce_cellprofiler_enum(IntensityDistributionZernikeMode, value).value


def parse_intensity_distribution_center_choice(value: str) -> str:
    """Return the absorbed-function center-choice literal for a CP setting."""
    normalized = normalize_cellprofiler_setting_name(value)
    aliases = {
        "these_objects": IntensityDistributionCenterChoice.SELF,
        "self": IntensityDistributionCenterChoice.SELF,
        "centers_of_other_objects": IntensityDistributionCenterChoice.CENTERS_OF_OTHER,
        "centers_of_other": IntensityDistributionCenterChoice.CENTERS_OF_OTHER,
        "edges_of_other_objects": IntensityDistributionCenterChoice.EDGES_OF_OTHER,
        "edges_of_other": IntensityDistributionCenterChoice.EDGES_OF_OTHER,
    }
    if normalized in aliases:
        return aliases[normalized].value
    return coerce_cellprofiler_enum(IntensityDistributionCenterChoice, value).value
