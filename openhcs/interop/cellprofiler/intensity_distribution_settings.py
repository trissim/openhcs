"""CellProfiler MeasureObjectIntensityDistribution setting semantics."""

from __future__ import annotations

from enum import Enum

from openhcs.core.registry_strategies import enum_member_with_payload

from .settings_binder import coerce_cellprofiler_enum


class IntensityDistributionCenterChoice(Enum):
    """Nominal CP center choices for radial intensity distribution."""

    def __new__(
        cls,
        absorbed_value: str,
        *cellprofiler_literals: str,
    ) -> "IntensityDistributionCenterChoice":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

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
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    NONE = ("none",)
    MAGNITUDES = ("magnitudes", "Magnitudes only")
    MAGNITUDES_AND_PHASE = ("magnitudes_and_phase", "Magnitudes and phase")


def parse_intensity_distribution_zernike_mode(value: str) -> str:
    """Return the absorbed-function Zernike mode literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionZernikeMode, value).value


def parse_intensity_distribution_center_choice(value: str) -> str:
    """Return the absorbed-function center-choice literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionCenterChoice, value).value
