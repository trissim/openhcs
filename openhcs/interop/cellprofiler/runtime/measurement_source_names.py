"""CellProfiler measurement source-name projection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhcs.core.artifacts import ArtifactSpec, ArtifactSpecCollection
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.invocation import (
        CellProfilerMeasurementImage,
    )


def single_source_name(source_names: tuple[str, ...]) -> str | None:
    """Return the sole distinct source name, if exactly one exists."""
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) == 1:
        return unique_names[0]
    return None


def measurement_source_name_for_specs(
    image_inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    """Return the scalar source identity represented by declared image inputs."""
    source_names = ArtifactSpecCollection(image_inputs).names()
    distinct_source_names = tuple(
        source_name
        for index, source_name in enumerate(source_names)
        if source_name not in source_names[:index]
    )
    if not distinct_source_names:
        return None
    if len(distinct_source_names) == 1:
        return distinct_source_names[0]
    if len(distinct_source_names) == 2:
        return RuntimeMeasurementSourcePair(*distinct_source_names).source_name
    return None


def measurement_row_source_names_required(
    measurement_images: tuple["CellProfilerMeasurementImage", ...],
) -> bool:
    """Return whether row ownership must distinguish multiple image sources."""
    unique_names = tuple(
        dict.fromkeys(image.source_image_name for image in measurement_images)
    )
    return len(unique_names) > 1
