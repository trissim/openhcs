"""Measurement fact primitives for runtime equivalence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from openhcs.core.equivalence.cells import RuntimeCellSignature, runtime_cell_signature
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.runtime_semantics import MeasurementScope
from openhcs.core.runtime_values import RuntimeValue, SpatialGrid

RuntimeMeasurementFact = tuple[
    RuntimeMeasurementFeatureKey,
    RuntimeCellSignature,
]
RuntimeMeasurementFacts = tuple[RuntimeMeasurementFact, ...]
RuntimeMeasurementFactCounters = dict[
    RuntimeMeasurementFeatureKey,
    Counter[RuntimeCellSignature],
]
RuntimeRequiredMeasurementKeys = frozenset[RuntimeMeasurementFeatureKey] | None


def record_measurement_facts(
    values_by_feature: RuntimeMeasurementFactCounters,
    facts: Iterable[RuntimeMeasurementFact],
    *,
    required_keys: RuntimeRequiredMeasurementKeys = None,
) -> None:
    for key, value in facts:
        if required_keys is not None and key not in required_keys:
            continue
        values_by_feature.setdefault(key, Counter())[value] += 1


def spatial_grid_measurement_facts(
    value: RuntimeValue,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFacts:
    """Project a typed spatial-grid artifact to CellProfiler-style image facts."""
    return tuple(
        fact
        for grid in _spatial_grids_from_runtime_value(value)
        for fact in _single_spatial_grid_measurement_facts(value, grid, policy)
    )


def _single_spatial_grid_measurement_facts(
    value: RuntimeValue,
    grid: SpatialGrid,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFacts:
    grid_name = normalize_runtime_identifier(value.name or grid.name)
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
    fields = (
        ("columns", grid.columns),
        ("rows", grid.rows),
        ("x_location_of_lowest_x_spot", grid.x_location_of_lowest_x_spot),
        ("x_spacing", grid.x_spacing),
        ("y_location_of_lowest_y_spot", grid.y_location_of_lowest_y_spot),
        ("y_spacing", grid.y_spacing),
    )
    return tuple(
        (
            RuntimeMeasurementFeatureKey(
                subject,
                f"defined_grid_{grid_name}_{field_name}",
            ),
            runtime_cell_signature(str(field_value), policy),
        )
        for field_name, field_value in fields
    )


def _spatial_grids_from_runtime_value(value: RuntimeValue) -> tuple[SpatialGrid, ...]:
    if value.schema.slice_aligned:
        if not isinstance(value.data, tuple | list):
            raise TypeError(
                f"Slice-aligned spatial grid '{value.name}' payload must be a "
                f"sequence of mappings, got {type(value.data).__name__}."
            )
        return tuple(
            SpatialGrid.from_mapping(value.name, item)
            for item in value.data
            if isinstance(item, Mapping)
        )
    return (SpatialGrid.from_runtime_value(value),)
