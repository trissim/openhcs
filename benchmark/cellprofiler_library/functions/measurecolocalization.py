"""Compatibility aliases for CellProfiler MeasureColocalization backend semantics."""

from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationCostesBackendStrategy,
    ColocalizationCostesThresholdBatch,
    ColocalizationCostesThresholdRequest,
    ColocalizationCostesThresholds,
    ColocalizationImagePairContext,
    ColocalizationMeasurementOptions,
    ColocalizationMeasurementSchema,
    ColocalizationMeasurements,
    ColocalizationObjectLabelContext,
    CostesMethod,
    ObjectColocalizationMeasurements,
    _costes_first_channel_bin_threshold,
    _divide_costes_measurements,
    measure_colocalization,
    measure_colocalization_objects,
    measure_colocalization_objects_batch,
)

__all__ = [
    "ColocalizationCostesBackendStrategy",
    "ColocalizationCostesThresholdBatch",
    "ColocalizationCostesThresholdRequest",
    "ColocalizationCostesThresholds",
    "ColocalizationImagePairContext",
    "ColocalizationMeasurementOptions",
    "ColocalizationMeasurementSchema",
    "ColocalizationMeasurements",
    "ColocalizationObjectLabelContext",
    "CostesMethod",
    "ObjectColocalizationMeasurements",
    "_costes_first_channel_bin_threshold",
    "_divide_costes_measurements",
    "measure_colocalization",
    "measure_colocalization_objects",
    "measure_colocalization_objects_batch",
]
