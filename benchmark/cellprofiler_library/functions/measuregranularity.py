"""
Converted from CellProfiler: MeasureGranularity
Original: MeasureGranularity module
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_schemas import (
    DataclassCompanionSchema,
    DataclassFieldInsertion,
)
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler.granularity import (
    GranularityImageSeriesRequest,
    object_granularity_values,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


GRANULARITY_FIELDS = [
    "slice_index",
    "gs1",
    "gs2",
    "gs3",
    "gs4",
    "gs5",
    "gs6",
    "gs7",
    "gs8",
    "gs9",
    "gs10",
    "gs11",
    "gs12",
    "gs13",
    "gs14",
    "gs15",
    "gs16",
]


@dataclass
class GranularityMeasurement:
    """Granularity spectrum measurements for an image."""

    slice_index: int
    gs1: float
    gs2: float
    gs3: float
    gs4: float
    gs5: float
    gs6: float
    gs7: float
    gs8: float
    gs9: float
    gs10: float
    gs11: float
    gs12: float
    gs13: float
    gs14: float
    gs15: float
    gs16: float


ObjectGranularityMeasurement = DataclassCompanionSchema(
    source_type=GranularityMeasurement,
    companion_name="ObjectGranularityMeasurement",
    insertions=(
        DataclassFieldInsertion("object_id", int, after_field="slice_index"),
    ),
    module_name=__name__,
    doc="Granularity spectrum measurements per object.",
).materialize()


def _granularity_measurement(gs_values: list[float]) -> GranularityMeasurement:
    while len(gs_values) < 16:
        gs_values.append(0.0)
    return GranularityMeasurement(
        slice_index=0,
        gs1=gs_values[0],
        gs2=gs_values[1],
        gs3=gs_values[2],
        gs4=gs_values[3],
        gs5=gs_values[4],
        gs6=gs_values[5],
        gs7=gs_values[6],
        gs8=gs_values[7],
        gs9=gs_values[8],
        gs10=gs_values[9],
        gs11=gs_values[10],
        gs12=gs_values[11],
        gs13=gs_values[12],
        gs14=gs_values[13],
        gs15=gs_values[14],
        gs16=gs_values[15],
    )


def _object_granularity_measurement(
    object_id: int,
    gs: np.ndarray,
) -> ObjectGranularityMeasurement:
    return ObjectGranularityMeasurement(
        slice_index=0,
        object_id=int(object_id),
        gs1=gs[0],
        gs2=gs[1],
        gs3=gs[2],
        gs4=gs[3],
        gs5=gs[4],
        gs6=gs[5],
        gs7=gs[6],
        gs8=gs[7],
        gs9=gs[8],
        gs10=gs[9],
        gs11=gs[10],
        gs12=gs[11],
        gs13=gs[12],
        gs14=gs[13],
        gs15=gs[14],
        gs16=gs[15],
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "granularity_measurements",
        csv_materializer(
            fields=GRANULARITY_FIELDS,
            analysis_type="granularity",
        ),
    )
)
def measure_granularity(
    image: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> Tuple[np.ndarray, GranularityMeasurement]:
    """Measure granularity spectrum of an image."""
    series = GranularityImageSeriesRequest(
        image=image,
        subsample_size=subsample_size,
        background_subsample_size=background_subsample_size,
        element_radius=element_radius,
        spectrum_length=spectrum_length,
        profile_function="measure_granularity",
    ).series()
    pixels = series.pixels

    startmean = max(np.mean(pixels), np.finfo(float).eps)
    currentmean = startmean
    gs_values = []
    for index, reconstruction in enumerate(series.reconstructions):
        prevmean = currentmean
        currentmean = np.mean(reconstruction)
        gs = (prevmean - currentmean) * 100 / startmean
        if index > 0 and gs < 0.0:
            gs = 0.0
        gs_values.append(gs)

    return image, _granularity_measurement(gs_values)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "object_granularity_measurements",
        csv_materializer(
            fields=["slice_index", "object_id", *GRANULARITY_FIELDS[1:]],
            analysis_type="object_granularity",
        ),
    )
)
def measure_granularity_objects(
    image: np.ndarray,
    labels: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> Tuple[np.ndarray, List[ObjectGranularityMeasurement]]:
    """Measure granularity spectrum within labeled objects."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    object_range = np.unique(labels[labels > 0]).astype(np.int32, copy=False)
    if object_range.size == 0:
        return image, []

    series = GranularityImageSeriesRequest(
        image=image,
        subsample_size=subsample_size,
        background_subsample_size=background_subsample_size,
        element_radius=element_radius,
        spectrum_length=spectrum_length,
        profile_function="measure_granularity_objects",
    ).series()
    gs_per_object = object_granularity_values(
        image,
        labels,
        object_range,
        series,
        subsample_size=subsample_size,
        spectrum_length=spectrum_length,
    )
    return image, [
        _object_granularity_measurement(int(object_id), gs_per_object[index])
        for index, object_id in enumerate(object_range)
    ]


def _prepare_granularity_backend() -> None:
    """Compile Numba kernels used by the granularity backend before execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_granularity.__wrapped__(
        image,
        subsample_size=1.0,
        background_subsample_size=0.25,
        element_radius=10,
        spectrum_length=5,
    )
    measure_granularity_objects.__wrapped__(
        image,
        labels,
        subsample_size=1.0,
        background_subsample_size=0.25,
        element_radius=10,
        spectrum_length=5,
    )


measure_granularity.__openhcs_prepare__ = _prepare_granularity_backend
measure_granularity_objects.__openhcs_prepare__ = _prepare_granularity_backend
