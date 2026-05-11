"""Compatibility implementation for legacy CellProfiler Align."""

from __future__ import annotations

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.interop.cellprofiler.align_settings import AlignCropMode
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.alignment import (
    AlignAdditionalModes,
    AlignCropModeStrategy,
    AlignExecution,
    AlignShiftMeasurement,
    prepare_align,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(("align_measurements", csv_materializer(
    fields=["slice_index", "output_index", "x_shift", "y_shift"],
    analysis_type="alignment",
)))
def align(
    image: np.ndarray,
    *,
    method: str = "Mutual Information",
    crop_mode: AlignCropMode | str = AlignCropMode.KEEP_SIZE,
    additional_alignment_modes: AlignAdditionalModes = (),
    alignment_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[object, ...]:
    """Align primary images and apply declared additional-image shifts."""
    return AlignExecution(
        image=image,
        method=method,
        crop_mode=crop_mode,
        additional_alignment_modes=additional_alignment_modes,
        alignment_backend_provider=alignment_backend_provider,
    ).execute()


align.__openhcs_prepare__ = prepare_align
