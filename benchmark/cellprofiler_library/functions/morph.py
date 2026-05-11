"""
Converted from CellProfiler: Morph.

The CellProfiler Morph operation semantics live in the OpenHCS morphology
backend.  This module only exposes the decorated benchmark-library entrypoint
and preserves the historical import surface for tests/conversion code.
"""

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MorphOperation,
    MorphOperationRequest,
    MorphOperationStrategy,
    RepeatMode,
    RepeatModeStrategy,
    apply_morph_operation,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def morph(
    image: np.ndarray,
    operation: MorphOperation = MorphOperation.THIN,
    repeat_mode: RepeatMode = RepeatMode.ONCE,
    custom_repeats: int = 2,
    rescale_values: bool = True,
    line_length: int = 3,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """
    Perform CellProfiler-compatible morphological operations.

    Args:
        image: Input image (H, W), converted to binary for most operations.
        operation: Morph operation to perform.
        repeat_mode: Repeat policy.
        custom_repeats: Number of repetitions when repeat_mode is CUSTOM.
        rescale_values: For DISTANCE operation, rescale output to 0-1.
        line_length: For OPENLINES operation, minimum line length to keep.

    Returns:
        Processed image (H, W).
    """
    return apply_morph_operation(
        image=image,
        operation=operation,
        repeat_mode=repeat_mode,
        custom_repeats=custom_repeats,
        rescale_values=rescale_values,
        line_length=line_length,
        morphology_backend_provider=morphology_backend_provider,
    )


__all__ = [
    "MorphOperation",
    "MorphOperationRequest",
    "MorphOperationStrategy",
    "RepeatMode",
    "RepeatModeStrategy",
    "morph",
]
