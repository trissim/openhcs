"""
Converted from CellProfiler: Smooth.

CellProfiler Smooth semantics live in the OpenHCS smoothing backend.  This
module keeps the decorated function boundary, compiler warmup, and runtime
batch registration used by the benchmark/library loader.
"""

from typing import Any

import numpy as np

from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.smoothing import (
    SmoothingBackendProviderPolicy,
    SmoothingBackendSelectionRequest,
    SmoothingMethod,
    SmoothingRequest,
    SmoothingStrategy,
    SmoothingStrategyKey,
    smooth_image,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def smooth(
    image: np.ndarray,
    smoothing_method: SmoothingMethod = SmoothingMethod.GAUSSIAN_FILTER,
    auto_object_size: bool = True,
    object_size: float = 16.0,
    edge_intensity_difference: float = 0.1,
    clip_polynomial: bool = True,
    smoothing_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """
    Smooth (blur) an image using CellProfiler-compatible filtering methods.

    Args:
        image: Input grayscale image (H, W).
        smoothing_method: Method to use for smoothing.
        auto_object_size: If True, calculate artifact diameter automatically.
        object_size: Typical artifact diameter in pixels if auto sizing is off.
        edge_intensity_difference: Edge threshold for smooth_keeping_edges.
        clip_polynomial: Whether to clip polynomial fit results to [0, 1].

    Returns:
        Smoothed image (H, W).
    """
    return smooth_image(
        image=image,
        smoothing_method=smoothing_method,
        auto_object_size=auto_object_size,
        object_size=object_size,
        edge_intensity_difference=edge_intensity_difference,
        clip_polynomial=clip_polynomial,
        smoothing_backend_provider=smoothing_backend_provider,
    )


def _smooth_batch(request: RuntimePure2DSliceBatchRequest) -> list[Any]:
    slices_2d = request.slices_2d
    kwargs = request.kwargs
    smoothing_method = coerce_cellprofiler_enum(
        SmoothingMethod,
        kwargs.get("smoothing_method", SmoothingMethod.GAUSSIAN_FILTER),
    )
    pixel_stack = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(image_payload_data(slice_2d), dtype=np.float32)
                for slice_2d in slices_2d
            ],
            axis=0,
        ),
    )
    selection_request = SmoothingBackendSelectionRequest(
        method=smoothing_method,
        auto_object_size=bool(kwargs.get("auto_object_size", True)),
        object_size=float(kwargs.get("object_size", 16.0)),
        image_shape=tuple(int(axis) for axis in pixel_stack.shape[1:]),
    )
    backend_provider = SmoothingBackendProviderPolicy.resolve(
        smoothing_method,
        kwargs.get(
            "smoothing_backend_provider",
            DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        ),
        selection_request,
    )
    strategy = SmoothingStrategy.for_key(
        SmoothingStrategyKey(backend_provider, smoothing_method)
    )
    if not strategy.supports_stack_batch:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]

    masks = tuple(image_payload_mask(slice_2d) for slice_2d in slices_2d)
    mask_stack = None
    if any(mask is not None for mask in masks):
        mask_stack = np.stack(
            [
                np.ones(pixel_stack.shape[1:], dtype=bool)
                if mask is None
                else np.asarray(mask, dtype=bool)
                for mask in masks
            ],
            axis=0,
        )

    output_stack = strategy.smooth_stack(
        pixel_stack,
        mask_stack,
        float(selection_request.sigma),
    ).astype(
        np.float32,
        copy=False,
    )

    return [
        image_payload_with_context(
            output_stack[slice_index],
            mask=masks[slice_index],
            metadata=image_payload_metadata(
                slice_2d
            ).without_unit_interval_intensity_scale(),
        )
        for slice_index, slice_2d in enumerate(slices_2d)
    ]


@processing_prepare(smooth)
def _prepare_smooth() -> None:
    """Compile default Gaussian smoothing before timed execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    smooth.__wrapped__(image)


pure_2d_batch_executor(_smooth_batch)(smooth)


__all__ = [
    "SmoothingBackendProviderPolicy",
    "SmoothingBackendSelectionRequest",
    "SmoothingMethod",
    "SmoothingRequest",
    "SmoothingStrategy",
    "SmoothingStrategyKey",
    "smooth",
]
