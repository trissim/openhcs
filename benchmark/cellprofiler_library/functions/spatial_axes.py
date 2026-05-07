"""Helpers for CellProfiler operations over trailing spatial axes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np

ArrayT = TypeVar("ArrayT", bound=np.ndarray)


def trailing_spatial_target_shape(
    shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Return a full shape that preserves leading axes before spatial axes."""
    spatial_rank = len(spatial_shape)
    if spatial_rank <= 0:
        raise ValueError("spatial_shape must contain at least one axis.")
    if len(shape) < spatial_rank:
        raise ValueError(
            "Cannot apply spatial shape with rank greater than input rank: "
            f"{spatial_shape!r} for {shape!r}."
        )
    return (*shape[: len(shape) - spatial_rank], *spatial_shape)


def trailing_spatial_factors(
    ndim: int,
    spatial_factors: tuple[float, ...],
) -> tuple[float, ...]:
    """Return full-rank factors that preserve leading axes before spatial axes."""
    spatial_rank = len(spatial_factors)
    if spatial_rank <= 0:
        raise ValueError("spatial_factors must contain at least one axis.")
    if ndim < spatial_rank:
        raise ValueError(
            "Cannot apply spatial factors with rank greater than input rank: "
            f"{spatial_factors!r} for ndim={ndim}."
        )
    return (*((1.0,) * (ndim - spatial_rank)), *spatial_factors)


def apply_over_trailing_spatial_axes(
    array: ArrayT,
    spatial_rank: int,
    operation: Callable[[ArrayT], ArrayT],
    *,
    fill_value: object = 0,
) -> ArrayT:
    """Apply an operation to trailing spatial axes while preserving leading axes."""
    if spatial_rank <= 0:
        raise ValueError("spatial_rank must be positive.")
    if array.ndim < spatial_rank:
        raise ValueError(
            f"Cannot apply spatial_rank={spatial_rank} to shape {array.shape!r}."
        )
    if array.ndim == spatial_rank:
        return operation(array)
    output = np.full_like(array, fill_value)
    leading_shape = array.shape[: array.ndim - spatial_rank]
    for leading_index in np.ndindex(leading_shape):
        output[leading_index] = operation(array[leading_index])
    return output
