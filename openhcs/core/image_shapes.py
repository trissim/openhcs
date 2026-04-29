"""Shared image payload shape predicates for OpenHCS runtime paths."""

from __future__ import annotations

from typing import Any


COLOR_CHANNEL_COUNTS = frozenset((3, 4))


def is_grayscale_image_slice(value: Any) -> bool:
    """Return True for one 2D grayscale image plane."""
    return hasattr(value, "ndim") and value.ndim == 2


def is_color_image_slice(value: Any) -> bool:
    """Return True for one HWC RGB/RGBA image plane."""
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and value.ndim == 3
        and value.shape[-1] in COLOR_CHANNEL_COUNTS
    )


def is_grayscale_image_stack(value: Any) -> bool:
    """Return True for an OpenHCS grayscale stack shaped (N, H, W)."""
    return hasattr(value, "ndim") and value.ndim == 3 and not is_color_image_slice(value)


def is_color_image_stack(value: Any) -> bool:
    """Return True for an OpenHCS color stack shaped (N, H, W, C)."""
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and value.ndim == 4
        and value.shape[-1] in COLOR_CHANNEL_COUNTS
    )


def is_image_stack(value: Any) -> bool:
    """Return True for OpenHCS main-flow image stacks."""
    return is_grayscale_image_stack(value) or is_color_image_stack(value)
