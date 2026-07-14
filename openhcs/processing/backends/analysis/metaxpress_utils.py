"""Shared primitives for MetaXpress-style 2D analysis backends."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class HiddenPixelSize(float):
    """Pipeline-injected pixel size that is intentionally absent from the UI."""

    _ui_hidden = True


def odd_size(value: float) -> int:
    """Round a derived spatial scale up to a positive odd integer."""

    size = max(1, int(np.ceil(value)))
    return size if size % 2 else size + 1


def local_background_response(
    image: np.ndarray,
    *,
    object_width_px: float,
    bright_objects: bool,
) -> np.ndarray:
    """Return intensity above or below an adaptive local background.

    The background window is derived from the measured object width so all
    MetaXpress-style backends share the same physical-unit interpretation.
    """

    source_array = np.asarray(image)
    if source_array.ndim != 2:
        raise ValueError(
            f"local_background_response requires a 2D image, got "
            f"shape {source_array.shape}"
        )
    if not np.isfinite(object_width_px) or object_width_px <= 0:
        raise ValueError("object_width_px must be a finite value > 0")

    working_dtype = np.result_type(source_array.dtype, np.float32)
    working_image = np.asarray(source_array, dtype=working_dtype)
    background_window = odd_size(2.0 * object_width_px + 1.0)

    if bright_objects:
        local_background = ndi.grey_opening(
            working_image,
            size=(background_window, background_window),
            mode="nearest",
        )
        return working_image - local_background

    local_background = ndi.grey_closing(
        working_image,
        size=(background_window, background_window),
        mode="nearest",
    )
    return local_background - working_image
