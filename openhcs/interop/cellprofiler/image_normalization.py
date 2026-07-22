"""CellProfiler image intensity normalization semantics."""

from __future__ import annotations

from typing import Any

import numpy as np

from openhcs.core.runtime_image_values import (
    normalize_image_payload_intensity,
)


def normalize_cellprofiler_image_payload(
    payload: Any,
    *,
    dtype: Any = np.float32,
    channel_index: int = 0,
) -> Any:
    """Return payload in native CellProfiler's float image intensity domain."""
    return normalize_image_payload_intensity(
        payload,
        dtype=np.dtype(dtype),
        channel_index=channel_index,
    )
