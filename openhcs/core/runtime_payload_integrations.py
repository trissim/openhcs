"""Default external payload registrations for runtime artifact validation."""

from __future__ import annotations

from functools import cache

from openhcs.core.runtime_values import (
    register_array_payload_type,
    register_columnar_rows_type,
)


@cache
def register_runtime_payload_integrations() -> None:
    """Register installed external payload classes with runtime ABCs."""
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        register_array_payload_type(np.ndarray)

    try:
        import cupy as cp
    except ImportError:
        pass
    else:
        register_array_payload_type(cp.ndarray)

    try:
        import torch
    except ImportError:
        pass
    else:
        register_array_payload_type(torch.Tensor)

    try:
        import pandas as pd
    except ImportError:
        pass
    else:
        register_columnar_rows_type(pd.DataFrame)
