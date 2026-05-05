"""Opt-in capture of real runtime arrays for optimization microbenchmarks."""

from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Any

import numpy as np


_CAPTURE_DIR_ENV = "OPENHCS_CAPTURE_CELLPROFILER_FIXTURES_DIR"
_CAPTURE_LIMIT_ENV = "OPENHCS_CAPTURE_CELLPROFILER_FIXTURES_LIMIT"
_DEFAULT_LIMIT = 16
_capture_counts: dict[str, int] = {}


def capture_enabled() -> bool:
    """Return whether runtime fixture capture is enabled for this process."""
    return bool(os.environ.get(_CAPTURE_DIR_ENV))


def capture_array_fixture(name: str, **arrays: Any) -> None:
    """Persist one real execution fixture for later local microbenchmarks."""
    root_text = os.environ.get(_CAPTURE_DIR_ENV)
    if not root_text:
        return

    limit = int(os.environ.get(_CAPTURE_LIMIT_ENV, str(_DEFAULT_LIMIT)))
    count = _capture_counts.get(name, 0)
    if count >= limit:
        return
    _capture_counts[name] = count + 1

    root = Path(root_text)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        key: np.asarray(value)
        for key, value in arrays.items()
        if value is not None
    }
    if not payload:
        return
    path = root / f"{name}_{count:03d}_{time.perf_counter_ns()}.npz"
    np.savez_compressed(path, **payload)
