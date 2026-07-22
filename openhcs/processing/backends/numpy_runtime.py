"""NumPy runtime capabilities used by compiled processing backends."""

from __future__ import annotations

import ctypes

from numpy.core import _multiarray_umath

_NUMPY_UMATH_GLOBAL = ctypes.CDLL(
    _multiarray_umath.__file__,
    mode=ctypes.RTLD_GLOBAL,
)


def numpy_avx512_svml_symbol_available(symbol: str) -> bool:
    """Return whether one NumPy SVML symbol is executable on this CPU."""

    cpu_features = _multiarray_umath.__cpu_features__
    return "AVX512F" in cpu_features and bool(cpu_features["AVX512F"]) and hasattr(
        _NUMPY_UMATH_GLOBAL,
        symbol,
    )
