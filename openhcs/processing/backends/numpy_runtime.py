"""NumPy runtime capabilities used by compiled processing backends."""

from __future__ import annotations

import ctypes

from numpy.core import _multiarray_umath

_NUMPY_UMATH_GLOBAL = ctypes.CDLL(
    _multiarray_umath.__file__,
    mode=ctypes.RTLD_GLOBAL,
)


def numpy_avx512_skx_svml_symbol_available(symbol: str) -> bool:
    """Return whether one NumPy AVX-512-SKX SVML symbol is executable."""

    cpu_features = _multiarray_umath.__cpu_features__
    return (
        "AVX512_SKX" in cpu_features
        and bool(cpu_features["AVX512_SKX"])
        and hasattr(
            _NUMPY_UMATH_GLOBAL,
            symbol,
        )
    )
