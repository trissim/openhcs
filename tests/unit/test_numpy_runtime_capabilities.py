"""NumPy native-runtime capability tests."""

from __future__ import annotations

from numpy.core import _multiarray_umath

from openhcs.processing.backends.numpy_runtime import (
    numpy_avx512_skx_svml_symbol_available,
)


def test_svml_symbol_requires_runtime_avx512_skx(monkeypatch) -> None:
    """AVX-512F alone cannot execute NumPy's AVX-512-SKX SVML symbols."""

    monkeypatch.setitem(_multiarray_umath.__cpu_features__, "AVX512F", True)
    monkeypatch.setitem(_multiarray_umath.__cpu_features__, "AVX512_SKX", False)

    assert numpy_avx512_skx_svml_symbol_available("__svml_pow8") is False


def test_svml_symbol_rejects_non_x86_feature_map(monkeypatch) -> None:
    """A NumPy build without x86 features cannot expose executable AVX-512."""

    monkeypatch.delitem(
        _multiarray_umath.__cpu_features__,
        "AVX512_SKX",
        raising=False,
    )

    assert numpy_avx512_skx_svml_symbol_available("__svml_pow8") is False


def test_svml_capability_rejects_missing_symbol(monkeypatch) -> None:
    """CPU capability alone cannot make an absent NumPy symbol available."""

    monkeypatch.setitem(_multiarray_umath.__cpu_features__, "AVX512_SKX", True)

    assert numpy_avx512_skx_svml_symbol_available("__openhcs_missing_svml") is False
