"""
Image processors for different backends.

This module contains image processing functions implemented for different
computational backends (numpy, cupy, torch, tensorflow, jax).

Each processor module provides the same set of functions but optimized
for its specific backend.
"""

from __future__ import annotations

import importlib

__all__ = [
    'numpy_processor',
    'cupy_processor',
    'torch_processor',
    'tensorflow_processor',
    'jax_processor',
    'pyclesperanto_processor'
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
