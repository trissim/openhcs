"""Canonical array payload handling for runtime equivalence."""

from __future__ import annotations

import hashlib

import numpy as np

from openhcs.core.memory import (
    MEMORY_TYPE_NUMPY,
    _FRAMEWORK_CONFIG,
    convert_memory,
)


def array_memory_type(value: object) -> str | None:
    """Return the ArrayBridge memory type for an array payload, if known."""
    if isinstance(value, np.ndarray):
        return MEMORY_TYPE_NUMPY

    module_name = type(value).__module__
    top_level = module_name.split(".")[0]
    for memory_type, config in _FRAMEWORK_CONFIG.items():
        import_name = str(config["import_name"])
        aliases = {import_name}
        if import_name == "jax":
            aliases.add("jaxlib")
        if top_level in aliases:
            return memory_type.value
    return None


def canonical_numpy_array(value: object) -> np.ndarray | None:
    """Convert an ArrayBridge-supported array to canonical CPU NumPy."""
    source_type = array_memory_type(value)
    if source_type is None:
        return None
    array = (
        value
        if source_type == MEMORY_TYPE_NUMPY
        else convert_memory(value, source_type, MEMORY_TYPE_NUMPY, 0)
    )
    return np.ascontiguousarray(array)


def canonical_scalar(value: object) -> object:
    """Return the Python scalar for backend scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def semantic_array_payload(value: object) -> tuple[str, str, tuple[int, ...], str] | None:
    """Return a backend-independent exact payload for array content."""
    array = canonical_numpy_array(value)
    if array is None:
        return None
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return (
        "array",
        str(array.dtype),
        tuple(int(axis) for axis in array.shape),
        digest.hexdigest(),
    )


def semantic_array_shape(value: object) -> tuple[int, ...] | None:
    """Return backend-independent array shape without hashing content."""
    array = canonical_numpy_array(value)
    if array is None:
        return None
    return tuple(int(axis) for axis in array.shape)
