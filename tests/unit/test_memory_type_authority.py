"""Cross-package authority proofs for the memory-type taxonomy."""

import pickle

from openhcs.constants import (
    CPU_MEMORY_TYPES,
    GPU_MEMORY_TYPES,
    SUPPORTED_MEMORY_TYPES,
    VALID_GPU_MEMORY_TYPES,
    VALID_MEMORY_TYPES,
    MemoryType,
)
from openhcs.constants.constants import MemoryType as RawConstantsMemoryType
from openhcs.core.memory import MemoryType as RuntimeMemoryType
from openhcs.core.pipeline import MemoryType as PipelineMemoryType

from arraybridge import MemoryType as ArrayBridgeMemoryType
from arraybridge.types import (
    CPU_MEMORY_TYPES as ARRAYBRIDGE_CPU_MEMORY_TYPES,
    GPU_MEMORY_TYPES as ARRAYBRIDGE_GPU_MEMORY_TYPES,
    SUPPORTED_MEMORY_TYPES as ARRAYBRIDGE_SUPPORTED_MEMORY_TYPES,
    VALID_GPU_MEMORY_TYPES as ARRAYBRIDGE_VALID_GPU_MEMORY_TYPES,
    VALID_MEMORY_TYPES as ARRAYBRIDGE_VALID_MEMORY_TYPES,
)


def test_memory_type_taxonomy_has_one_cross_package_identity() -> None:
    assert MemoryType is ArrayBridgeMemoryType
    assert RawConstantsMemoryType is ArrayBridgeMemoryType
    assert RuntimeMemoryType is ArrayBridgeMemoryType
    assert PipelineMemoryType is ArrayBridgeMemoryType


def test_memory_type_sets_are_owner_reexports() -> None:
    assert CPU_MEMORY_TYPES is ARRAYBRIDGE_CPU_MEMORY_TYPES
    assert GPU_MEMORY_TYPES is ARRAYBRIDGE_GPU_MEMORY_TYPES
    assert SUPPORTED_MEMORY_TYPES is ARRAYBRIDGE_SUPPORTED_MEMORY_TYPES
    assert VALID_MEMORY_TYPES is ARRAYBRIDGE_VALID_MEMORY_TYPES
    assert VALID_GPU_MEMORY_TYPES is ARRAYBRIDGE_VALID_GPU_MEMORY_TYPES


def test_legacy_openhcs_memory_type_pickle_resolves_to_owner_identity() -> None:
    legacy_payload = (
        b"copenhcs.constants.constants\nMemoryType\np0\n"
        b"(Vnumpy\np1\ntp2\nRp3\n."
    )

    assert pickle.loads(legacy_payload) is ArrayBridgeMemoryType.NUMPY
