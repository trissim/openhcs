"""Shared mapping lookup primitives for CellProfiler runtime policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar


MappingLookupKeyT = TypeVar("MappingLookupKeyT")
MappingLookupValueT = TypeVar("MappingLookupValueT")


@dataclass(frozen=True, slots=True)
class MappingValueLookup(Generic[MappingLookupKeyT, MappingLookupValueT]):
    """Explicit cache/default lookup for runtime mappings."""

    mapping: Mapping[MappingLookupKeyT, MappingLookupValueT]
    key: MappingLookupKeyT

    def value_or(self, fallback: MappingLookupValueT) -> MappingLookupValueT:
        if self.key in self.mapping:
            return self.mapping[self.key]
        return fallback
