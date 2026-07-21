"""Execution-local cache for already materialized runtime image stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from openhcs.core.runtime_array_values import RuntimeArrayData


@dataclass(frozen=True, slots=True)
class RuntimeImageStackCacheKey:
    """Identity for a stack loaded from an ordered runtime path set."""

    paths: tuple[str, ...]
    memory_type: str

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str, ...],
        *,
        memory_type: str,
    ) -> "RuntimeImageStackCacheKey":
        if not paths:
            raise ValueError("Runtime image stack cache paths cannot be empty.")
        if not memory_type:
            raise ValueError("Runtime image stack cache memory_type cannot be empty.")
        return cls(tuple(str(path) for path in paths), str(memory_type))


@dataclass(frozen=True, slots=True)
class RuntimeImageStackCacheValue:
    """Cached payload for an ordered runtime-slice path set."""

    stack: RuntimeArrayData


@dataclass(slots=True)
class RuntimeImageStackCache:
    """Processing-context-local cache for adjacent step stack reuse."""

    stacks: dict[RuntimeImageStackCacheKey, RuntimeImageStackCacheValue] = field(
        default_factory=dict
    )

    def get(
        self,
        paths: tuple[str, ...],
        *,
        memory_type: str,
    ) -> RuntimeImageStackCacheValue | None:
        """Return a cached stack for the exact ordered path and memory contract."""
        return self.stacks.get(
            RuntimeImageStackCacheKey.from_paths(paths, memory_type=memory_type)
        )

    def store(
        self,
        paths: tuple[str, ...],
        *,
        memory_type: str,
        stack: RuntimeArrayData,
    ) -> None:
        """Store a stack produced for a just-saved ordered path set."""
        self.stacks[
            RuntimeImageStackCacheKey.from_paths(paths, memory_type=memory_type)
        ] = RuntimeImageStackCacheValue(stack=stack)

    def discard_paths(self, paths: tuple[str, ...]) -> None:
        """Discard cached stacks that include any of the supplied paths."""
        path_set = frozenset(str(path) for path in paths)
        if not path_set:
            return
        for key in tuple(self.stacks):
            if path_set.intersection(key.paths):
                del self.stacks[key]
