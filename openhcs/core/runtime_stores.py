"""Runtime stores for typed artifact values."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from openhcs.core.artifacts import ArtifactKey, ArtifactKind
from openhcs.core.runtime_values import RuntimeValue


_UNSET = object()


@dataclass(frozen=True, slots=True)
class StoredRuntimeValue:
    """A validated runtime value with its persistence boundary."""

    value: RuntimeValue
    path: str
    backend: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("StoredRuntimeValue.path cannot be empty.")
        if not self.backend:
            raise ValueError("StoredRuntimeValue.backend cannot be empty.")

    @property
    def key(self) -> ArtifactKey:
        return self.value.key


class RuntimeValueStore:
    """Source of truth for validated runtime artifact values in one context."""

    def __init__(self) -> None:
        self._records_by_key: OrderedDict[ArtifactKey, StoredRuntimeValue] = (
            OrderedDict()
        )

    def record(
        self,
        value: RuntimeValue,
        *,
        path: str,
        backend: str,
    ) -> StoredRuntimeValue:
        """Record a validated value and its persistence location."""
        record = StoredRuntimeValue(value=value, path=path, backend=backend)
        existing = self._records_by_key.get(value.key)
        if existing is not None:
            _validate_overwrite(existing, record)
        self._records_by_key[value.key] = record
        return record

    def get(self, key: ArtifactKey) -> StoredRuntimeValue:
        """Return one stored value by exact typed artifact key."""
        try:
            return self._records_by_key[key]
        except KeyError as exc:
            raise KeyError(f"Runtime artifact key not found: {key!r}") from exc

    def find(
        self,
        *,
        name: str | None = None,
        kind: ArtifactKind | None = None,
        axis_id: str | None = None,
        group_key: Any = _UNSET,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find stored values by semantic identity fields."""
        records: list[StoredRuntimeValue] = []
        for record in self._records_by_key.values():
            key = record.key
            if name is not None and key.name != name:
                continue
            if kind is not None and key.kind is not kind:
                continue
            if axis_id is not None and key.scope.axis_id != axis_id:
                continue
            if group_key is not _UNSET and key.scope.group_key != group_key:
                continue
            records.append(record)
        return tuple(records)

    def find_by_location(
        self,
        *,
        path: str,
        backend: str,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find stored values persisted at a VFS location."""
        return tuple(
            record
            for record in self._records_by_key.values()
            if record.path == path and record.backend == backend
        )

    def keys(self) -> tuple[ArtifactKey, ...]:
        """Return stored keys in insertion order."""
        return tuple(self._records_by_key.keys())

    def values(self) -> tuple[StoredRuntimeValue, ...]:
        """Return stored records in insertion order."""
        return tuple(self._records_by_key.values())

    def __len__(self) -> int:
        return len(self._records_by_key)


def _validate_overwrite(
    existing: StoredRuntimeValue,
    incoming: StoredRuntimeValue,
) -> None:
    if existing.backend != incoming.backend:
        raise ValueError(
            f"Runtime artifact '{incoming.key.name}' already exists in backend "
            f"'{existing.backend}', cannot overwrite from '{incoming.backend}'."
        )
    if existing.path != incoming.path:
        raise ValueError(
            f"Runtime artifact '{incoming.key.name}' already exists at "
            f"'{existing.path}', cannot overwrite at '{incoming.path}'."
        )
