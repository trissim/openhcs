"""Runtime stores for typed artifact values."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhcs.core.artifacts import ArtifactKey, ArtifactKind
from openhcs.core.runtime_values import RuntimeValue


def require_runtime_value_store(
    owner: object,
    *,
    owner_name: str,
) -> "RuntimeValueStore":
    """Return the runtime value store attached to an execution owner."""
    store = getattr(owner, "runtime_value_store", None)
    if store is None:
        raise RuntimeError(f"{owner_name}.runtime_value_store is required.")
    if not isinstance(store, RuntimeValueStore):
        raise TypeError(
            f"{owner_name}.runtime_value_store must be RuntimeValueStore, "
            f"got {type(store).__name__}."
        )
    return store


@dataclass(frozen=True, slots=True)
class RuntimeArtifactLocation:
    """VFS location for one persisted runtime artifact payload."""

    path: str
    backend: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("RuntimeArtifactLocation.path cannot be empty.")
        if not self.backend:
            raise ValueError("RuntimeArtifactLocation.backend cannot be empty.")


@dataclass(frozen=True, slots=True)
class RuntimeStoreObservationCursor:
    """Cursor into the append-only runtime artifact observation stream."""

    index: int
    revision: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("RuntimeStoreObservationCursor.index cannot be negative.")
        if self.revision < 0:
            raise ValueError(
                "RuntimeStoreObservationCursor.revision cannot be negative."
            )


class RuntimeArtifactQueryTarget:
    """Nominal runtime-artifact address matched after semantic key fields."""

    def matches(self, record: "StoredRuntimeValue") -> bool:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RuntimeArtifactLocationTarget(RuntimeArtifactQueryTarget):
    """Runtime-artifact query target for one persisted VFS location."""

    location: RuntimeArtifactLocation

    def matches(self, record: "StoredRuntimeValue") -> bool:
        return record.location == self.location


@dataclass(frozen=True, slots=True)
class RuntimeArtifactGroupTarget(RuntimeArtifactQueryTarget):
    """Runtime-artifact query target for one exact execution group."""

    group_key: str | None

    def matches(self, record: "StoredRuntimeValue") -> bool:
        return record.key.scope.group_key == self.group_key


def replace_runtime_artifact_payload(
    filemanager: Any,
    data: Any,
    location: RuntimeArtifactLocation,
) -> None:
    """Persist the current payload for a latest-binding runtime artifact."""
    filemanager.ensure_directory(str(Path(location.path).parent), location.backend)
    if filemanager.exists(location.path, location.backend):
        filemanager.delete(location.path, location.backend)
    filemanager.save(data, location.path, location.backend)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactQuery:
    """Typed lookup for one planned runtime artifact record."""

    name: str
    kind: ArtifactKind
    axis_id: str
    target: RuntimeArtifactQueryTarget

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RuntimeArtifactQuery.name cannot be empty.")
        if not self.axis_id:
            raise ValueError("RuntimeArtifactQuery.axis_id cannot be empty.")
        if not isinstance(self.target, RuntimeArtifactQueryTarget):
            raise TypeError(
                "RuntimeArtifactQuery.target must be RuntimeArtifactQueryTarget, "
                f"got {type(self.target).__name__}."
            )

    def matches(self, record: "StoredRuntimeValue") -> bool:
        key = record.key
        if key.name != self.name:
            return False
        if key.kind is not self.kind:
            return False
        if key.scope.axis_id != self.axis_id:
            return False
        if not self.target.matches(record):
            return False
        return True


@dataclass(frozen=True, slots=True)
class StoredRuntimeValue:
    """A validated runtime value with its persistence boundary."""

    value: RuntimeValue
    location: RuntimeArtifactLocation

    @property
    def key(self) -> ArtifactKey:
        return self.value.key

    @property
    def path(self) -> str:
        return self.location.path

    @property
    def backend(self) -> str:
        return self.location.backend


class RuntimeValueStore:
    """Source of truth for validated runtime artifact values in one context."""

    def __init__(self) -> None:
        self._records_by_location: OrderedDict[
            tuple[ArtifactKey, RuntimeArtifactLocation],
            StoredRuntimeValue,
        ] = OrderedDict()
        self._observation_records: list[StoredRuntimeValue] = []
        self._current_location_by_key: dict[ArtifactKey, RuntimeArtifactLocation] = {}
        self._revision = 0
        self._find_cache: dict[
            tuple[
                int,
                str | None,
                ArtifactKind | None,
                str | None,
                str | None,
                bool,
            ],
            tuple[StoredRuntimeValue, ...],
        ] = {}

    @property
    def revision(self) -> int:
        """Return the mutation revision for cache-safe runtime queries."""
        return self._revision

    def record(
        self,
        value: RuntimeValue,
        *,
        path: str,
        backend: str,
    ) -> StoredRuntimeValue:
        """Record a validated value and its persistence location."""
        record = StoredRuntimeValue(
            value=value,
            location=RuntimeArtifactLocation(path=path, backend=backend),
        )
        existing = self._current_record(value.key)
        if existing is not None:
            _validate_overwrite(existing, record)
        self._records_by_location[(value.key, record.location)] = record
        self._observation_records.append(record)
        self._current_location_by_key[value.key] = record.location
        self._mark_mutated()
        return record

    def replace(
        self,
        value: RuntimeValue,
        *,
        path: str,
        backend: str,
    ) -> StoredRuntimeValue:
        """Replace the current binding for a typed artifact key.

        Path planning treats repeated producers for the same artifact name as a
        new workspace binding. This method makes that replacement explicit while
        keeping record() strict for accidental duplicate writes.
        """
        record = StoredRuntimeValue(
            value=value,
            location=RuntimeArtifactLocation(path=path, backend=backend),
        )
        self._records_by_location[(value.key, record.location)] = record
        self._observation_records.append(record)
        self._current_location_by_key[value.key] = record.location
        self._mark_mutated()
        return record

    def resolve(
        self,
        query: RuntimeArtifactQuery,
        *,
        purpose: str,
    ) -> StoredRuntimeValue:
        """Resolve exactly one runtime artifact record for a planned operation."""
        records = self.find_matching(query)
        if not records:
            raise RuntimeError(
                f"Missing RuntimeValueStore record for {purpose} "
                f"'{query.name}' ({query.kind.value}) on axis '{query.axis_id}'."
            )
        if len(records) > 1:
            raise RuntimeError(
                f"Ambiguous RuntimeValueStore records for {purpose} "
                f"'{query.name}' ({query.kind.value}) on axis '{query.axis_id}': "
                f"{records!r}."
            )
        return records[0]

    def find_matching(
        self,
        query: RuntimeArtifactQuery,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return stored records matched by a typed runtime artifact query."""
        return tuple(
            record
            for record in self._records_by_location.values()
            if query.matches(record)
        )

    def get(self, key: ArtifactKey) -> StoredRuntimeValue:
        """Return one stored value by exact typed artifact key."""
        record = self._current_record(key)
        if record is None:
            raise KeyError(f"Runtime artifact key not found: {key!r}")
        return record

    def find(
        self,
        *,
        name: str | None = None,
        kind: ArtifactKind | None = None,
        axis_id: str | None = None,
        group_key: str | None = None,
        match_group: bool = False,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find stored values by semantic identity fields."""
        cache_key = (self._revision, name, kind, axis_id, group_key, match_group)
        cached = self._find_cache.get(cache_key)
        if cached is not None:
            return cached
        records: list[StoredRuntimeValue] = []
        for record in self._records_by_location.values():
            key = record.key
            if name is not None and key.name != name:
                continue
            if kind is not None and key.kind is not kind:
                continue
            if axis_id is not None and key.scope.axis_id != axis_id:
                continue
            if match_group and key.scope.group_key != group_key:
                continue
            records.append(record)
        result = tuple(records)
        self._find_cache[cache_key] = result
        return result

    def find_by_location(
        self,
        *,
        path: str,
        backend: str,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find stored values persisted at a VFS location."""
        location = RuntimeArtifactLocation(path=path, backend=backend)
        return tuple(
            record
            for record in self._records_by_location.values()
            if record.location == location
        )

    def keys(self) -> tuple[ArtifactKey, ...]:
        """Return stored keys in insertion order."""
        return tuple(record.key for record in self._records_by_location.values())

    def values(self) -> tuple[StoredRuntimeValue, ...]:
        """Return stored records in insertion order."""
        return tuple(self._records_by_location.values())

    @property
    def observed_values(self) -> tuple[StoredRuntimeValue, ...]:
        """Return every runtime artifact write in insertion order."""
        return tuple(self._observation_records)

    def observation_cursor(self) -> RuntimeStoreObservationCursor:
        """Return a cursor for future observation-delta queries."""
        return RuntimeStoreObservationCursor(
            index=len(self._observation_records),
            revision=self._revision,
        )

    def observed_values_after(
        self,
        cursor: RuntimeStoreObservationCursor,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return runtime artifact writes recorded after ``cursor``."""
        if cursor.index > len(self._observation_records):
            raise ValueError(
                "RuntimeStoreObservationCursor.index is beyond the current "
                f"observation stream length: {cursor.index} > "
                f"{len(self._observation_records)}."
            )
        return tuple(self._observation_records[cursor.index :])

    def clear(self) -> None:
        """Release every runtime artifact record owned by this execution context."""
        self._records_by_location.clear()
        self._observation_records.clear()
        self._current_location_by_key.clear()
        self._mark_mutated()

    def merge_observed_values(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> None:
        """Merge observed records produced across an execution boundary."""
        if records and tuple(self._observation_records) == records:
            return
        for record in records:
            key = (record.key, record.location)
            self._records_by_location[key] = record
            self._current_location_by_key[record.key] = record.location
        if records:
            self._observation_records.extend(records)
            self._mark_mutated()

    def __len__(self) -> int:
        return len(self._records_by_location)

    def _current_record(self, key: ArtifactKey) -> StoredRuntimeValue | None:
        location = self._current_location_by_key.get(key)
        if location is None:
            return None
        return self._records_by_location.get((key, location))

    def _mark_mutated(self) -> None:
        self._revision += 1
        self._find_cache.clear()


def _validate_overwrite(
    existing: StoredRuntimeValue,
    incoming: StoredRuntimeValue,
) -> None:
    if existing.location.backend != incoming.location.backend:
        raise ValueError(
            f"Runtime artifact '{incoming.key.name}' already exists in backend "
            f"'{existing.location.backend}', cannot overwrite from "
            f"'{incoming.location.backend}'."
        )
    if existing.location.path != incoming.location.path:
        raise ValueError(
            f"Runtime artifact '{incoming.key.name}' already exists at "
            f"'{existing.location.path}', cannot overwrite at "
            f"'{incoming.location.path}'."
        )
