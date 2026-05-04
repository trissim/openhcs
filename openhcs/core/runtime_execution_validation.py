"""Validation primitives for runtime artifact execution state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
    runtime_export_failures,
)
from openhcs.core.runtime_stores import StoredRuntimeValue, require_runtime_value_store


@dataclass(frozen=True, slots=True)
class RuntimeArtifactExecutionExpectation:
    """Runtime artifacts and file exports expected from one execution."""

    artifact_kinds: frozenset[ArtifactKind]
    exports: RuntimeExportExpectation

    @classmethod
    def from_output_specs(
        cls,
        output_specs: Iterable[ArtifactSpec],
        *,
        exports: RuntimeExportExpectation,
    ) -> "RuntimeArtifactExecutionExpectation":
        return cls(
            artifact_kinds=frozenset(spec.kind for spec in output_specs),
            exports=exports,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_kinds",
            frozenset(
                kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)
                for kind in self.artifact_kinds
            ),
        )
        if not isinstance(self.exports, RuntimeExportExpectation):
            raise TypeError(
                "RuntimeArtifactExecutionExpectation.exports must be "
                f"RuntimeExportExpectation, got {type(self.exports).__name__}."
            )


@dataclass(frozen=True, slots=True)
class RuntimeArtifactExecutionObservation:
    """Observed runtime artifacts and file exports from one execution."""

    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    exports: RuntimeExportObservation

    @classmethod
    def from_contexts(
        cls,
        execution_contexts: Mapping[object, object],
        output_root: Path | None = None,
    ) -> "RuntimeArtifactExecutionObservation":
        return cls(
            records_by_axis=runtime_records_by_axis(execution_contexts),
            exports=RuntimeExportObservation.from_output_roots(
                _runtime_output_roots(execution_contexts, output_root),
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records_by_axis",
            MappingProxyType(
                {
                    str(axis): tuple(records)
                    for axis, records in self.records_by_axis.items()
                }
            ),
        )
        if not isinstance(self.exports, RuntimeExportObservation):
            raise TypeError(
                "RuntimeArtifactExecutionObservation.exports must be "
                f"RuntimeExportObservation, got {type(self.exports).__name__}."
            )

    @property
    def record_counts_by_axis(self) -> Mapping[str, Mapping[ArtifactKind, int]]:
        return MappingProxyType(
            {
                axis: MappingProxyType(Counter(record.key.kind for record in records))
                for axis, records in self.records_by_axis.items()
            }
        )


def runtime_records_by_axis(
    execution_contexts: Mapping[object, object],
) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
    """Return stored runtime records from compiled execution contexts."""
    records_by_axis: dict[str, tuple[StoredRuntimeValue, ...]] = {}
    for axis_id, context in execution_contexts.items():
        store = require_runtime_value_store(
            context,
            owner_name=f"compiled context {axis_id!r}",
        )
        records_by_axis[str(axis_id)] = tuple(store.values())
    return MappingProxyType(records_by_axis)


def runtime_output_roots(
    execution_contexts: Mapping[object, object],
    explicit_output_root: Path | None = None,
) -> tuple[Path, ...]:
    """Return authoritative runtime output roots for compiled contexts."""
    return _runtime_output_roots(execution_contexts, explicit_output_root)


def _runtime_output_roots(
    execution_contexts: Mapping[object, object],
    explicit_output_root: Path | None,
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for context in execution_contexts.values():
        step_plans = getattr(context, "step_plans", None)
        if not step_plans:
            continue
        for plan in step_plans.values():
            output_plate_root = getattr(plan, "output_plate_root", None)
            if output_plate_root is not None:
                roots.append(Path(output_plate_root))
            materialized_output = getattr(plan, "materialized_output", None)
            materialized_plate_root = getattr(
                materialized_output,
                "plate_root",
                None,
            )
            if materialized_plate_root is not None:
                roots.append(Path(materialized_plate_root))

    if roots:
        return tuple(dict.fromkeys(roots))
    if explicit_output_root is None:
        return ()
    return (Path(explicit_output_root),)


def runtime_artifact_execution_failures(
    expectation: RuntimeArtifactExecutionExpectation,
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    """Return validation failures for runtime artifacts and file exports."""
    return (
        *_runtime_artifact_failures(expectation, observation),
        *runtime_export_failures(
            expectation.exports,
            observation.exports,
            observation.records_by_axis,
        ),
    )


def _runtime_artifact_failures(
    expectation: RuntimeArtifactExecutionExpectation,
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    failures: list[str] = []
    for axis_id, counts in observation.record_counts_by_axis.items():
        for kind in sorted(
            expectation.artifact_kinds,
            key=lambda artifact_kind: artifact_kind.value,
        ):
            if counts.get(kind, 0) == 0:
                failures.append(
                    f"axis {axis_id!r} produced no runtime records for "
                    f"declared artifact kind {kind.value!r}"
                )
    return tuple(failures)
