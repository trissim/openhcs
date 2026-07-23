"""Runtime output snapshot construction for equivalence checks."""

from __future__ import annotations
from openhcs.core.runtime_measurements import MeasurementRowAxisField

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from os.path import commonprefix
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.equivalence.images import RuntimeImageSnapshot
from openhcs.core.equivalence.policy import normalize_runtime_identifier
from openhcs.core.equivalence.tables import RuntimeTableSnapshot
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import RuntimeExportObservation


@dataclass(frozen=True, slots=True)
class RuntimeOutputSnapshot:
    """Semantic snapshot of runtime file outputs."""

    tables: tuple[RuntimeTableSnapshot, ...] = ()
    images: tuple[RuntimeImageSnapshot, ...] = ()

    @classmethod
    def from_export_observation(
        cls,
        observation: RuntimeExportObservation,
    ) -> "RuntimeOutputSnapshot":
        """Build a semantic output snapshot from observed runtime exports."""
        return cls(
            tables=RuntimeTableNamespaceAdapter.normalize(
                tuple(
                    RuntimeTableSnapshot.from_csv(path)
                    for path in observation.table_outputs
                )
            ),
            images=tuple(
                RuntimeImageSnapshot.from_image_file(path)
                for path in observation.image_outputs
            ),
        )

    @classmethod
    def from_artifact_execution_observation(
        cls,
        observation: RuntimeArtifactExecutionObservation,
    ) -> "RuntimeOutputSnapshot":
        """Build a snapshot from files owned by observed runtime artifacts."""
        return cls.from_export_observation(
            observation.exports.with_runtime_artifact_tables(
                observation.records_by_axis
            )
        )

    @classmethod
    def from_output_root(cls, output_root: Path) -> "RuntimeOutputSnapshot":
        """Build a semantic output snapshot from an output directory."""
        root = Path(output_root)
        if not root.exists():
            raise FileNotFoundError(f"Runtime output root does not exist: {root}")
        return cls(
            tables=RuntimeTableNamespaceAdapter.normalize(
                tuple(RuntimeTableSnapshot.from_csv(path) for path in table_paths(root))
            ),
            images=tuple(
                RuntimeImageSnapshot.from_image_file(path)
                for path in image_paths(root)
            ),
        )


def table_paths(output_root: Path) -> tuple[Path, ...]:
    """Return non-empty CSV output paths under an output root."""
    root = Path(output_root)
    return tuple(
        path
        for path in sorted(root.rglob("*.csv"))
        if path.is_file() and path.stat().st_size > 0
    )


class RuntimeTableNamespaceAdapter(ABC, metaclass=AutoRegisterMeta):
    """Normalize file-export table namespace without changing table contents."""

    __registry_key__ = "namespace_adapter"
    __registry__: ClassVar[dict[str, type["RuntimeTableNamespaceAdapter"]]] = {}
    namespace_adapter: ClassVar[str | None] = None

    @classmethod
    def normalize(
        cls,
        tables: tuple[RuntimeTableSnapshot, ...],
    ) -> tuple[RuntimeTableSnapshot, ...]:
        adapters = tuple(
            adapter_type()
            for adapter_type in cls.__registry__.values()
            if adapter_type().supports(tables)
        )
        if not adapters:
            return tables
        if len(adapters) > 1:
            names = tuple(type(adapter).__name__ for adapter in adapters)
            raise ValueError(
                "Ambiguous runtime table namespace adapters for exported tables: "
                f"{names!r}."
            )
        return adapters[0].normalize_tables(tables)

    @abstractmethod
    def supports(self, tables: tuple[RuntimeTableSnapshot, ...]) -> bool:
        """Return whether this adapter owns the table namespace."""

    @abstractmethod
    def normalize_tables(
        self,
        tables: tuple[RuntimeTableSnapshot, ...],
    ) -> tuple[RuntimeTableSnapshot, ...]:
        """Return semantic table snapshots with normalized path identities."""


class CommonStemRuntimeTableNamespaceAdapter(RuntimeTableNamespaceAdapter):
    """Remove an exporter-wide filename namespace shared by all table outputs."""

    namespace_adapter = "common_stem"

    def supports(self, tables: tuple[RuntimeTableSnapshot, ...]) -> bool:
        return _common_table_namespace_prefix(tables) is not None

    def normalize_tables(
        self,
        tables: tuple[RuntimeTableSnapshot, ...],
    ) -> tuple[RuntimeTableSnapshot, ...]:
        prefix = _common_table_namespace_prefix(tables)
        if prefix is None:
            return tables
        return tuple(
            replace(
                table,
                path=table.path.with_name(f"{table.path.stem[len(prefix):]}{table.path.suffix}"),
            )
            for table in tables
        )


def _common_table_namespace_prefix(
    tables: tuple[RuntimeTableSnapshot, ...],
) -> str | None:
    if len(tables) < 2:
        return None
    stems = tuple(table.path.stem for table in tables)
    shared = commonprefix(stems)
    if "_" not in shared:
        return None
    prefix = shared[: shared.rfind("_") + 1]
    suffixes = tuple(stem[len(prefix):] for stem in stems)
    if not prefix or any(not suffix for suffix in suffixes):
        return None
    if all(_table_has_object_identity(table) for table in tables):
        return None
    return prefix


def _table_has_object_identity(table: RuntimeTableSnapshot) -> bool:
    normalized_header = {
        _normalize_table_header_field(field)
        for field in table.header
    }
    return bool(normalized_header & set(MeasurementRowAxisField.object_id_field_names()))


def _normalize_table_header_field(field: str) -> str:
    return normalize_runtime_identifier(field)


def image_paths(output_root: Path) -> tuple[Path, ...]:
    """Return image output paths under an output root."""
    root = Path(output_root)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_image_path(path)
    )


def _is_image_path(path: Path) -> bool:
    return ImageFileFormat.is_image_path(path)
