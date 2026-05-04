"""Runtime output snapshot construction for equivalence checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.core.equivalence.images import RuntimeImageSnapshot
from openhcs.core.equivalence.tables import RuntimeTableSnapshot
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import (
    RuntimeExportObservation,
    RuntimeImageExportSpec,
    image_runtime_records,
)


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
            tables=tuple(
                RuntimeTableSnapshot.from_csv(path)
                for path in observation.table_outputs
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
        *,
        image_artifact_names: frozenset[str] = frozenset(),
        image_export_specs: tuple[RuntimeImageExportSpec, ...] = (),
    ) -> "RuntimeOutputSnapshot":
        """Build a snapshot from files owned by observed runtime artifacts."""
        file_snapshot = cls.from_export_observation(
            observation.exports.with_runtime_artifact_tables(
                observation.records_by_axis
            )
        )
        image_specs = _image_export_specs(
            image_artifact_names=image_artifact_names,
            image_export_specs=image_export_specs,
        )
        if not image_specs:
            return cls(tables=file_snapshot.tables)
        return cls(
            tables=file_snapshot.tables,
            images=_image_snapshots_from_artifact_execution(
                observation,
                image_export_specs=image_specs,
            ),
        )

    @classmethod
    def from_output_root(cls, output_root: Path) -> "RuntimeOutputSnapshot":
        """Build a semantic output snapshot from an output directory."""
        root = Path(output_root)
        if not root.exists():
            raise FileNotFoundError(f"Runtime output root does not exist: {root}")
        return cls(
            tables=tuple(
                RuntimeTableSnapshot.from_csv(path) for path in table_paths(root)
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


def image_paths(output_root: Path) -> tuple[Path, ...]:
    """Return image output paths under an output root."""
    root = Path(output_root)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_image_path(path)
    )


def _image_snapshots_from_artifact_execution(
    observation: RuntimeArtifactExecutionObservation,
    *,
    image_export_specs: tuple[RuntimeImageExportSpec, ...],
) -> tuple[RuntimeImageSnapshot, ...]:
    return tuple(
        RuntimeImageSnapshot.from_array(
            f"{record.key.scope.axis_id}_{export_spec.artifact_name}",
            export_spec.prepare_payload(record.value.data),
        )
        for axis_id in sorted(observation.records_by_axis)
        for export_spec in image_export_specs
        for record in image_runtime_records(
            observation.records_by_axis[axis_id],
            artifact_names=frozenset((export_spec.artifact_name,)),
        )
    )


def _image_export_specs(
    *,
    image_artifact_names: frozenset[str],
    image_export_specs: tuple[RuntimeImageExportSpec, ...],
) -> tuple[RuntimeImageExportSpec, ...]:
    if image_export_specs:
        return image_export_specs
    return tuple(
        RuntimeImageExportSpec(artifact_name)
        for artifact_name in sorted(image_artifact_names)
    )


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".npy",
        ".png",
        ".tif",
        ".tiff",
    }
