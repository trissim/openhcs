from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as imageio
import numpy as np

from openhcs.core.artifacts import ArtifactKey, ArtifactKind, ArtifactScope
from openhcs.core.runtime_equivalence import (
    RuntimeEquivalencePolicy,
    RuntimeOutputSnapshot,
    runtime_artifact_execution_equivalence,
    runtime_output_equivalence,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import RuntimeValue, RuntimeValueSchema


def test_runtime_output_equivalence_ignores_table_paths_and_column_order(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text("b,a\n2,1\n4,3\n", encoding="utf-8")
    (candidate_root / "axis_Measurements_step1.csv").write_text(
        "a,b\n1,2\n3,4\n",
        encoding="utf-8",
    )

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.is_equivalent


def test_runtime_output_equivalence_uses_numeric_policy_for_tables(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "values.csv").write_text(
        "measurement\n1.000000001\n",
        encoding="utf-8",
    )
    (candidate_root / "values.csv").write_text(
        "measurement\n1.000000002\n",
        encoding="utf-8",
    )

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
        policy=RuntimeEquivalencePolicy(numeric_decimal_places=8),
    )

    assert report.is_equivalent


def test_runtime_output_equivalence_detects_table_value_mismatch(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "values.csv").write_text("measurement\n1.0\n", encoding="utf-8")
    (candidate_root / "values.csv").write_text("measurement\n2.0\n", encoding="utf-8")

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.failure_messages() == (
        "table schema ('measurement',) values differ",
    )


def test_runtime_output_equivalence_compares_decoded_image_pixels(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    pixels = np.arange(9, dtype=np.uint16).reshape(3, 3)
    imageio.imwrite(reference_root / "native_name.tif", pixels)
    imageio.imwrite(candidate_root / "openhcs_name.tif", pixels.copy())

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.is_equivalent


def test_runtime_output_equivalence_detects_image_pixel_mismatch(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    imageio.imwrite(
        reference_root / "native_name.tif",
        np.zeros((3, 3), dtype=np.uint8),
    )
    imageio.imwrite(
        candidate_root / "openhcs_name.tif",
        np.ones((3, 3), dtype=np.uint8),
    )

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.failure_messages() == ("image output content differs",)


def test_runtime_execution_equivalence_detects_artifact_count_mismatch(
    tmp_path: Path,
) -> None:
    reference_store = RuntimeValueStore()
    reference_store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=(),
            schema=RuntimeValueSchema(kind=ArtifactKind.MEASUREMENTS),
        ),
        path="/memory/Measurements.pkl",
        backend="memory",
    )
    reference = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=reference_store)},
        tmp_path / "reference",
    )
    candidate = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=RuntimeValueStore())},
        tmp_path / "candidate",
    )

    report = runtime_artifact_execution_equivalence(reference, candidate)

    assert report.failure_messages() == (
        "runtime artifact counts differ: "
        "reference={<ArtifactKind.MEASUREMENTS: 'measurements'>: 1}, "
        "candidate={}",
    )
