from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as imageio
import numpy as np

from openhcs.core.artifacts import ArtifactKey, ArtifactKind, ArtifactScope
from openhcs.core.runtime_equivalence import (
    RuntimeCellSignature,
    RuntimeCellValueKind,
    RuntimeEquivalencePolicy as _RuntimeEquivalencePolicy,
    RuntimeImageSnapshot,
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementSnapshot,
    RuntimeMeasurementSubjectKey,
    RuntimeOutputSnapshot,
    ObjectInstanceKeyPlaneAlignmentStrategy,
    runtime_artifact_execution_equivalence,
    runtime_measurement_equivalence,
    _dedupe_runtime_measurement_table_aggregate_rows,
    runtime_output_equivalence as _runtime_output_equivalence,
    runtime_reference_artifact_equivalence as _runtime_reference_artifact_equivalence,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
    cellprofiler_runtime_equivalence_policy,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
)
from openhcs.core.equivalence.tables import (
    RuntimeMeasurementRowFingerprint,
    exact_measurement_table_key,
)
from openhcs.core.runtime_exports import (
    RuntimeImageExportBitDepth,
    RuntimeImageExportSpec,
    RuntimeExportObservation,
)
from openhcs.core.runtime_semantics import (
    MeasurementStatistic,
    MeasurementScope,
    MeasurementSubject,
    ObjectInstanceKey,
    ObjectCoreMeasurementFeature,
    MeasurementObjectRowIdentity,
    ObjectMeasurementFeatureRole,
    ObjectLabelDomainScope,
    RelationshipSemantics,
    RuntimePlaneAxis,
)
from openhcs.core.equivalence.measurement_features import (
    object_measurement_feature_has_role,
    object_measurement_feature_requires_sparse_boundary_object_count_stability,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectRelationship,
    SpatialGrid,
    RuntimeValue,
    RuntimeValueSchema,
)


def RuntimeEquivalencePolicy(**kwargs):
    kwargs.setdefault("measurement_dialect", CELLPROFILER_MEASUREMENT_DIALECT)
    return _RuntimeEquivalencePolicy(**kwargs)


def runtime_output_equivalence(*args, policy=None, **kwargs):
    if policy is None:
        policy = RuntimeEquivalencePolicy()
    return _runtime_output_equivalence(*args, policy=policy, **kwargs)


def runtime_reference_artifact_equivalence(*args, policy=None, **kwargs):
    if policy is None:
        policy = RuntimeEquivalencePolicy()
    return _runtime_reference_artifact_equivalence(*args, policy=policy, **kwargs)


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


def test_runtime_output_snapshot_reads_vendor_style_csv_preamble(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "summary.csv").write_text(
        "Barcode,Plate-1,,\n"
        "Description,metadata preamble,,\n"
        "well,measurement\n"
        "A01,3\n",
        encoding="utf-8",
    )

    snapshot = RuntimeOutputSnapshot.from_output_root(root)

    assert snapshot.tables[0].header == ("well", "measurement")
    assert snapshot.tables[0].rows == (("A01", "3"),)


def test_runtime_output_snapshot_reads_contextual_duplicate_csv_header(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "objects.csv").write_text(
        "Image,Cells,Nuclei\n"
        "ImageNumber,Texture_Entropy,Texture_Entropy\n"
        "1,0.5,0.7\n",
        encoding="utf-8",
    )

    snapshot = RuntimeOutputSnapshot.from_output_root(root)

    assert snapshot.tables[0].header == (
        "ImageNumber",
        "Texture_Entropy_cells",
        "Texture_Entropy_nuclei",
    )
    assert snapshot.tables[0].rows == (("1", "0.5", "0.7"),)


def test_runtime_output_snapshot_reads_first_row_duplicate_csv_header(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "objects.csv").write_text(
        "ImageNumber,ObjectNumber,Metadata_Run,Metadata_Run,AreaShape_Area\n"
        "1,1,Sequence1,Sequence1,42\n",
        encoding="utf-8",
    )

    snapshot = RuntimeOutputSnapshot.from_output_root(root)

    assert snapshot.tables[0].header == (
        "ImageNumber",
        "ObjectNumber",
        "Metadata_Run",
        "Metadata_Run_4",
        "AreaShape_Area",
    )
    assert snapshot.tables[0].rows == (("1", "1", "Sequence1", "Sequence1", "42"),)


def test_runtime_measurement_snapshot_skips_contextual_padding_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "objects.csv").write_text(
        "Image,Cells,Cells,Nuclei\n"
        "ImageNumber,ObjectNumber,AreaShape_Area,Texture_Entropy\n"
        "1,1,10,0.7\n"
        "1,2,20,0.8\n"
        "1,3,nan,nan\n",
        encoding="utf-8",
    )

    snapshot = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(root),
        policy=RuntimeEquivalencePolicy(),
    )

    cells_area = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells"),
        "area",
    )
    nuclei_entropy = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei"),
        "entropy",
    )
    nan = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "nan")

    assert snapshot.values_by_feature[cells_area] == {
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0"): 1,
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.0"): 1,
    }
    assert snapshot.values_by_feature[nuclei_entropy] == {
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.7"): 1,
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.8"): 1,
    }
    assert nan not in snapshot.values_by_feature[cells_area]
    assert nan not in snapshot.values_by_feature[nuclei_entropy]


def test_runtime_measurement_snapshot_skips_contextual_feature_family_padding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "objects.csv").write_text(
        "Image,Cells,Cells,Cells\n"
        "ImageNumber,ObjectNumber,AreaShape_Area,Intensity_MeanIntensity\n"
        "1,1,10,0.7\n"
        "1,2,nan,0.8\n"
        "1,3,nan,nan\n",
        encoding="utf-8",
    )

    snapshot = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(root),
        policy=RuntimeEquivalencePolicy(),
    )

    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    cells_area = RuntimeMeasurementFeatureKey(subject, "area")
    cells_mean_intensity = RuntimeMeasurementFeatureKey(subject, "mean_intensity")
    nan = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "nan")

    assert snapshot.values_by_feature[cells_area] == {
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0"): 1,
    }
    assert snapshot.values_by_feature[cells_mean_intensity] == {
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.7"): 1,
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.8"): 1,
    }
    assert nan not in snapshot.values_by_feature[cells_area]
    assert nan not in snapshot.values_by_feature[cells_mean_intensity]


def test_runtime_reference_artifact_equivalence_skips_runtime_table_padding_rows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "Image,Cells,Cells,Cells\n"
        "ImageNumber,ObjectNumber,AreaShape_Area,AreaShape_Zernike_0_0\n"
        "1,1,10,0.82\n"
        "1,2,nan,nan\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "object_name": "Cells",
                "area": 10,
                "Zernike_0_0": 0.82,
            },
            {
                "object_label": 2,
                "object_name": "Cells",
                "area": np.nan,
                "Zernike_0_0": np.nan,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_skips_runtime_shape_padding_defaults(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "Image,Cells,Cells,Cells\n"
        "ImageNumber,ObjectNumber,AreaShape_Area,AreaShape_Center_X,"
        "AreaShape_Center_Z\n"
        "1,1,10,4.5,0\n",
        encoding="utf-8",
    )
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "object_name": "Cells",
                "area": 10,
                "center_x": 4.5,
                "center_z": 0.0,
            },
            {
                "object_label": 2,
                "object_name": "Cells",
                "area": np.nan,
                "center_x": np.nan,
                "center_z": 0.0,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_skips_runtime_category_padding_groups(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "Image,Cells,Cells,Cells\n"
        "ImageNumber,ObjectNumber,AreaShape_Area,Intensity_MeanIntensity\n"
        "1,1,10,0.7\n"
        "1,2,nan,0.8\n"
        "1,3,nan,nan\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MergedMeasurements",
        rows=(
            {
                "object_label": 1,
                "object_name": "Cells",
                "AreaShape_Area": 10,
                "Intensity_MeanIntensity": 0.7,
            },
            {
                "object_label": 2,
                "object_name": "Cells",
                "AreaShape_Area": np.nan,
                "Intensity_MeanIntensity": 0.8,
            },
            {
                "object_label": 3,
                "object_name": "Cells",
                "AreaShape_Area": np.nan,
                "Intensity_MeanIntensity": np.nan,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MergedMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MergedMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_output_equivalence_ignores_experiment_metadata_map(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Experiment.csv").write_text(
        "Key,Value\nCellProfiler_Version,4.2.8\n",
        encoding="utf-8",
    )

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.is_equivalent


def test_runtime_measurement_snapshot_ignores_experiment_metadata_map(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    (root / "Experiment.csv").write_text(
        "Key,Value\nCellProfiler_Version,4.2.8\n",
        encoding="utf-8",
    )

    snapshot = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(root),
        policy=RuntimeEquivalencePolicy(),
    )

    assert snapshot.is_empty


def test_runtime_reference_artifact_equivalence_projects_spatial_grid_measurements(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,DefinedGrid_Grid_Columns,DefinedGrid_Grid_Rows,"
        "DefinedGrid_Grid_XLocationOfLowestXSpot,DefinedGrid_Grid_XSpacing,"
        "DefinedGrid_Grid_YLocationOfLowestYSpot,DefinedGrid_Grid_YSpacing\n"
        "1,12,8,71,102.5,57,103.25\n",
        encoding="utf-8",
    )
    grid = SpatialGrid(
        name="Grid",
        rows=8,
        columns=12,
        x_spacing=102.5,
        y_spacing=103.25,
        x_origin=71,
        y_origin=57,
    )
    value = RuntimeValue(
        key=ArtifactKey(
            name="Grid",
            kind=ArtifactKind.SPATIAL_GRID,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=grid.runtime_payload(),
        schema=grid.runtime_schema(grid.runtime_payload()),
    )
    store = RuntimeValueStore()
    store.record(value, path="/memory/Grid.pkl", backend="memory")
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_projects_slice_aligned_spatial_grid_measurements(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,DefinedGrid_Grid_Columns,DefinedGrid_Grid_Rows,"
        "DefinedGrid_Grid_XLocationOfLowestXSpot,DefinedGrid_Grid_XSpacing,"
        "DefinedGrid_Grid_YLocationOfLowestYSpot,DefinedGrid_Grid_YSpacing\n"
        "1,12,8,71,102.5,57,103.25\n"
        "2,12,8,72,102.5,57,103.25\n",
        encoding="utf-8",
    )
    grids = (
        SpatialGrid(
            name="Grid",
            rows=8,
            columns=12,
            x_spacing=102.5,
            y_spacing=103.25,
            x_origin=71,
            y_origin=57,
        ),
        SpatialGrid(
            name="Grid",
            rows=8,
            columns=12,
            x_spacing=102.5,
            y_spacing=103.25,
            x_origin=72,
            y_origin=57,
        ),
    )
    value = RuntimeValue(
        key=ArtifactKey(
            name="Grid",
            kind=ArtifactKind.SPATIAL_GRID,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=tuple(grid.runtime_payload() for grid in grids),
        schema=RuntimeValueSchema(
            kind=ArtifactKind.SPATIAL_GRID,
            slice_aligned=True,
        ),
    )
    store = RuntimeValueStore()
    store.record(value, path="/memory/Grid.pkl", backend="memory")
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_projects_object_numbers_per_plane(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber\n"
        "1,1\n"
        "1,2\n"
        "2,1\n"
        "2,2\n",
        encoding="utf-8",
    )
    labels = np.asarray(
        (
            ((1, 1), (2, 2)),
            ((1, 1), (2, 2)),
        ),
        dtype=np.int32,
    )
    label_set = ObjectLabelSet(
        name="Cells",
        labels=labels,
        declared_object_count=2,
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    )
    payload = label_set.runtime_payload()
    value = RuntimeValue(
        key=ArtifactKey(
            name="Cells",
            kind=ArtifactKind.OBJECT_LABELS,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=payload,
        schema=label_set.runtime_schema(payload),
    )
    store = RuntimeValueStore()
    store.record(value, path="/memory/Cells.pkl", backend="memory")
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_projects_payload_object_numbers_once(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "StraightenedWorms.csv").write_text(
        "ImageNumber,ObjectNumber\n"
        "1,1\n"
        "1,2\n",
        encoding="utf-8",
    )
    labels = np.asarray(
        (
            ((1, 1), (2, 2)),
            ((1, 1), (2, 2)),
        ),
        dtype=np.int32,
    )
    label_set = ObjectLabelSet(
        name="StraightenedWorms",
        labels=labels,
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    )
    payload = label_set.runtime_payload()
    value = RuntimeValue(
        key=ArtifactKey(
            name="StraightenedWorms",
            kind=ArtifactKind.OBJECT_LABELS,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=payload,
        schema=label_set.runtime_schema(payload),
    )
    store = RuntimeValueStore()
    store.record(value, path="/memory/StraightenedWorms.pkl", backend="memory")
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
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


def test_runtime_output_equivalence_compares_npy_image_exports(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    pixels = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.save(reference_root / "native_name.npy", pixels)
    np.save(candidate_root / "openhcs_name.npy", pixels.copy())

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.is_equivalent


def test_runtime_output_equivalence_treats_replicated_rgb_as_grayscale_exactly(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    pixels = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.save(reference_root / "native_name.npy", np.repeat(pixels[..., None], 3, axis=2))
    np.save(candidate_root / "openhcs_name.npy", pixels.copy())

    report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )

    assert report.is_equivalent


def test_cellprofiler_runtime_policy_allows_float32_image_roundoff(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    pixels = np.linspace(1.0, 2.0, 9, dtype=np.float32).reshape(3, 3)
    candidate_pixels = np.nextafter(pixels, np.float32(np.inf))
    np.save(reference_root / "native_name.npy", np.repeat(pixels[..., None], 3, axis=2))
    np.save(candidate_root / "openhcs_name.npy", candidate_pixels)

    strict_report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )
    cellprofiler_report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert strict_report.failure_messages() == ("image output content differs",)
    assert cellprofiler_report.is_equivalent


def test_cellprofiler_runtime_policy_allows_measurement_roundoff() -> None:
    feature = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, None),
        "final_threshold",
        source_name="Embryos",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.111704132"): 1}
            )
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(
                        RuntimeCellValueKind.NUMBER,
                        "0.111704111",
                    ): 1
                }
            )
        }
    )

    strict_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(),
    )
    cellprofiler_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert not strict_report.is_equivalent
    assert cellprofiler_report.is_equivalent


def test_cellprofiler_measurement_projection_localizes_parent_image_numbers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    (reference_root / "Sequence2").mkdir(parents=True)
    (candidate_root / "Sequence2").mkdir(parents=True)
    (reference_root / "Sequence2" / "Embryos.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "TrackObjects_ParentImageNumber_50,"
        "TrackObjects_Lifetime_50\n"
        "22,1,0,1\n"
        "23,1,22,2\n",
        encoding="utf-8",
    )
    (candidate_root / "Sequence2" / "Embryos.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "TrackObjects_ParentImageNumber_50,"
        "TrackObjects_Lifetime_50\n"
        "1,1,0,1\n"
        "2,1,1,2\n",
        encoding="utf-8",
    )
    (reference_root / "Sequence2" / "Image.csv").write_text(
        "ImageNumber,Mean_Embryos_TrackObjects_ParentImageNumber_50\n"
        "22,0\n"
        "23,22\n",
        encoding="utf-8",
    )
    (candidate_root / "Sequence2" / "Image.csv").write_text(
        "ImageNumber,Mean_Embryos_TrackObjects_ParentImageNumber_50\n"
        "1,0\n"
        "2,1\n",
        encoding="utf-8",
    )

    reference = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        policy=cellprofiler_runtime_equivalence_policy(),
    )
    candidate = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(candidate_root),
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
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


def test_runtime_output_equivalence_allows_sparse_image_pixel_jitter(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_pixels = np.zeros((10, 10), dtype=np.uint8)
    candidate_pixels = reference_pixels.copy()
    candidate_pixels[3, 4] = 255
    imageio.imwrite(reference_root / "native_name.tif", reference_pixels)
    imageio.imwrite(candidate_root / "openhcs_name.tif", candidate_pixels)

    strict_report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
    )
    jitter_report = runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        RuntimeOutputSnapshot.from_output_root(candidate_root),
        policy=RuntimeEquivalencePolicy(image_max_different_fraction=0.02),
    )

    assert strict_report.failure_messages() == ("image output content differs",)
    assert jitter_report.is_equivalent


def test_runtime_reference_artifact_equivalence_uses_declared_image_artifacts(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_images = candidate_root / "images"
    candidate_images.mkdir(parents=True)
    rgb_pixels = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    imageio.imwrite(reference_root / "native_name.tif", rgb_pixels)
    imageio.imwrite(
        candidate_images / "source_channel.tif",
        np.zeros((3, 3), dtype=np.uint8),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="RGBImage",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=rgb_pixels.copy(),
            schema=RuntimeValueSchema(kind=ArtifactKind.IMAGE),
        ),
        path="/memory/RGBImage.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Intermediate",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.zeros((3, 3), dtype=np.uint8),
            schema=RuntimeValueSchema(kind=ArtifactKind.IMAGE),
        ),
        path="/memory/Intermediate.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        candidate_image_artifact_names=frozenset(("RGBImage",)),
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_can_use_exported_image_files(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_images = candidate_root / "images"
    candidate_images.mkdir(parents=True)
    first_pixels = np.full((3, 3), 7, dtype=np.uint8)
    second_pixels = np.full((3, 3), 11, dtype=np.uint8)
    imageio.imwrite(reference_root / "native_first.tif", first_pixels)
    imageio.imwrite(reference_root / "native_second.tif", second_pixels)
    imageio.imwrite(candidate_images / "openhcs_first.tif", first_pixels)
    imageio.imwrite(candidate_images / "openhcs_second.tif", second_pixels)
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MontageImage",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.zeros((3, 3), dtype=np.uint8),
            schema=RuntimeValueSchema(kind=ArtifactKind.IMAGE),
        ),
        path="/memory/MontageImage.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        candidate_image_artifact_names=frozenset(("MontageImage",)),
        candidate_image_snapshots=RuntimeOutputSnapshot.from_export_observation(
            observation.exports
        ).images,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_long_form_aggregate_rows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Mean_Embryos_TrackObjects_Label_50\n"
        "1,1.5\n",
        encoding="utf-8",
    )
    (reference_root / "Embryos.csv").write_text(
        "ImageNumber,ObjectNumber,TrackObjects_Label_50\n"
        "1,1,1\n"
        "1,2,2\n",
        encoding="utf-8",
    )
    table = MeasurementTable(
        name="TrackObjectsMeasurements",
        rows=(
            {
                "image_number": 1,
                "object_name": "Embryos",
                "object_label": 1,
                "measurement_name": "TrackObjects_Label_50",
                "measurement_value": 1,
            },
            {
                "image_number": 1,
                "object_name": "Embryos",
                "object_label": 2,
                "measurement_name": "TrackObjects_Label_50",
                "measurement_value": 2,
            },
            {
                "image_number": 1,
                "source_image_name": "Image",
                "measurement_name": "Mean_Embryos_TrackObjects_Label_50",
                "measurement_value": 1.5,
            },
        ),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="TrackObjectsMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/TrackObjectsMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_skips_candidate_images_without_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Count_Nuclei\n1,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="RGBImage",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.zeros((3, 3), dtype=np.uint8),
            schema=RuntimeValueSchema(kind=ArtifactKind.IMAGE),
        ),
        path="/memory/RGBImage.pkl",
        backend="memory",
    )
    native_table = MeasurementTable(
        name="ImageMeasurements",
        rows=({"count_nuclei": 2},),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="ImageMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/ImageMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    def fail_from_array(*_args, **_kwargs):
        raise AssertionError("candidate images should not be snapshotted")

    monkeypatch.setattr(RuntimeImageSnapshot, "from_array", fail_from_array)

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        candidate_image_artifact_names=frozenset(("RGBImage",)),
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_ignores_internal_tables_without_reference_tables(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="InternalMeasurements",
        rows=({"object_label": 1, "object_name": "Cells", "area": 10},),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="InternalMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/InternalMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_applies_image_export_encoding(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    float_pixels = np.linspace(0.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
    imageio.imwrite(
        reference_root / "native_name.tif",
        np.rint(float_pixels * 255.0).astype(np.uint8),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="RGBImage",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=float_pixels,
            schema=RuntimeValueSchema(kind=ArtifactKind.IMAGE),
        ),
        path="/memory/RGBImage.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        candidate_image_export_specs=(
            RuntimeImageExportSpec(
                "RGBImage",
                bit_depth=RuntimeImageExportBitDepth.UINT8,
                file_format="tiff",
            ),
        ),
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_collapses_singleton_image_stack(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    imageio.imwrite(reference_root / "native_name.png", pixels)
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="OverlayImage",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=pixels[np.newaxis, ...],
            schema=RuntimeValueSchema(kind=ArtifactKind.IMAGE),
        ),
        path="/memory/OverlayImage.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        candidate_image_export_specs=(
            RuntimeImageExportSpec(
                "OverlayImage",
                bit_depth=RuntimeImageExportBitDepth.UINT8,
                file_format="png",
            ),
        ),
    )

    assert report.is_equivalent


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


def test_runtime_output_snapshot_from_artifact_execution_ignores_auxiliary_tables(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    output_root.mkdir()
    (
        output_root / "A01_Measurements_step1.csv"
    ).write_text("measurement\n1\n", encoding="utf-8")
    (output_root / "metaxpress_style_summary.csv").write_text(
        "Barcode,OpenHCS-Plate,,\n"
        "Plate Name,Auxiliary Summary,,\n"
        "well,metric\n"
        "A01,1\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
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
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        output_root,
    )

    snapshot = RuntimeOutputSnapshot.from_artifact_execution_observation(observation)

    assert tuple(table.path.name for table in snapshot.tables) == (
        "A01_Measurements_step1.csv",
    )


def test_runtime_output_snapshot_from_artifact_execution_ignores_undeclared_images(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True)
    imageio.imwrite(image_dir / "A01_source.png", np.zeros((2, 2), dtype=np.uint8))
    store = RuntimeValueStore()
    store.record(
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
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        output_root,
    )

    snapshot = RuntimeOutputSnapshot.from_artifact_execution_observation(observation)

    assert snapshot.images == ()


def test_runtime_reference_artifact_equivalence_uses_measurement_facts(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Area\n"
        "1,1,3.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=({"object_label": 1, "area": 3.0, "object_name": "Cells"},),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_uses_numeric_tolerance(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Area\n"
        "1,1,3.0000001\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=({"object_label": 1, "area": 3.0000002, "object_name": "Cells"},),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(numeric_abs_tolerance=1e-6),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_uses_feature_numeric_tolerance() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="granularity_1",
        source_name="DNA",
    )
    object_count_feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="object_count",
        statistic="count",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0"): 1,
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.0"): 1,
                }
            ),
            object_count_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "2"): 1}
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.4"): 1,
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.4"): 1,
                }
            ),
            object_count_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "2"): 1}
            ),
        }
    )

    strict_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(),
    )
    tolerant_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(
            feature_numeric_tolerances=(
                RuntimeMeasurementFeatureNumericTolerance(
                    feature_name_prefixes=("granularity_",),
                    subject_scope=MeasurementScope.OBJECT,
                    statistic="value",
                    numeric_abs_tolerance=0.5,
                    require_object_count_stability=True,
                ),
            )
        ),
    )

    assert not strict_report.is_equivalent
    assert tolerant_report.is_equivalent


def test_cellprofiler_track_objects_motion_features_use_declared_numeric_tolerance() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Embryos")
    feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="track_objects_integrated_distance_50",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "28.4402354989"): 1}
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "28.4077925806"): 1}
            ),
        }
    )

    assert runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    ).is_equivalent


def test_runtime_measurement_equivalence_allows_duplicate_location_projection() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    feature = RuntimeMeasurementFeatureKey(subject, "center_x")
    mean_feature = RuntimeMeasurementFeatureKey(
        subject,
        "center_x",
        statistic="mean",
    )
    first = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.5")
    second = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.5")
    mean = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "15.5")
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter({first: 1, second: 2}),
            mean_feature: Counter({mean: 1}),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter({first: 2, second: 4}),
            mean_feature: Counter({mean: 2}),
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_allows_duplicate_object_value_projection() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    feature = RuntimeMeasurementFeatureKey(subject, "area")
    first = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0")
    second = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.0")
    reference = RuntimeMeasurementSnapshot({feature: Counter({first: 1, second: 2})})
    candidate = RuntimeMeasurementSnapshot({feature: Counter({first: 3, second: 6})})

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_allows_duplicate_object_value_aggregate() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    value_feature = RuntimeMeasurementFeatureKey(subject, "area")
    mean_feature = RuntimeMeasurementFeatureKey(
        subject,
        "area",
        statistic="mean",
    )
    first = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0")
    second = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.0")
    mean = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "15.0")
    reference = RuntimeMeasurementSnapshot(
        {
            value_feature: Counter({first: 1, second: 1}),
            mean_feature: Counter({mean: 1}),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            value_feature: Counter({first: 3, second: 3}),
            mean_feature: Counter({mean: 2}),
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_allows_reference_relationship_row_projection() -> None:
    cells_subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    cytoplasm_subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        "cytoplasm",
    )
    child_count_feature = RuntimeMeasurementFeatureKey(
        cells_subject,
        "cytoplasm_count",
    )
    parent_feature = RuntimeMeasurementFeatureKey(cytoplasm_subject, "cells")
    mean_feature = RuntimeMeasurementFeatureKey(
        cells_subject,
        "cytoplasm_count",
        statistic="mean",
    )
    one = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.0")
    two = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "2.0")
    mean = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.0")
    reference = RuntimeMeasurementSnapshot(
        {
            child_count_feature: Counter({one: 81}),
            parent_feature: Counter({one: 3, two: 2}),
            mean_feature: Counter({mean: 3}),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            child_count_feature: Counter({one: 34}),
            parent_feature: Counter({one: 1, two: 1}),
            mean_feature: Counter({mean: 1}),
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_allows_relationship_zero_padding() -> None:
    subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        "eroded_downsized_nuclei",
    )
    feature = RuntimeMeasurementFeatureKey(subject, "downsized_nuclei")
    zero = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.0")
    one = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.0")
    two = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "2.0")
    reference = RuntimeMeasurementSnapshot({feature: Counter({one: 1, two: 1})})
    candidate = RuntimeMeasurementSnapshot({feature: Counter({zero: 17, one: 1, two: 1})})

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_rejects_reference_projection_for_shape_values() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    feature = RuntimeMeasurementFeatureKey(subject, "area")
    first = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0")
    second = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.0")
    reference = RuntimeMeasurementSnapshot({feature: Counter({first: 3, second: 2})})
    candidate = RuntimeMeasurementSnapshot({feature: Counter({first: 1, second: 1})})

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert not report.is_equivalent


def test_runtime_measurement_equivalence_rejects_new_location_values() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    feature = RuntimeMeasurementFeatureKey(subject, "center_x")
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.5"): 1}
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.5"): 1,
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.5"): 1,
                }
            ),
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert not report.is_equivalent


def test_runtime_measurement_equivalence_allows_duplicated_boundary_location_jitter() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Worms")
    feature = RuntimeMeasurementFeatureKey(subject, "center_x")
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0"): 1,
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "20.0"): 1,
                }
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "11.0"): 2,
                    RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "19.0"): 2,
                }
            ),
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_ignores_child_object_number_means() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Parents")
    feature = RuntimeMeasurementFeatureKey(
        subject,
        "mean_children_object_number",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.0"): 1}
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "12.0"): 1}
            ),
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_feature_numeric_tolerance_does_not_apply_to_other_features() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="mean_intensity",
        source_name="DNA",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.0"): 1}
            )
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "10.4"): 1}
            )
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(
            feature_numeric_tolerances=(
                RuntimeMeasurementFeatureNumericTolerance(
                    feature_name_prefixes=("granularity_",),
                    subject_scope=MeasurementScope.OBJECT,
                    statistic="value",
                    numeric_abs_tolerance=0.5,
                ),
            )
        ),
    )

    assert not report.is_equivalent


def test_cellprofiler_policy_tolerates_area_derived_colocalized_fraction() -> None:
    feature = RuntimeMeasurementFeatureKey(
        subject=RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image"),
        feature_name="colocalized_stain_1",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(
                        RuntimeCellValueKind.NUMBER,
                        "0.520751619082368",
                    ): 1
                }
            )
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(
                        RuntimeCellValueKind.NUMBER,
                        "0.5207991242474002",
                    ): 1
                }
            )
        }
    )

    strict_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(),
    )
    cellprofiler_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert not strict_report.is_equivalent
    assert cellprofiler_report.is_equivalent


def test_cellprofiler_dialect_places_image_sources_in_feature_suffix() -> None:
    identity = CELLPROFILER_MEASUREMENT_DIALECT.encode_source_qualified_feature(
        "align_xshift",
        "Stain1",
        MeasurementScope.IMAGE,
    )

    assert identity.feature_name == "align_xshift_stain_1"
    assert identity.source_name is None


def test_cellprofiler_dialect_places_sources_before_row_qualifiers() -> None:
    identity = CELLPROFILER_MEASUREMENT_DIALECT.encode_source_qualified_feature(
        "sum_variance_10_01_256",
        "CorrGray",
        MeasurementScope.OBJECT,
        qualifiers=("10", "01", "256"),
    )

    assert identity.feature_name == "sum_variance_corr_gray_10_01_256"
    assert identity.source_name is None


def test_cellprofiler_policy_tolerates_image_texture_numeric_drift() -> None:
    feature = RuntimeMeasurementFeatureKey(
        subject=RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image"),
        feature_name="sum_variance_corr_gray_10_01_256",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "831.4919758885818"): 1}
            )
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "831.6043097244797"): 1}
            )
        }
    )

    strict_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(),
    )
    cellprofiler_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert not strict_report.is_equivalent
    assert cellprofiler_report.is_equivalent


def test_cellprofiler_object_texture_tolerance_requires_stable_object_count() -> None:
    feature = RuntimeMeasurementFeatureKey(
        subject=RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "comet"),
        feature_name="sum_variance_corr_gray_10_01_256",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "831.4919758885818"): 1}
            )
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(
                        RuntimeCellValueKind.NUMBER,
                        "831.6043097244797",
                    ): 2,
                }
            )
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert not report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_max_location_ties(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "Intensity_MaxIntensity_CropBlue,"
        "Location_MaxIntensity_X_CropBlue\n"
        "1,1,0.5,10\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "max_intensity": 0.5,
                "max_intensity_x": 12,
                "source_image_name": "CropBlue",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectIntensity",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    tie_policy_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(
            allow_tie_sensitive_location_mismatches=True
        ),
    )

    assert strict_report.failure_messages() == (
        "measurement feature object:nuclei/max_intensity_x_crop_blue values differ",
    )
    assert tie_policy_report.is_equivalent


def test_runtime_measurement_equivalence_allows_aggregate_max_location_ties_when_value_is_stable() -> None:
    location_feature = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei"),
        "mean_h_2_ax_max_intensity_x_orig_green",
    )
    value_feature = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei"),
        "mean_h_2_ax_max_intensity_orig_green",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            location_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "371.1578947368"): 1}
            ),
            value_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.5698784421"): 1}
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            location_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "371.1315789474"): 1}
            ),
            value_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.5698784421"): 1}
            ),
        }
    )

    strict_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(),
    )
    tie_policy_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(
            allow_tie_sensitive_location_mismatches=True
        ),
    )

    assert strict_report.failure_messages() == (
        "measurement feature object:nuclei/mean_h_2_ax_max_intensity_x_orig_green values differ",
    )
    assert tie_policy_report.is_equivalent


def test_runtime_reference_artifact_equivalence_uses_object_table_source_image(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "Image,Cells,Cells\n"
        "ImageNumber,ObjectNumber,Texture_AngularSecondMoment_DNA_3_00_256\n"
        "1,1,0.25\n",
        encoding="utf-8",
    )
    table = MeasurementTable(
        name="MeasureTexture",
        rows=(
            {
                "slice_index": 0,
                "object_label": 1,
                "object_name": "Cells",
                "scale": 3,
                "direction": 0,
                "gray_levels": 256,
                "angular_second_moment": 0.25,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
        object_name="Cells",
        source_image_name="DNA",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureTexture",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/MeasureTexture.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_stable_geometry_zernike_drift(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "AreaShape_Area,AreaShape_Center_X,AreaShape_Center_Y,"
        "AreaShape_Zernike_0_0\n"
        "1,1,10,3.5,4.5,0.80\n"
        "1,2,20,7.5,8.5,0.90\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "area": 10,
                "center_x": 3.5,
                "center_y": 4.5,
                "Zernike_0_0": 0.82,
            },
            {
                "object_label": 2,
                "object_name": "Nuclei",
                "area": 20,
                "center_x": 7.5,
                "center_y": 8.5,
                "Zernike_0_0": 0.88,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    shape_policy_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(allow_unstable_shape_descriptors=True),
    )

    assert strict_report.failure_messages() == (
        "measurement feature object:nuclei/zernike_0_0 values differ",
    )
    assert shape_policy_report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_sparse_zernike_after_stable_values(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_rows = [
        f"1,{object_number},{object_number * 10},{object_number + 0.5},"
        f"{object_number + 1.5},{0.1 + object_number * 0.0001:.10f}"
        for object_number in range(1, 11)
    ]
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "AreaShape_Area,AreaShape_Center_X,AreaShape_Center_Y,"
        "AreaShape_Zernike_0_0\n"
        + "\n".join(reference_rows)
        + "\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    rows = []
    for object_number in range(1, 11):
        zernike = 0.1 + object_number * 0.0001
        if object_number == 4:
            zernike += 0.02
        if object_number == 8:
            zernike -= 0.02
        rows.append(
            {
                "object_label": object_number,
                "object_name": "Nuclei",
                "area": object_number * 10,
                "center_x": object_number + 0.5,
                "center_y": object_number + 1.5,
                "Zernike_0_0": zernike,
            }
        )
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=tuple(rows),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(
            numeric_abs_tolerance=1e-6,
            allow_unstable_shape_descriptors=True,
        ),
    )

    assert report.is_equivalent


def test_runtime_measurement_equivalence_tolerates_intensity_zernike_boundary_drift_only_with_stable_geometry() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    magnitude_feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="ZernikeMagnitude_9_9",
        source_name="df_image",
    )
    phase_feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="ZernikePhase_9_9",
        source_name="df_image",
    )
    area_feature = RuntimeMeasurementFeatureKey(subject=subject, feature_name="area")
    center_x_feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="center_x",
    )
    center_y_feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="center_y",
    )
    object_number_feature = RuntimeMeasurementFeatureKey(
        subject=subject,
        feature_name="number_object_number",
    )
    reference = RuntimeMeasurementSnapshot(
        {
            magnitude_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.00000031"): 1}
            ),
            phase_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "2.40"): 1}
            ),
            area_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "915"): 1}
            ),
            center_x_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1180.5"): 1}
            ),
            center_y_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "356.16666667"): 1}
            ),
            object_number_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "637"): 1}
            ),
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            magnitude_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.00000079"): 1}
            ),
            phase_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "2.62"): 1}
            ),
            area_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "915"): 1}
            ),
            center_x_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1180.5"): 1}
            ),
            center_y_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "356.16666667"): 1}
            ),
            object_number_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "637"): 1}
            ),
        }
    )
    unstable_geometry_candidate = RuntimeMeasurementSnapshot(
        {
            **candidate.values_by_feature,
            center_y_feature: Counter(
                {RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "357.16666667"): 1}
            ),
        }
    )

    strict_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(),
    )
    zernike_policy_report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=RuntimeEquivalencePolicy(allow_unstable_zernike_descriptors=True),
    )
    unstable_geometry_report = runtime_measurement_equivalence(
        reference,
        unstable_geometry_candidate,
        policy=RuntimeEquivalencePolicy(allow_unstable_zernike_descriptors=True),
    )

    assert not strict_report.is_equivalent
    assert zernike_policy_report.is_equivalent
    assert not unstable_geometry_report.is_equivalent


def test_runtime_reference_artifact_equivalence_rejects_zernike_drift_when_geometry_changes(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "AreaShape_Area,AreaShape_Center_X,AreaShape_Center_Y,"
        "AreaShape_Zernike_0_0\n"
        "1,1,10,3.5,4.5,0.80\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "area": 11,
                "center_x": 3.5,
                "center_y": 4.5,
                "Zernike_0_0": 0.82,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(allow_unstable_shape_descriptors=True),
    )

    assert not report.is_equivalent
    assert (
        "measurement feature object:nuclei/area values differ"
        in report.failure_messages()
    )
    assert (
        "measurement feature object:nuclei/zernike_0_0 values differ"
        in report.failure_messages()
    )


def test_runtime_reference_artifact_equivalence_allows_stable_geometry_orientation_sign(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "AreaShape_Area,AreaShape_Center_X,AreaShape_Center_Y,"
        "AreaShape_Orientation\n"
        "1,1,10,3.5,4.5,-45\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "area": 10,
                "center_x": 3.5,
                "center_y": 4.5,
                "orientation": 45,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    shape_policy_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(allow_unstable_shape_descriptors=True),
    )

    assert strict_report.failure_messages() == (
        "measurement feature object:nuclei/orientation values differ",
    )
    assert shape_policy_report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_relationship_mean_orientation_boundary_drift(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "AreaShape_Area,AreaShape_Center_X,AreaShape_Center_Y,"
        "Mean_Nucleoli_AreaShape_Orientation\n"
        "1,1,10,3.5,4.5,-45\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="RelateObjects",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "area": 10,
                "center_x": 3.5,
                "center_y": 4.5,
                "mean_nucleoli_orientation": 45,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="RelateObjects",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/RelateObjects.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    shape_policy_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(allow_unstable_shape_descriptors=True),
    )

    assert strict_report.failure_messages() == (
        "measurement feature object:nuclei/mean_nucleoli_orientation values differ",
    )
    assert shape_policy_report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_mean_aggregates(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Mean_Cells_AreaShape_Area\n"
        "1,2.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {"object_label": 1, "area": 1.0, "object_name": "Cells"},
            {"object_label": 2, "area": 3.0, "object_name": "Cells"},
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_mean_aggregates_ignore_nonfinite(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Mean_Cells_AreaShape_Area\n"
        "1,2.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {"object_label": 1, "area": 1.0, "object_name": "Cells"},
            {"object_label": 2, "area": float("nan"), "object_name": "Cells"},
            {"object_label": 3, "area": 3.0, "object_name": "Cells"},
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_threshold_entropy_jitter(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Threshold_SumOfEntropies_Cells\n"
        "1,-10.60\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    image_table = MeasurementTable(
        name="IdentifyPrimaryObjects",
        rows=({"image_number": 1, "sum_of_entropies_cells": -10.96},),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="IdentifyPrimaryObjects",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=image_table.rows,
            schema=image_table.runtime_schema(image_table.rows),
        ),
        path="/memory/IdentifyPrimaryObjects.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    entropy_policy_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(threshold_entropy_abs_tolerance=0.5),
    )

    assert strict_report.failure_messages() == (
        "measurement feature image:image/sum_of_entropies_cells values differ",
    )
    assert entropy_policy_report.is_equivalent


def test_cellprofiler_runtime_policy_allows_threshold_entropy_roundoff(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Threshold_SumOfEntropies_Cells\n"
        "1,-12.7688016836\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    image_table = MeasurementTable(
        name="IdentifyPrimaryObjects",
        rows=({"image_number": 1, "sum_of_entropies_cells": -12.7688158687},),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="IdentifyPrimaryObjects",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=image_table.rows,
            schema=image_table.runtime_schema(image_table.rows),
        ),
        path="/memory/IdentifyPrimaryObjects.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_cellprofiler_runtime_policy_allows_max_location_ties(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "Intensity_MaxIntensity_CropBlue,"
        "Location_MaxIntensity_X_CropBlue\n"
        "1,1,0.5,10\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "max_intensity": 0.5,
                "max_intensity_x": 12,
                "source_image_name": "CropBlue",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectIntensity",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_sparse_object_boundary_jitter(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_rows = "\n".join(
        f"1,{object_number},10"
        for object_number in range(1, 11)
    )
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Area\n"
        f"{reference_rows}\n",
        encoding="utf-8",
    )
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Count_Cells,Mean_Cells_AreaShape_Area\n"
        "1,10,10.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    object_rows = tuple(
        {
            "object_label": object_number,
            "object_name": "Cells",
            "area": 13 if object_number == 4 else 10,
        }
        for object_number in range(1, 11)
    )
    object_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=object_rows,
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    image_table = MeasurementTable(
        name="IdentifyPrimaryObjects",
        rows=({"image_number": 1, "count_cells": 10},),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    for table in (object_table, image_table):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/{table.name}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    boundary_policy_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(
            allow_sparse_object_boundary_jitter=True,
            object_boundary_jitter_abs_tolerance=5,
            object_boundary_jitter_max_unstable_values=1,
            object_boundary_jitter_max_unstable_fraction=0,
            object_boundary_jitter_aggregate_abs_tolerance=0.5,
        ),
    )

    assert "measurement feature object:cells/area values differ" in (
        strict_report.failure_messages()
    )
    assert (
        "measurement feature object:cells/mean(area) values differ"
        in strict_report.failure_messages()
    )
    assert boundary_policy_report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_boundary_jitter_without_count(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_rows = "\n".join(
        f"1,{object_number},10"
        for object_number in range(1, 11)
    )
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Area\n"
        f"{reference_rows}\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    object_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=tuple(
            {
                "object_label": object_number,
                "object_name": "Cells",
                "area": 13 if object_number == 4 else 10,
            }
            for object_number in range(1, 11)
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=object_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=object_table.rows,
            schema=object_table.runtime_schema(object_table.rows),
        ),
        path=f"/memory/{object_table.name}.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(
            allow_sparse_object_boundary_jitter=True,
            object_boundary_jitter_abs_tolerance=5,
            object_boundary_jitter_max_unstable_values=1,
            object_boundary_jitter_max_unstable_fraction=0,
        ),
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_qualified_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Texture_AngularSecondMoment_CropBlue_3_00_256\n"
        "1,1,0.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureTexture",
        rows=(
            {
                "object_label": 1,
                "scale": 3,
                "direction": 0,
                "gray_levels": 256,
                "angular_second_moment": 0.5,
                "object_name": "Nuclei",
                "source_image_name": "CropBlue",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureTexture",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureTexture.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_binned_row_qualifiers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,RadialDistribution_FracAtD_CropBlue_1of4\n"
        "1,1,0.25\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectIntensityDistribution",
        rows=(
            {
                "object_label": 1,
                "bin_index": 1,
                "bin_count": 4,
                "frac_at_d": 0.25,
                "object_name": "Nuclei",
                "source_image_name": "CropBlue",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectIntensityDistribution",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectIntensityDistribution.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_cellprofiler_runtime_policy_allows_source_qualified_intensity_boundary_jitter(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_rows = "\n".join(
        f"1,{object_number},10.0"
        for object_number in range(1, 101)
    )
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Intensity_IntegratedIntensity_CorrProtein\n"
        f"{reference_rows}\n",
        encoding="utf-8",
    )
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Count_Cells,Mean_Cells_Intensity_IntegratedIntensity_CorrProtein\n"
        "1,100,10.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    object_rows = tuple(
        {
            "object_label": object_number,
            "object_name": "Cells",
            "integrated_intensity": 10.5 if object_number == 4 else 10.0,
            "source_image_name": "CorrProtein",
        }
        for object_number in range(1, 101)
    )
    object_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=object_rows,
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    image_table = MeasurementTable(
        name="IdentifyPrimaryObjects",
        rows=({"image_number": 1, "count_cells": 100},),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    for table in (object_table, image_table):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/{table.name}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_cellprofiler_runtime_policy_allows_calculated_object_boundary_jitter(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_rows = "\n".join(
        f"1,{object_number},1.0"
        for object_number in range(1, 101)
    )
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Math_Ratio1\n"
        f"{reference_rows}\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    object_rows = tuple(
        {
            "object_label": object_number,
            "object_name": "Cells",
            "feature_name": "Math_Ratio1",
            "result_value": 1.002 if object_number == 4 else 1.0,
        }
        for object_number in range(1, 101)
    )
    object_table = MeasurementTable(
        name="CalculateMath",
        rows=object_rows,
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    count_table = MeasurementTable(
        name="IdentifyPrimaryObjects",
        rows=({"image_number": 1, "count_cells": 100},),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    for table in (object_table, count_table):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/{table.name}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_cellprofiler_dialect_classifies_worm_and_related_child_features_as_calculated() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "NonOverlappingWorms")

    for feature_name in (
        "worm_length",
        "worm_control_point_x_1",
        "fat_regions_count",
        "mean_fat_regions_area",
    ):
        assert object_measurement_feature_has_role(
            RuntimeMeasurementFeatureKey(
                subject=subject,
                feature_name=feature_name,
                statistic=MeasurementStatistic.VALUE.value,
            ),
            ObjectMeasurementFeatureRole.CALCULATED,
            CELLPROFILER_MEASUREMENT_DIALECT,
        )


def test_calculated_object_features_do_not_require_count_stability_for_boundary_jitter() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "NonOverlappingWorms")

    assert not object_measurement_feature_requires_sparse_boundary_object_count_stability(
        RuntimeMeasurementFeatureKey(
            subject=subject,
            feature_name="mean_fat_regions_area",
            statistic=MeasurementStatistic.VALUE.value,
        ),
        CELLPROFILER_MEASUREMENT_DIALECT,
    )


def test_runtime_reference_artifact_equivalence_ignores_missing_row_qualifiers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Texture_Entropy\n"
        "1,1,0.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureTexture",
        rows=(
            {
                "object_label": 1,
                "direction": float("nan"),
                "texture_entropy": 0.5,
                "object_name": "Nuclei",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureTexture",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureTexture.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_numbered_feature_aliases(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Granularity_1_CropBlue_Nuclei\n"
        "1,1,0.75\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureGranularity",
        rows=(
            {
                "object_id": 1,
                "gs1": 0.75,
                "object_name": "Nuclei",
                "source_image_name": "CropBlue",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureGranularity",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureGranularity.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_measurement_snapshot_normalizes_exported_numeric_qualifiers(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    output_root.mkdir()
    (output_root / "texture.csv").write_text(
        "slice_index,scale,direction,gray_levels,angular_second_moment,source_image_name\n"
        "0.0,3.0,0.0,256.0,0.5,CropBlue\n",
        encoding="utf-8",
    )

    snapshot = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(output_root),
        policy=RuntimeEquivalencePolicy(),
    )

    feature_names = {key.feature_name for key in snapshot.values_by_feature}
    assert "angular_second_moment_crop_blue_3_00_256" in feature_names


def test_runtime_reference_artifact_equivalence_matches_relationship_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Children_Nuclei_Count\n"
        "1,1,2\n"
        "1,2,1\n",
        encoding="utf-8",
    )
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Parent_Cells\n"
        "1,1,1\n"
        "1,2,1\n"
        "1,3,2\n",
        encoding="utf-8",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1, 2),
        target_ids=(1, 2, 3),
        relationship_type=semantics.relationship_type,
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_pads_relationship_counts_from_labels(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Children_Nuclei_Count\n"
        "1,1,0\n"
        "1,2,1\n"
        "1,3,0\n",
        encoding="utf-8",
    )
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Parent_Cells\n"
        "1,1,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    labels = np.array([[1, 2, 3]], dtype=np.int32)
    object_labels = ObjectLabelSet(name="Cells", labels=labels)
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=object_labels.runtime_payload(),
            schema=object_labels.runtime_schema(object_labels.runtime_payload()),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(2,),
        target_ids=(1,),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_relationship_child_means(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green,"
        "Mean_Nuclei_Number_Object_Number\n"
        "1,1,2.0,1.5\n"
        "1,2,5.0,3.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 1.0,
            },
            {
                "object_name": "Nuclei",
                "object_label": 2,
                "integrated_intensity": 3.0,
            },
            {
                "object_name": "Nuclei",
                "object_label": 3,
                "integrated_intensity": 5.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1, 2),
        target_ids=(1, 2, 3),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_parent_qualified_relationship_distances(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nucleoli_Distance_Centroid,"
        "Mean_Nucleoli_Distance_Minimum\n"
        "1,1,2.0,3.0\n"
        "1,2,6.0,8.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="RelateObjects",
        rows=(
            {
                "object_name": "Nucleoli",
                "object_label": 1,
                "distance_centroid_nuclei": 1.0,
                "distance_minimum_nuclei": 2.0,
            },
            {
                "object_name": "Nucleoli",
                "object_label": 2,
                "distance_centroid_nuclei": 3.0,
                "distance_minimum_nuclei": 4.0,
            },
            {
                "object_name": "Nucleoli",
                "object_label": 3,
                "distance_centroid_nuclei": 6.0,
                "distance_minimum_nuclei": 8.0,
            },
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/RelateObjects.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Nuclei", "Nucleoli")
    relationship = ObjectRelationship(
        name="Nuclei_Nucleoli_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1, 2),
        target_ids=(1, 2, 3),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Nuclei_Nucleoli_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_relationship_child_means_with_canonical_endpoint_names(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_H2AX_Intensity_IntegratedIntensity_Green\n"
        "1,1,2.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_name": "h_2_ax",
                "object_label": 1,
                "integrated_intensity": 1.0,
            },
            {
                "object_name": "h_2_ax",
                "object_label": 2,
                "integrated_intensity": 3.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Nuclei", "H2AX")
    relationship = ObjectRelationship(
        name="Nuclei_H2AX_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Nuclei_H2AX_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_relationship_child_location_means_with_canonical_endpoint_names(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "Mean_H2AX_Location_Center_X,"
        "Mean_H2AX_Location_Center_Y,"
        "Mean_H2AX_Location_Center_Z\n"
        "1,1,1.0,2.0,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectLocation",
        rows=(
            {
                "object_name": "h_2_ax",
                "object_label": 1,
                ObjectCoreMeasurementFeature.CENTER_X.value: 0.0,
                ObjectCoreMeasurementFeature.CENTER_Y.value: 1.0,
                ObjectCoreMeasurementFeature.CENTER_Z.value: 0.0,
            },
            {
                "object_name": "h_2_ax",
                "object_label": 2,
                ObjectCoreMeasurementFeature.CENTER_X.value: 2.0,
                ObjectCoreMeasurementFeature.CENTER_Y.value: 3.0,
                ObjectCoreMeasurementFeature.CENTER_Z.value: 0.0,
            },
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectLocation.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Nuclei", "H2AX")
    relationship = ObjectRelationship(
        name="Nuclei_H2AX_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Nuclei_H2AX_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_relationship_child_location_means_from_object_labels(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "Mean_H2AX_Location_Center_X,"
        "Mean_H2AX_Location_Center_Y,"
        "Mean_H2AX_Location_Center_Z\n"
        "1,1,1.0,0.5,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    h2ax = ObjectLabelSet(
        name="H2AX",
        labels=np.asarray(
            (
                (1, 1, 0),
                (0, 2, 2),
            ),
            dtype=np.uint16,
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=h2ax.name,
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=h2ax.runtime_payload(),
            schema=h2ax.runtime_schema(h2ax.runtime_payload()),
        ),
        path="/memory/H2AX.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Nuclei", "H2AX")
    relationship = ObjectRelationship(
        name="Nuclei_H2AX_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Nuclei_H2AX_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_relationship_child_location_means_from_multiplane_object_labels(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,"
        "Mean_H2AX_Location_Center_X,"
        "Mean_H2AX_Location_Center_Y,"
        "Mean_H2AX_Location_Center_Z\n"
        "1,1,1.0,0.5,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    h2ax = ObjectLabelSet(
        name="H2AX",
        labels=np.asarray(
            (
                (
                    (1, 1, 0),
                    (0, 0, 0),
                ),
                (
                    (0, 0, 0),
                    (0, 2, 2),
                ),
            ),
            dtype=np.uint16,
        ),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=h2ax.name,
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=h2ax.runtime_payload(),
            schema=h2ax.runtime_schema(h2ax.runtime_payload()),
        ),
        path="/memory/H2AX.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Nuclei", "H2AX")
    relationship = ObjectRelationship(
        name="Nuclei_H2AX_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Nuclei_H2AX_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_object_instance_plane_alignment_preserves_scoped_relationship_children_when_unscoping_values() -> None:
    child_ids_by_parent = {
        ObjectInstanceKey(1, slice_index=0): (
            ObjectInstanceKey(1, slice_index=0),
            ObjectInstanceKey(2, slice_index=0),
        ),
        ObjectInstanceKey(1): (),
    }
    values_by_child_id = {
        ObjectInstanceKey(1): 0.0,
        ObjectInstanceKey(2): 2.0,
    }

    aligned = ObjectInstanceKeyPlaneAlignmentStrategy.align_child_ids_by_parent(
        child_ids_by_parent,
        values_by_child_id,
    )

    assert aligned == {
        ObjectInstanceKey(1): (
            ObjectInstanceKey(1),
            ObjectInstanceKey(2),
        ),
    }


def test_runtime_reference_artifact_equivalence_uses_sparse_object_identity_domain_for_relationship_means(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,2.0\n"
        "1,3,6.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    cells = ObjectLabelSet(
        name="Cells",
        labels=np.asarray(
            (
                (1, 1, 0),
                (0, 3, 3),
            ),
            dtype=np.uint16,
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=cells.name,
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=cells.runtime_payload(),
            schema=cells.runtime_schema(cells.runtime_payload()),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 2.0,
            },
            {
                "object_name": "Nuclei",
                "object_label": 2,
                "integrated_intensity": 6.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 3),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_uses_represented_relationship_sources_without_declared_object_domain(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,2.0\n"
        "1,3,6.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 2.0,
            },
            {
                "object_name": "Nuclei",
                "object_label": 2,
                "integrated_intensity": 6.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 3),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_omits_missing_relationship_child_means(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,2.0\n"
        "1,2,\n"
        "1,3,6.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    cells = ObjectLabelSet(
        name="Cells",
        labels=np.asarray(
            (
                (1, 1, 0),
                (2, 2, 0),
                (3, 3, 0),
            ),
            dtype=np.uint16,
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=cells.name,
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=cells.runtime_payload(),
            schema=cells.runtime_schema(cells.runtime_payload()),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 2.0,
            },
            {
                "object_name": "Nuclei",
                "object_label": 2,
                "integrated_intensity": 6.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 3),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_aligns_image_numbered_child_rows_to_relationship_slices(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,10.0\n"
        "2,1,100.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "image_number": 1,
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 10.0,
            },
            {
                "image_number": 2,
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 100.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 1),
        relationship_type=semantics.relationship_type,
        slice_indices=(0, 1),
        slice_count=2,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_aligns_scoped_child_rows_to_relationship_slices(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,10.0\n"
        "2,1,100.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    for group_key, value in (("site0", 10.0), ("site1", 100.0)):
        child_table = MeasurementTable(
            name="MeasureObjectIntensity",
            rows=(
                {
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "integrated_intensity": value,
                },
            ),
            source_image_name="Green",
        )
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=child_table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01", group_key=group_key),
                ),
                data=child_table.rows,
                schema=child_table.runtime_schema(child_table.rows),
            ),
            path=f"/memory/MeasureObjectIntensity_{group_key}.pkl",
            backend="memory",
        )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 1),
        relationship_type=semantics.relationship_type,
        slice_indices=(0, 1),
        slice_count=2,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_aligns_scoped_relationships_to_child_rows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,10.0\n"
        "2,1,100.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    for group_key, value in (("site0", 10.0), ("site1", 100.0)):
        child_table = MeasurementTable(
            name="MeasureObjectIntensity",
            rows=(
                {
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "integrated_intensity": value,
                },
            ),
            source_image_name="Green",
        )
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=child_table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01", group_key=group_key),
                ),
                data=child_table.rows,
                schema=child_table.runtime_schema(child_table.rows),
            ),
            path=f"/memory/MeasureObjectIntensity_{group_key}.pkl",
            backend="memory",
        )
        relationship = ObjectRelationship(
            name="Cells_Nuclei_relationships",
            source=semantics.source,
            target=semantics.target,
            source_ids=(1,),
            target_ids=(1,),
            relationship_type=semantics.relationship_type,
        )
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=relationship.name,
                    kind=ArtifactKind.RELATIONSHIPS,
                    scope=ArtifactScope(axis_id="A01", group_key=group_key),
                ),
                data=relationship.runtime_payload(),
                schema=relationship.runtime_schema(relationship.runtime_payload()),
            ),
            path=f"/memory/Cells_Nuclei_relationships_{group_key}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_keys_relationship_child_means_by_slice(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,10.0\n"
        "2,1,100.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 10.0,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 100.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 1),
        relationship_type=semantics.relationship_type,
        slice_indices=(0, 1),
        slice_count=2,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_aligns_unsliced_relationship_to_single_slice_child_rows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Intensity_IntegratedIntensity_Green\n"
        "1,1,2.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    child_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "integrated_intensity": 1.0,
            },
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 2,
                "integrated_intensity": 3.0,
            },
        ),
        source_image_name="Green",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=child_table.name,
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=child_table.rows,
            schema=child_table.runtime_schema(child_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_relationship_child_label_center_means(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Mean_Nuclei_Location_Center_X,"
        "Mean_Nuclei_Location_Center_Y,Mean_Nuclei_Location_Center_Z\n"
        "1,1,1.0,0.5,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    nuclei = ObjectLabelSet(
        name="Nuclei",
        labels=np.array(
            [
                [1, 1, 0],
                [0, 2, 2],
            ],
            dtype=np.uint16,
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=nuclei.name,
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=nuclei.runtime_payload(),
            schema=nuclei.runtime_schema(nuclei.runtime_payload()),
        ),
        path="/memory/Nuclei.pkl",
        backend="memory",
    )
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_does_not_duplicate_explicit_relationship_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Children_Nuclei_Count\n"
        "1,1,2\n"
        "1,2,1\n",
        encoding="utf-8",
    )
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Parent_Cells\n"
        "1,1,1\n"
        "1,2,1\n"
        "1,3,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 1, 2),
        target_ids=(1, 2, 3),
        relationship_type=semantics.relationship_type,
    )
    for artifact_name, table in (
        (
            "Cells_relationship_measurements",
            MeasurementTable(
                name="Cells_relationship_measurements",
                rows=(
                    {
                        "object_name": "Cells",
                        "object_label": 1,
                        "Children_Nuclei_Count": 2,
                    },
                    {
                        "object_name": "Cells",
                        "object_label": 2,
                        "Children_Nuclei_Count": 1,
                    },
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            ),
        ),
        (
            "Nuclei_relationship_measurements",
            MeasurementTable(
                name="Nuclei_relationship_measurements",
                rows=(
                    {
                        "object_name": "Nuclei",
                        "object_label": 1,
                        "Parent_Cells": 1,
                    },
                    {
                        "object_name": "Nuclei",
                        "object_label": 2,
                        "Parent_Cells": 1,
                    },
                    {
                        "object_name": "Nuclei",
                        "object_label": 3,
                        "Parent_Cells": 2,
                    },
                ),
                subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
            ),
        ),
    ):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=artifact_name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/{artifact_name}.pkl",
            backend="memory",
        )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=relationship.name,
                kind=ArtifactKind.RELATIONSHIPS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=relationship.runtime_payload(),
            schema=relationship.runtime_schema(relationship.runtime_payload()),
        ),
        path="/memory/Cells_Nuclei_relationships.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_source_image_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Intensity_MeanIntensity_CropBlue\n"
        "1,1,0.25\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_label": 1,
                "mean_intensity": 0.25,
                "object_name": "Cells",
                "source_image_name": "CropBlue",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectIntensity",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_does_not_parse_object_mean_fields_as_aggregates(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Intensity_MeanIntensityEdge_CropBlue\n"
        "1,1,0.25\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=({"object_label": 1, "mean_intensity_edge": 0.25},),
        object_name="Nuclei",
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectIntensity",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_neighbor_distance_qualifiers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Neighbors_NumberOfNeighbors_4\n"
        "1,1,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectNeighbors",
        rows=({"object_id": 1, "scale": 4, "number_of_neighbors": 2},),
        object_name="Nuclei",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectNeighbors",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectNeighbors.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_named_math_results(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Math_Ratio\n"
        "1,1,0.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="CalculateMath",
        rows=(
            {
                "object_label": 1,
                "object_name": "Nuclei",
                "feature_name": "Math_Ratio",
                "result_value": 0.5,
            },
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="CalculateMath",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/CalculateMath.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_multi_source_image_measurements(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_Correlation_Stain2_Stain1\n"
        "1,0.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=({"correlation": 0.5},),
        source_image_name="Stain1__Stain2",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_reversed_pair_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_K_Stain2_Stain1\n"
        "1,0.25\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=({"k2": 0.25},),
        source_image_name="Stain1__Stain2",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_colocalization_correlation_and_overlap_orientation(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_Correlation_Stain2_Stain1,"
        "Correlation_Overlap_Stain2_Stain1\n"
        "1,0.5,0.9\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=({"correlation": 0.5, "overlap": 0.9},),
        source_image_name="Stain1__Stain2",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_area_occupied_owner_suffixes(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,AreaOccupied_AreaOccupied_Nuclei,"
        "AreaOccupied_Perimeter_Nuclei,AreaOccupied_TotalArea_Nuclei\n"
        "1,6,10,30\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureImageAreaOccupied",
        rows=(
            {
                "area_occupied": 6,
                "perimeter": 10,
                "total_area": 30,
                "source_image_name": "Nuclei",
            },
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureImageAreaOccupied",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureImageAreaOccupied.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_scopes_aggregate_math_to_image(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Math_PercentPositive\n"
        "1,6.428571428571428\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="CalculateMath",
        object_name="Nuclei",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": "",
                "feature_name": "Math_PercentPositive",
                "result_value": 6.428571428571428,
            },
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="CalculateMath",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/CalculateMath.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_image_source_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,ImageQuality_FocusScore_OrigBlue\n"
        "1,0.75\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureImageQuality",
        rows=({"focus_score": 0.75,},),
        source_image_name="OrigBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureImageQuality",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureImageQuality.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_preserves_row_source_qualified_image_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,ImageQuality_FocusScore_OrigBlue,"
        "ImageQuality_FocusScore_OrigGreen\n"
        "1,0.75,0.5\n",
        encoding="utf-8",
    )
    table = MeasurementTable(
        name="MeasureImageQuality",
        rows=(
            {
                "image_number": 1,
                "focus_score": 0.75,
                "source_image_name": "OrigBlue",
            },
            {
                "image_number": 1,
                "focus_score": 0.5,
                "source_image_name": "OrigGreen",
            },
        ),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureImageQuality",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/MeasureImageQuality.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_qualified_neighbor_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Nuclei.csv").write_text(
        "ImageNumber,ObjectNumber,Neighbors_NumberOfNeighbors_4\n"
        "1,1,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectNeighbors",
        rows=(
            {
                "object_id": 1,
                "object_name": "Nuclei",
                "scale": 4,
                "number_of_neighbors": 2,
            },
        ),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectNeighbors",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectNeighbors.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_preserves_qualified_correlation_texture_feature(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Texture_Correlation_CropBlue_3_00_256\n"
        "1,0.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureTexture",
        rows=(
            {
                "slice_index": 0,
                "scale": 3,
                "direction": 0,
                "gray_levels": 256,
                "correlation": 0.5,
            },
        ),
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureTexture",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureTexture.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_ignores_image_provenance_fields(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,MD5Digest_OrigBlue,Scaling_OrigBlue,ImageQuality_Scaling_OrigBlue,"
        "ImageQuality_FocusScore_OrigBlue\n"
        "1,abcdef,1.0,255,0.75\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureImageQuality",
        rows=({"focus_score": 0.75},),
        source_image_name="OrigBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureImageQuality",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureImageQuality.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_crop_feature_aliases(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Crop_AreaRetainedAfterCropping_CropBlue,"
        "Crop_OriginalImageArea_CropBlue\n"
        "1,25,100\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="Crop",
        rows=({"area_retained": 25, "original_area": 100},),
        source_image_name="CropBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Crop",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/Crop.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_keeps_crop_original_area_semantic(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Crop_AreaRetainedAfterCropping_Cropped,"
        "Crop_OriginalImageArea_Cropped\n"
        "1,25,100\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Original",
                kind=ArtifactKind.IMAGE,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.ones((10, 10), dtype=np.float32),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.IMAGE,
                source_image_name="Original",
            ),
        ),
        path="/memory/Original.pkl",
        backend="memory",
    )
    native_table = MeasurementTable(
        name="Crop",
        rows=({"area_retained": 25, "original_area": 100},),
        source_image_name="Cropped",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Crop",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/Crop.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_image_quality_qualifiers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,ImageQuality_LocalFocusScore_OrigBlue_20,"
        "ImageQuality_Correlation_OrigBlue_20,"
        "ImageQuality_ThresholdOtsu_OrigBlue_2W\n"
        "1,0.5,0.25,0.75\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureImageQuality",
        rows=(
            {
                "local_focus_score": 0.5,
                "correlation": 0.25,
                "threshold_otsu": 0.75,
            },
        ),
        source_image_name="OrigBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureImageQuality",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureImageQuality.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_detects_unmatched_scale_counts(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,ImageQuality_Correlation_OrigBlue_20,"
        "ImageQuality_Correlation_OrigBlue_40\n"
        "1,0.25,0.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureImageQuality",
        rows=({"correlation": 0.25},),
        source_image_name="OrigBlue",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureImageQuality",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureImageQuality.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.failure_messages() == (
        "measurement feature image:image/correlation_orig_blue values differ",
    )


def test_runtime_reference_artifact_equivalence_matches_directional_pair_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_K_CropBlue_CropGreen,"
        "Correlation_K_CropGreen_CropBlue,"
        "Correlation_Manders_CropBlue_CropGreen,"
        "Correlation_Manders_CropGreen_CropBlue,"
        "Correlation_RWC_CropBlue_CropGreen,"
        "Correlation_RWC_CropGreen_CropBlue,"
        "Correlation_Costes_CropBlue_CropGreen,"
        "Correlation_Costes_CropGreen_CropBlue\n"
        "1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=(
            {
                "k1": 0.1,
                "k2": 0.2,
                "manders_m1": 0.3,
                "manders_m2": 0.4,
                "rwc1": 0.5,
                "rwc2": 0.6,
                "costes_m1": 0.7,
                "costes_m2": 0.8,
            },
        ),
        source_image_name="CropBlue__CropGreen",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_undirected_pair_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_Correlation_Stain2_Stain1,"
        "Correlation_Overlap_Stain2_Stain1\n"
        "1,0.1,0.2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=(
            {
                "correlation": 0.1,
                "overlap": 0.2,
            },
        ),
        source_image_name="Stain1__Stain2",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_reversed_regression_slope(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_Correlation_Stain2_Stain1,"
        "Correlation_Slope_Stain2_Stain1\n"
        "1,0.5,0.125\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=(
            {
                "correlation": 0.5,
                "slope": 2.0,
            },
        ),
        source_image_name="Stain1__Stain2",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_cellprofiler_policy_tolerates_regression_slope_roundoff() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
    feature = RuntimeMeasurementFeatureKey(subject, "slope", "value")
    reference = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(
                        RuntimeCellValueKind.NUMBER,
                        "0.3683298538",
                    ): 1
                }
            )
        }
    )
    candidate = RuntimeMeasurementSnapshot(
        {
            feature: Counter(
                {
                    RuntimeCellSignature(
                        RuntimeCellValueKind.NUMBER,
                        "0.3683313118",
                    ): 1
                }
            )
        }
    )

    report = runtime_measurement_equivalence(
        reference,
        candidate,
        policy=cellprofiler_runtime_equivalence_policy(),
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_allows_threshold_sensitive_pair_drift(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Correlation_Costes_Stain1_Stain2,"
        "Correlation_Costes_Stain2_Stain1,"
        "Correlation_Manders_Stain1_Stain2,"
        "Correlation_Manders_Stain2_Stain1,"
        "Correlation_RWC_Stain1_Stain2,"
        "Correlation_RWC_Stain2_Stain1,"
        "Correlation_K_Stain1_Stain2,"
        "Correlation_K_Stain2_Stain1\n"
        "1,0.207,0.588,0.407,0.688,0.307,0.788,1.207,1.588\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureColocalization",
        rows=(
            {
                "costes_m1": 0.223,
                "costes_m2": 0.572,
                "manders_m1": 0.423,
                "manders_m2": 0.672,
                "rwc1": 0.323,
                "rwc2": 0.772,
                "k1": 1.223,
                "k2": 1.572,
            },
        ),
        source_image_name="Stain1__Stain2",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureColocalization",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureColocalization.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    strict_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )
    threshold_pair_report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(
            threshold_sensitive_pair_abs_tolerance=0.025,
        ),
    )

    assert strict_report.failure_messages() == (
        "measurement feature image:image/costes_stain_1_stain_2 values differ",
        "measurement feature image:image/costes_stain_2_stain_1 values differ",
        "measurement feature image:image/k_stain_1_stain_2 values differ",
        "measurement feature image:image/k_stain_2_stain_1 values differ",
        "measurement feature image:image/manders_stain_1_stain_2 values differ",
        "measurement feature image:image/manders_stain_2_stain_1 values differ",
        "measurement feature image:image/rwc_stain_1_stain_2 values differ",
        "measurement feature image:image/rwc_stain_2_stain_1 values differ",
    )
    assert threshold_pair_report.is_equivalent


def test_runtime_reference_artifact_equivalence_matches_object_label_counts(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Count_Cells\n"
        "1,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array([[0, 1], [2, 2]], dtype=np.uint16),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_measurement_snapshot_derives_required_object_numbers_from_labels(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="PH3",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array([[0, 1], [2, 2]], dtype=np.uint16),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="PH3",
            ),
        ),
        path="/memory/PH3.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "PH3")
    required_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({required_key}),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[required_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
    }


def test_runtime_measurement_snapshot_completes_repeated_declared_object_domains(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "GridObjects")
    required_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_x_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )
    for index in range(2):
        store.replace(
            RuntimeValue(
                key=ArtifactKey(
                    name="GridObjects",
                    kind=ArtifactKind.OBJECT_LABELS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=ObjectLabelPayload(
                    labels=np.array([[0, 1], [0, 3]], dtype=np.uint16),
                    declared_object_count=4,
                    domain_scope=ObjectLabelDomainScope.PLANE,
                ),
                schema=RuntimeValueSchema(
                    kind=ArtifactKind.OBJECT_LABELS,
                ),
            ),
            path=f"/memory/GridObjects_{index}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({required_key, center_x_key}),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[required_key].items()
    } == {
        ("number", "1.0", 2),
        ("number", "2.0", 2),
        ("number", "3.0", 2),
        ("number", "4.0", 2),
    }


def test_runtime_measurement_snapshot_uses_declared_plane_object_domains(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    required_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_x_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [[1, 0], [0, 2]],
                        [[1, 0], [3, 4]],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
                domain_scope=ObjectLabelDomainScope.PLANE,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({required_key, center_x_key}),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[required_key].items()
    } == {
        ("number", "1.0", 2),
        ("number", "2.0", 2),
        ("number", "3.0", 1),
        ("number", "4.0", 1),
    }
    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[center_x_key].items()
    } == {
        ("number", "0.0", 3),
        ("number", "1.0", 2),
    }


def test_runtime_measurement_snapshot_collapses_repeated_diagonal_plane_domains(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "SSC")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_x_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )
    plane = np.array([[1, 0], [0, 2]], dtype=np.uint16)
    labels = np.zeros((2, 2, 2, 2), dtype=np.uint16)
    labels[0, 0] = plane
    labels[1, 1] = plane
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="SSC",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=labels,
                declared_object_id_domains=((1, 2), (1, 2)),
                domain_scope=ObjectLabelDomainScope.PLANE,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="SSC",
            ),
        ),
        path="/memory/SSC.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({object_number_key, center_x_key}),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_number_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
    }
    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[center_x_key].items()
    } == {
        ("number", "0.0", 1),
        ("number", "1.0", 1),
    }


def test_runtime_measurement_snapshot_skips_multi_source_measurement_domain_object_numbers(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "SSC")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    labels = np.zeros((2, 2, 2, 2), dtype=np.uint16)
    labels[0, 0] = np.array([[1, 0], [0, 2]], dtype=np.uint16)
    labels[1, 1] = np.array([[3, 0], [0, 4]], dtype=np.uint16)
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="SSC",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=labels,
                declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
                domain_scope=ObjectLabelDomainScope.PLANE,
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                source_image_names=("BF_image", "MorphBf"),
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="SSC",
            ),
        ),
        path="/memory/SSC.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({object_number_key}),
    )

    assert object_number_key not in snapshot.values_by_feature


def test_runtime_measurement_snapshot_skips_row_sequence_object_numbers(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    table = MeasurementTable(
        name="MeasureTexture",
        rows=(
            {
                "object_label": 1,
                "object_name": "Cells",
                "scale": 3,
                "direction": 0,
                "gray_levels": 256,
                "angular_second_moment": 0.25,
                MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD: (
                    MeasurementObjectRowIdentity.ROW_SEQUENCE.value
                ),
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureTexture",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/MeasureTexture.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({object_number_key}),
    )

    assert object_number_key not in snapshot.values_by_feature


def test_runtime_measurement_snapshot_collapses_repeated_homogeneous_plane_domains(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "SSC")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_x_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )
    plane = np.array([[1, 0], [0, 2]], dtype=np.uint16)
    labels = np.stack((plane, plane), axis=0)
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="SSC",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=labels,
                declared_object_id_domains=((1, 2), (1, 2)),
                domain_scope=ObjectLabelDomainScope.PLANE,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="SSC",
            ),
        ),
        path="/memory/SSC.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({object_number_key, center_x_key}),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_number_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
    }
    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[center_x_key].items()
    } == {
        ("number", "0.0", 1),
        ("number", "1.0", 1),
    }


def test_runtime_measurement_snapshot_suppresses_label_geometry_mean_when_locations_are_measured(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells")
    mean_center_x_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
        MeasurementStatistic.MEAN.value,
    )
    measurement_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "image_number": 1,
                "object_name": "Cells",
                "object_label": 1,
                "center_x": 3.0,
            },
            {
                "image_number": 1,
                "object_name": "Cells",
                "object_label": 2,
                "center_x": 5.0,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=measurement_table.rows,
            schema=measurement_table.runtime_schema(measurement_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array([[1, 2], [0, 0]], dtype=np.uint16),
                domain_scope=ObjectLabelDomainScope.PLANE,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({mean_center_x_key}),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[mean_center_x_key].items()
    } == {("number", "4.0", 1)}


def test_runtime_measurement_snapshot_projects_identity_only_object_exports(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    (reference_root / "StraightenedWorms.csv").write_text(
        "ImageNumber,ObjectNumber\n1,1\n1,2\n1,3\n",
        encoding="utf-8",
    )
    subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        "StraightenedWorms",
    )
    key = RuntimeMeasurementFeatureKey(subject, "object_number")

    snapshot = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
        ("number", "3.0", 1),
    }


def test_runtime_measurement_snapshot_deduplicates_aggregate_group_tables(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    table = MeasurementTable(
        name="AggregateMeasurements",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Area",
                "result_value": 10,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Area",
                "result_value": 20,
            },
        ),
    )
    store = RuntimeValueStore()
    for group_key in ("1", "2"):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name="AggregateMeasurements",
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01", group_key=group_key),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/AggregateMeasurements_{group_key}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei")
    area_key = RuntimeMeasurementFeatureKey(subject, "area")

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[area_key].items()
    } == {
        ("number", "10.0", 1),
        ("number", "20.0", 1),
    }


def test_runtime_measurement_snapshot_preserves_long_form_aggregate_row_axes(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    table = MeasurementTable(
        name="AggregateMeasurements",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Phase",
                "result_value": 1.5707963268,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Phase",
                "result_value": 1.5707963268,
            },
        ),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="AggregateMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/AggregateMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    phase_key = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei"),
        "phase",
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert snapshot.values_by_feature[phase_key] == {
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.5707963268"): 2,
    }


def test_runtime_measurement_table_aggregate_row_dedupe_preserves_wide_row_axes() -> None:
    table = MeasurementTable(
        name="MeasureObjectIntensityDistribution",
        rows=(
            {
                "slice_index": 0,
                "object_name": "SSC",
                "object_label": 1,
                "zernike_phase_0_0": 0.0,
                "zernike_magnitude_0_0": 0.0,
            },
            {
                "slice_index": 1,
                "object_name": "SSC",
                "object_label": 1,
                "zernike_phase_0_0": 0.0,
                "zernike_magnitude_0_0": 0.0,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "SSC"),
    )

    deduped = _dedupe_runtime_measurement_table_aggregate_rows(table)

    assert tuple(deduped.rows) == table.rows


def test_runtime_measurement_table_aggregate_row_dedupe_preserves_object_row_counts() -> None:
    table = MeasurementTable(
        name="MeasureObjectIntensityDistribution",
        rows=(
            {
                "object_name": "SSC",
                "object_label": 1,
                "feature_name": "IntensityDistribution_ZernikePhase_0_0",
                "result_value": 1.5707963268,
                "image_number": 1,
                "slice_index": 0,
            },
            {
                "object_name": "SSC",
                "object_label": 1,
                "feature_name": "IntensityDistribution_ZernikePhase_0_0",
                "result_value": 1.5707963268,
                "image_number": 1,
                "slice_index": 0,
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "SSC"),
    )

    deduped = _dedupe_runtime_measurement_table_aggregate_rows(table)

    assert tuple(deduped.rows) == table.rows


def test_runtime_measurement_table_dedupe_identity_uses_row_fingerprint() -> None:
    table = MeasurementTable(
        name="Measurements",
        rows=(
            {"image_number": 1, "feature_name": "Area", "result_value": 10},
            {"image_number": 2, "feature_name": "Area", "result_value": 20},
        ),
    )
    equivalent_table = MeasurementTable(
        name="Measurements",
        rows=(
            {"image_number": 1, "feature_name": "Area", "result_value": 10},
            {"image_number": 2, "feature_name": "Area", "result_value": 20},
        ),
    )

    identity = exact_measurement_table_key(table)

    assert identity == exact_measurement_table_key(equivalent_table)
    assert isinstance(identity.rows, RuntimeMeasurementRowFingerprint)
    assert identity.rows.row_count == 2


def test_runtime_measurement_equivalence_normalizes_long_form_image_number_features(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    sequence_root = reference_root / "Sequence2"
    sequence_root.mkdir(parents=True)
    candidate_root.mkdir()
    (sequence_root / "Embryos.csv").write_text(
        "ImageNumber,ObjectNumber,TrackObjects_ParentImageNumber_50\n"
        "22,1,0\n"
        "23,1,22\n",
        encoding="utf-8",
    )
    table = MeasurementTable(
        name="TrackObjects",
        rows=(
            {
                "image_number": 22,
                "object_label": 1,
                "feature_name": "TrackObjects_ParentImageNumber_50",
                "measurement_value": 0,
            },
            {
                "image_number": 23,
                "object_label": 1,
                "feature_name": "TrackObjects_ParentImageNumber_50",
                "measurement_value": 22,
            },
        ),
        object_name="Embryos",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="TrackObjects",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="Sequence2"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/TrackObjects.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"Sequence2": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    policy = RuntimeEquivalencePolicy()
    reference = RuntimeMeasurementSnapshot.from_output_snapshot(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        policy=policy,
    )
    candidate = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=policy,
        required_measurement_keys=frozenset(reference.values_by_feature),
    )

    report = runtime_measurement_equivalence(reference, candidate, policy=policy)

    assert report.is_equivalent


def test_runtime_measurement_snapshot_preserves_group_local_duplicate_rows(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    table = MeasurementTable(
        name="LocalMeasurements",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Area",
                "result_value": 10,
            },
        ),
    )
    store = RuntimeValueStore()
    for group_key in ("1", "2"):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name="LocalMeasurements",
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01", group_key=group_key),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/LocalMeasurements_{group_key}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei")
    area_key = RuntimeMeasurementFeatureKey(subject, "area")

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[area_key].items()
    } == {
        ("number", "10.0", 2),
    }


def test_runtime_measurement_snapshot_encodes_object_source_as_feature_suffix(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    table = MeasurementTable(
        name="MeasureObjectIntensity",
        rows=(
            {
                "object_name": "Cells",
                "object_label": 1,
                "mean_intensity": 0.25,
            },
        ),
        source_image_name="RawGFP",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectIntensity",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/MeasureObjectIntensity.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    required_key = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells"),
        "mean_intensity_raw_gfp",
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset({required_key}),
    )

    assert (
        snapshot.values_by_feature[required_key][
            RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "0.25")
        ]
        == 1
    )


def test_runtime_measurement_snapshot_derives_means_per_row_image_identity(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Area",
                "result_value": 10,
            },
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 2,
                "feature_name": "Area",
                "result_value": 20,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 1,
                "feature_name": "Area",
                "result_value": 100,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 2,
                "feature_name": "Area",
                "result_value": 300,
            },
        ),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="ObjectMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/ObjectMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei")
    mean_area_key = RuntimeMeasurementFeatureKey(subject, "area", "mean")

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[mean_area_key].items()
    } == {
        ("number", "15.0", 1),
        ("number", "200.0", 1),
    }


def test_runtime_measurement_snapshot_derives_wide_means_per_row_image_identity(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=(
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 1,
                "Area": 10,
            },
            {
                "slice_index": 0,
                "object_name": "Nuclei",
                "object_label": 2,
                "Area": 20,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 1,
                "Area": 100,
            },
            {
                "slice_index": 1,
                "object_name": "Nuclei",
                "object_label": 2,
                "Area": 300,
            },
        ),
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="ObjectMeasurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=table.rows,
            schema=table.runtime_schema(table.rows),
        ),
        path="/memory/ObjectMeasurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei")
    mean_area_key = RuntimeMeasurementFeatureKey(subject, "area", "mean")

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[mean_area_key].items()
    } == {
        ("number", "15.0", 1),
        ("number", "200.0", 1),
    }


def test_runtime_measurement_snapshot_derives_means_per_axis_local_image_identity(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Nuclei")
    mean_area_key = RuntimeMeasurementFeatureKey(subject, "area", "mean")

    def store_for_axis(area_a: int, area_b: int) -> SimpleNamespace:
        table = MeasurementTable(
            name="ObjectMeasurements",
            rows=(
                {
                    "slice_index": 0,
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "Area": area_a,
                },
                {
                    "slice_index": 0,
                    "object_name": "Nuclei",
                    "object_label": 2,
                    "Area": area_b,
                },
            ),
        )
        store = RuntimeValueStore()
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name="ObjectMeasurements",
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="axis"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path="/memory/ObjectMeasurements.pkl",
            backend="memory",
        )
        return SimpleNamespace(runtime_value_store=store)

    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {
            "A01": store_for_axis(10, 20),
            "A02": store_for_axis(100, 300),
        },
        candidate_root,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[mean_area_key].items()
    } == {
        ("number", "15.0", 1),
        ("number", "200.0", 1),
    }


def test_runtime_reference_artifact_equivalence_preserves_count_object_digits(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Count_PH3\n"
        "1,1\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="PH3",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array([[0, 1], [0, 0]], dtype=np.uint16),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="PH3",
            ),
        ),
        path="/memory/PH3.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_object_label_centers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "PH3.csv").write_text(
        "ImageNumber,ObjectNumber,Location_Center_X,Location_Center_Y,"
        "Location_Center_Z,Number_Object_Number\n"
        "1,1,1.5,0.0,0.0,1\n"
        "1,2,0.5,2.0,0.0,2\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="PH3",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array(
                [
                    [0, 1, 1],
                    [0, 0, 0],
                    [2, 2, 0],
                ],
                dtype=np.uint16,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="PH3",
            ),
        ),
        path="/memory/PH3.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_label_center_means(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Mean_Cells_Location_Center_X,"
        "Mean_Cells_Location_Center_Y,Mean_Cells_Location_Center_Z\n"
        "1,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array(
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 2, 2],
                ],
                dtype=np.uint16,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_declared_label_centers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Location_Center_X,Location_Center_Y,"
        "Location_Center_Z\n"
        "1,1,0.5,0.0,0.0\n"
        "1,2,nan,nan,0.0\n"
        "1,3,nan,nan,0.0\n"
        "1,4,0.5,2.0,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
                data=ObjectLabelPayload(
                    labels=np.array(
                        [
                            [1, 1, 0],
                            [0, 0, 0],
                            [4, 4, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    declared_object_count=4,
                ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_uses_declared_label_count(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Count_Cells\n"
        "1,4\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [1, 1, 0],
                        [0, 0, 0],
                        [4, 4, 0],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_count=4,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_measurement_snapshot_declared_object_domain_owns_count_before_row_fallback(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells_measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=[
                {"image_number": 1, "object_label": 1, "Center_X": 0.0},
                {"image_number": 1, "object_label": 2, "Center_X": 1.0},
                {"image_number": 1, "object_label": 4, "Center_X": 2.0},
            ],
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells_measurements.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [1, 1, 0],
                        [0, 0, 0],
                        [4, 4, 0],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_count=4,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    object_count_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
        MeasurementStatistic.COUNT.value,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_count_key].items()
    } == {("number", "4.0", 1)}


def test_runtime_measurement_snapshot_row_source_identity_does_not_qualify_image_subject_feature(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="TrackObjects_measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=[
                {
                    "image_number": 1,
                    "feature_name": "TrackObjects_NewObjectCount_Embryos_50",
                    "measurement_value": 4,
                    "source_image_name": "image",
                }
            ],
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                object_name="Embryos",
            ),
        ),
        path="/memory/TrackObjects_measurements.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    key = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "image"),
        "track_objects_new_object_count_embryos_50",
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=cellprofiler_runtime_equivalence_policy(),
        known_source_names=("OrigColor",),
        required_measurement_keys=frozenset({key}),
    )

    assert snapshot.values_by_feature[key] == {
        RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "4.0"): 1
    }


def test_runtime_measurement_snapshot_completes_object_numbers_from_primary_rows(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells_measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=[
                {"image_number": 1, "object_label": 1, "Center_X": 0.0},
                {"image_number": 1, "object_label": 2, "Center_X": 1.0},
                {"image_number": 1, "object_label": 4, "Center_X": 2.0},
            ],
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells_measurements.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [1, 1, 0],
                        [0, 0, 0],
                        [4, 4, 0],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_ids=(1, 2, 4),
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
    )
    center_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset((object_number_key, center_key)),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_number_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
        ("number", "4.0", 1),
    }


def test_runtime_measurement_snapshot_does_not_complete_explicit_object_numbers(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells_measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=[
                {
                    "image_number": 1,
                    "object_label": 1,
                    "Number_Object_Number": 1,
                    "Center_X": 0.0,
                },
                {
                    "image_number": 1,
                    "object_label": 2,
                    "Number_Object_Number": 2,
                    "Center_X": 1.0,
                },
            ],
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells_measurements.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [1, 1, 0],
                        [0, 2, 2],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_count=2,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset((object_number_key, center_key)),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_number_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
    }


def test_runtime_measurement_snapshot_completes_partial_explicit_object_numbers(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells_measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=[
                {
                    "image_number": 1,
                    "object_label": 1,
                    "Number_Object_Number": 1,
                    "Center_X": 0.0,
                },
                {"image_number": 1, "object_label": 2, "Center_X": 1.0},
            ],
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells_measurements.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [1, 1, 0],
                        [0, 2, 2],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_count=2,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset((object_number_key, center_key)),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_number_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
    }


def test_runtime_measurement_snapshot_rejects_axis_only_rows_as_identifier_domain(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells_measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=[
                {"object_label": 1, "Center_X": 0.0},
                {"object_label": 2, "Center_X": 1.0},
                {"object_label": 3, "Center_X": 2.0},
            ],
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells_measurements.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=ObjectLabelPayload(
                labels=np.array(
                    [
                        [1, 1, 0],
                        [0, 2, 2],
                    ],
                    dtype=np.uint16,
                ),
                declared_object_count=2,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "cells")
    object_number_key = RuntimeMeasurementFeatureKey(
        subject,
        "number_object_number",
    )
    center_key = RuntimeMeasurementFeatureKey(
        subject,
        ObjectCoreMeasurementFeature.CENTER_X.value,
    )

    snapshot = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        observation,
        policy=RuntimeEquivalencePolicy(),
        required_measurement_keys=frozenset((object_number_key, center_key)),
    )

    assert {
        (signature.kind.value, signature.value, count)
        for signature, count in snapshot.values_by_feature[object_number_key].items()
    } == {
        ("number", "1.0", 1),
        ("number", "2.0", 1),
    }


def test_runtime_reference_artifact_equivalence_does_not_duplicate_explicit_centers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Location_Center_X,Location_Center_Y,"
        "Location_Center_Z\n"
        "1,1,1.5,0.0,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    explicit_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "Center_X": 1.5,
                "Center_Y": 0.0,
                "Center_Z": 0.0,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=explicit_table.rows,
            schema=explicit_table.runtime_schema(explicit_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array([[0, 1, 1]], dtype=np.uint16),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_derives_label_center_means_with_explicit_centers(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Mean_Cells_Location_Center_X,"
        "Mean_Cells_Location_Center_Y,Mean_Cells_Location_Center_Z\n"
        "1,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    explicit_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "Center_X": 0.5,
                "Center_Y": 0.0,
                "Center_Z": 0.0,
                "object_name": "Cells",
            },
            {
                "object_label": 2,
                "Center_X": 1.5,
                "Center_Y": 2.0,
                "Center_Z": 0.0,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=explicit_table.rows,
            schema=explicit_table.runtime_schema(explicit_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Cells",
                kind=ArtifactKind.OBJECT_LABELS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=np.array(
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 2, 2],
                ],
                dtype=np.uint16,
            ),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                object_name="Cells",
            ),
        ),
        path="/memory/Cells.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_collapses_same_row_cp_aliases(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Center_X,Location_Center_X\n"
        "1,1,7.5,7.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "Center_X": 7.5,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_prefers_primary_same_row_alias(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Center_X,Location_Center_X\n"
        "1,1,7.5,8.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    native_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "object_label": 1,
                "Center_X": 7.5,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="MeasureObjectSizeShape",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=native_table.rows,
            schema=native_table.runtime_schema(native_table.rows),
        ),
        path="/memory/MeasureObjectSizeShape.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_merges_object_location_alias_rows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,AreaShape_Center_X,Location_Center_X\n"
        "1,1,7.5,8.5\n"
        "1,2,nan,9.5\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    size_shape_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "image_number": 1,
                "object_label": 1,
                "Center_X": 7.5,
                "object_name": "Cells",
            },
            {
                "image_number": 1,
                "object_label": 2,
                "Center_X": np.nan,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    location_table = MeasurementTable(
        name="IdentifyObjects",
        rows=(
            {
                "image_number": 1,
                "object_label": 1,
                "feature_name": "Location_Center_X",
                "result_value": 8.5,
                "object_name": "Cells",
            },
            {
                "image_number": 1,
                "object_label": 2,
                "feature_name": "Location_Center_X",
                "result_value": 9.5,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    for table in (location_table, size_shape_table):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/{table.name}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_uses_nominal_image_identity_dominance_for_aggregates(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Mean_Cells_Location_Center_X\n"
        "1,9.0\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    size_shape_table = MeasurementTable(
        name="MeasureObjectSizeShape",
        rows=(
            {
                "image_number": 1,
                "slice_index": 0,
                "object_label": 1,
                "Center_X": 8.5,
                "object_name": "Cells",
            },
            {
                "image_number": 1,
                "slice_index": 0,
                "object_label": 2,
                "Center_X": 9.5,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    location_table = MeasurementTable(
        name="IdentifyObjects",
        rows=(
            {
                "image_number": 1,
                "object_label": 1,
                "feature_name": "Location_Center_X",
                "result_value": 8.5,
                "object_name": "Cells",
            },
            {
                "image_number": 1,
                "object_label": 2,
                "feature_name": "Location_Center_X",
                "result_value": 9.5,
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    for table in (location_table, size_shape_table):
        store.record(
            RuntimeValue(
                key=ArtifactKey(
                    name=table.name,
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            path=f"/memory/{table.name}.pkl",
            backend="memory",
        )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        candidate_root,
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_ignores_duplicate_measurement_artifacts(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Texture_Entropy_DNA_3_00_256\n"
        "1,1,0.25\n",
        encoding="utf-8",
    )
    table = MeasurementTable(
        name="MeasureTexture",
        rows=(
            {
                "image_number": 1,
                "object_label": 1,
                "scale": 3,
                "direction": 0,
                "gray_levels": 256,
                "entropy": 0.25,
                "source_image_name": "DNA",
                "object_name": "Cells",
            },
        ),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "DNA"),
    )
    records = tuple(
        StoredRuntimeValue(
            RuntimeValue(
                key=ArtifactKey(
                    name="MeasureTexture",
                    kind=ArtifactKind.MEASUREMENTS,
                    scope=ArtifactScope(axis_id="A01"),
                ),
                data=table.rows,
                schema=table.runtime_schema(table.rows),
            ),
            RuntimeArtifactLocation(
                path=f"/memory/MeasureTexture_{index}.pkl",
                backend="memory",
            ),
        )
        for index in (1, 2)
    )
    observation = RuntimeArtifactExecutionObservation(
        {"A01": records},
        RuntimeExportObservation.from_output_root(candidate_root),
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_ignores_duplicate_aggregate_tables_across_axes(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        "ImageNumber,Threshold_FinalThreshold_Tissue\n"
        "1,0.25\n"
        "2,0.75\n",
        encoding="utf-8",
    )
    table = MeasurementTable(
        name="IdentifyPrimaryObjects",
        rows=(
            {"image_number": 1, "final_threshold": 0.25, "object_name": "Tissue"},
            {"image_number": 2, "final_threshold": 0.75, "object_name": "Tissue"},
            {"image_number": 3, "final_threshold": 0.25, "object_name": "Tissue"},
            {"image_number": 4, "final_threshold": 0.75, "object_name": "Tissue"},
        ),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Tissue"),
    )
    records_by_axis = {
        axis: (
            StoredRuntimeValue(
                RuntimeValue(
                    key=ArtifactKey(
                        name="IdentifyPrimaryObjects",
                        kind=ArtifactKind.MEASUREMENTS,
                        scope=ArtifactScope(axis_id=axis),
                    ),
                    data=table.rows,
                    schema=table.runtime_schema(table.rows),
                ),
                RuntimeArtifactLocation(
                    path=f"/memory/IdentifyPrimaryObjects_{axis}.pkl",
                    backend="memory",
                ),
            ),
        )
        for axis in ("A01", "A02")
    }
    observation = RuntimeArtifactExecutionObservation(
        records_by_axis,
        RuntimeExportObservation.from_output_root(candidate_root),
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent


def test_runtime_reference_artifact_equivalence_ignores_duplicate_object_rows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Cells.csv").write_text(
        "ImageNumber,ObjectNumber,Texture_Entropy_DNA_3_00_256\n"
        "1,1,0.25\n",
        encoding="utf-8",
    )
    table_rows = (
        {
            "image_number": 1,
            "scale": 3,
            "direction": 0,
            "gray_levels": 256,
            "entropy": 0.75,
            "source_image_name": "DNA",
        },
        {
            "image_number": 1,
            "object_label": 1,
            "scale": 3,
            "direction": 0,
            "gray_levels": 256,
            "entropy": 0.25,
            "source_image_name": "DNA",
            "object_name": "Cells",
        },
    )
    records = []
    for index, image_entropy in ((1, 0.75), (2, 0.80)):
        rows = (
            {**table_rows[0], "entropy": image_entropy},
            table_rows[1],
        )
        table = MeasurementTable(
            name="MeasureTexture",
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.IMAGE, "DNA"),
        )
        records.append(
            StoredRuntimeValue(
                RuntimeValue(
                    key=ArtifactKey(
                        name="MeasureTexture",
                        kind=ArtifactKind.MEASUREMENTS,
                        scope=ArtifactScope(axis_id="A01"),
                    ),
                    data=table.rows,
                    schema=table.runtime_schema(table.rows),
                ),
                RuntimeArtifactLocation(
                    path=f"/memory/MeasureTexture_{index}.pkl",
                    backend="memory",
                ),
            )
        )
    observation = RuntimeArtifactExecutionObservation(
        {"A01": tuple(records)},
        RuntimeExportObservation.from_output_root(candidate_root),
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
    )

    assert report.is_equivalent
