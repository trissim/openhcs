from __future__ import annotations

import sqlite3
from pathlib import Path

from benchmark.cellprofiler_export_equivalence import (
    cellprofiler_database_export_equivalence,
)
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerDatabaseColumnDialect,
    CellProfilerImageAggregateStatistic,
    CellProfilerImageStructuralFieldFamily,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
    cellprofiler_runtime_equivalence_policy,
)


def _write_database(
    path: Path,
    *,
    value: float = 1.5,
    value_not_null: bool = False,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Per_Image ("
            "ImageNumber INTEGER PRIMARY KEY, "
            "Image_Intensity_TotalIntensity_DNA REAL"
            + (" NOT NULL" if value_not_null else "")
            + ")"
        )
        connection.execute("INSERT INTO Per_Image VALUES (?, ?)", (1, value))


def _write_prefixed_combined_database(path: Path, *, table_prefix: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            f'CREATE TABLE "{table_prefix}Per_Image" ('
            "ImageNumber INTEGER PRIMARY KEY, Image_Count_Nuclei INTEGER)"
        )
        connection.execute(
            f'INSERT INTO "{table_prefix}Per_Image" VALUES (1, 0)'
        )
        connection.execute(
            f'CREATE TABLE "{table_prefix}Per_Object" ('
            "ImageNumber INTEGER, ObjectNumber INTEGER, "
            "PRIMARY KEY (ImageNumber, ObjectNumber))"
        )


def _write_tie_sensitive_object_database(
    path: Path,
    *,
    location: float,
    intensity: float,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Per_Cytoplasm ("
            "ImageNumber INTEGER, "
            "Cytoplasm_Number_Object_Number INTEGER, "
            "Cytoplasm_Intensity_MaxIntensity_rawGFP REAL, "
            "Cytoplasm_Location_MaxIntensity_X_rawGFP REAL, "
            "PRIMARY KEY (ImageNumber, Cytoplasm_Number_Object_Number))"
        )
        connection.execute(
            "INSERT INTO Per_Cytoplasm VALUES (?, ?, ?, ?)",
            (1, 1, intensity, location),
        )


def _write_tie_sensitive_image_database(
    path: Path,
    *,
    location: float,
    intensity: float,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Per_Image ("
            "ImageNumber INTEGER PRIMARY KEY, "
            "Image_Mean_Cytoplasm_Intensity_MaxIntensity_rawGFP REAL, "
            "Image_Mean_Cytoplasm_Location_MaxIntensity_X_rawGFP REAL)"
        )
        connection.execute(
            "INSERT INTO Per_Image VALUES (?, ?, ?)",
            (1, intensity, location),
        )


def _write_relationship_database(path: Path, *, target_object: int) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Per_Relationships ("
            "relationship_type_id INTEGER, image_number1 INTEGER, "
            "object_number1 INTEGER, image_number2 INTEGER, "
            "object_number2 INTEGER)"
        )
        connection.execute(
            "INSERT INTO Per_Relationships VALUES (?, ?, ?, ?, ?)",
            (1, 1, 1, 1, target_object),
        )


def _write_per_object_aggregate_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Per_Cells ("
            "ImageNumber INTEGER, "
            "Cells_Number_Object_Number INTEGER, "
            "Cells_Mean_Mitochondria_Number_Object_Number REAL, "
            "PRIMARY KEY (ImageNumber, Cells_Number_Object_Number))"
        )
        connection.execute("INSERT INTO Per_Cells VALUES (1, 1, 2.0)")


def _write_properties(
    path: Path,
    database_path: Path,
    *,
    image_table: str = "Per_Image",
    object_table: str = "Per_Cytoplasm",
    object_id: str = "Cytoplasm_Number_Object_Number",
    extra: str = "",
) -> None:
    path.write_text(
        "\n".join(
            (
                "# generated comment",
                "db_type = sqlite",
                f"db_sqlite_file = {database_path}",
                f"image_table = {image_table}",
                f"object_table = {object_table}",
                "image_id = ImageNumber",
                f"object_id = {object_id}",
                "image_names = DNA, RNA",
                extra,
            )
        ),
        encoding="utf-8",
    )


def _write_cpa_metadata(
    path: Path,
    *,
    sqlite_property: str,
    pipeline: bytes,
    timestamp: str,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Per_Experiment ("
            "Pipeline_Pipeline BLOB, CellProfiler_Version TEXT, "
            "Run_Timestamp TEXT, Modification_Timestamp TEXT)"
        )
        connection.execute(
            "INSERT INTO Per_Experiment VALUES (?, ?, ?, ?)",
            (pipeline, "4.2.8.1", timestamp, timestamp),
        )
        connection.execute(
            "CREATE TABLE Experiment_Properties ("
            "experiment_id INTEGER, object_name TEXT, field TEXT, value TEXT)"
        )
        connection.execute(
            "INSERT INTO Experiment_Properties VALUES (1, 'Object', "
            "'db_sqlite_file', ?)",
            (sqlite_property,),
        )


def test_database_export_equivalence_compares_sqlite_and_semantic_properties(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_database(reference / "analysis.db")
    _write_database(candidate / "analysis.db")
    _write_properties(
        reference / "analysis.properties",
        reference / "analysis.db",
    )
    _write_properties(
        candidate / "analysis.properties",
        candidate / "analysis.db",
    )

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert report.is_equivalent


def test_database_export_equivalence_resolves_prefixed_combined_cpa_schema(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    table_prefix = "BBBC022QC_"
    for output_root in (reference, candidate):
        database_path = output_root / "BBBC022QC.db"
        _write_prefixed_combined_database(
            database_path,
            table_prefix=table_prefix,
        )
        _write_properties(
            output_root / "BBBC022QC.properties",
            database_path,
            image_table=f"{table_prefix}Per_Image",
            object_table=f"{table_prefix}Per_Object",
            object_id="ObjectNumber",
        )

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert report.is_equivalent


def test_database_export_equivalence_uses_declared_database_and_run_metadata(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    reference_database = reference / "analysis.db"
    candidate_database = candidate / "analysis.db"
    _write_database(reference_database)
    _write_database(candidate_database)
    _write_database(reference / "stale.db", value=99.0)
    _write_properties(reference / "analysis.properties", reference_database)
    _write_properties(candidate / "analysis.properties", candidate_database)
    _write_cpa_metadata(
        reference_database,
        sqlite_property=str(reference_database),
        pipeline=b"native CellProfiler pipeline",
        timestamp="2026-01-01T00:00:00",
    )
    _write_cpa_metadata(
        candidate_database,
        sqlite_property=candidate_database.name,
        pipeline=b"",
        timestamp="",
    )

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert report.is_equivalent


def test_database_export_equivalence_reports_database_rows_and_property_keys(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_database(reference / "analysis.db", value=1.5)
    _write_database(candidate / "analysis.db", value=2.5)
    _write_properties(
        reference / "analysis.properties",
        reference / "analysis.db",
        extra="plate_type = 96",
    )
    _write_properties(
        candidate / "analysis.properties",
        candidate / "analysis.db",
    )

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    messages = report.failure_messages()
    assert any("values differ" in message for message in messages)
    assert any(
        "keys differ" in message and "plate_type" in message for message in messages
    )


def test_database_export_equivalence_reports_sqlite_nullability_drift(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_database(reference / "analysis.db")
    _write_database(candidate / "analysis.db", value_not_null=True)
    _write_properties(
        reference / "analysis.properties",
        reference / "analysis.db",
    )
    _write_properties(
        candidate / "analysis.properties",
        candidate / "analysis.db",
    )

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert any(
        "schema differs" in message and "Per_Image" in message
        for message in report.failure_messages()
    )


def test_database_export_equivalence_excludes_declared_structural_fields(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for root, diagnostic_columns in (
        (
            reference,
            ", Image_ExecutionTime_01Images float, Image_ModuleError_01Images INTEGER",
        ),
        (candidate, ""),
    ):
        database = root / "analysis.db"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE Per_Image ("
                "ImageNumber INTEGER PRIMARY KEY"
                f"{diagnostic_columns}, Image_Value REAL)"
            )
            columns = 4 if diagnostic_columns else 2
            connection.execute(
                f"INSERT INTO Per_Image VALUES ({', '.join('?' for _ in range(columns))})",
                (1, 0.5, 0, 1.5) if diagnostic_columns else (1, 1.5),
            )
        _write_properties(root / "analysis.properties", database)

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(
            measurement_dialect=CELLPROFILER_MEASUREMENT_DIALECT,
        ),
    )

    assert report.is_equivalent


def test_database_export_equivalence_excludes_structural_prefix_columns(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    with sqlite3.connect(reference / "analysis.db") as connection:
        connection.execute(
            "CREATE TABLE Per_Image (ImageNumber INTEGER PRIMARY KEY, "
            "Image_ExecutionTime_01ImagesExtra REAL, Image_Value REAL)"
        )
        connection.execute("INSERT INTO Per_Image VALUES (1, 0.5, 1.5)")
    with sqlite3.connect(candidate / "analysis.db") as connection:
        connection.execute(
            "CREATE TABLE Per_Image (ImageNumber INTEGER PRIMARY KEY, Image_Value REAL)"
        )
        connection.execute("INSERT INTO Per_Image VALUES (1, 1.5)")
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(
            measurement_dialect=CELLPROFILER_MEASUREMENT_DIALECT,
        ),
    )

    assert report.is_equivalent


def test_database_export_equivalence_keeps_genuine_measurement_columns_strict(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    with sqlite3.connect(reference / "analysis.db") as connection:
        connection.execute(
            "CREATE TABLE Per_Image (ImageNumber INTEGER PRIMARY KEY, "
            "Image_Intensity_TotalIntensity_DNA REAL, Image_Value REAL)"
        )
        connection.execute("INSERT INTO Per_Image VALUES (1, 0.5, 1.5)")
    with sqlite3.connect(candidate / "analysis.db") as connection:
        connection.execute(
            "CREATE TABLE Per_Image (ImageNumber INTEGER PRIMARY KEY, Image_Value REAL)"
        )
        connection.execute("INSERT INTO Per_Image VALUES (1, 1.5)")
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(
            measurement_dialect=CELLPROFILER_MEASUREMENT_DIALECT,
        ),
    )

    assert not report.is_equivalent
    assert any("schema differs" in message for message in report.failure_messages())


def test_database_export_equivalence_applies_numeric_tolerance(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_database(reference / "analysis.db", value=1.5)
    _write_database(candidate / "analysis.db", value=1.5005)
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(numeric_abs_tolerance=0.001),
    )

    assert report.is_equivalent


def test_database_export_equivalence_uses_per_object_table_owner_for_aggregate_fields(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_per_object_aggregate_database(reference / "analysis.db")
    _write_per_object_aggregate_database(candidate / "analysis.db")
    _write_properties(
        reference / "analysis.properties",
        reference / "analysis.db",
        object_table="Per_Cells",
        object_id="Cells_Number_Object_Number",
    )
    _write_properties(
        candidate / "analysis.properties",
        candidate / "analysis.db",
        object_table="Per_Cells",
        object_id="Cells_Number_Object_Number",
    )

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert report.is_equivalent


def test_database_export_equivalence_compares_object_columns_strictly(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_tie_sensitive_object_database(
        reference / "analysis.db",
        location=10.0,
        intensity=0.5,
    )
    _write_tie_sensitive_object_database(
        candidate / "analysis.db",
        location=12.0,
        intensity=0.5,
    )
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert not report.is_equivalent


def test_database_export_equivalence_uses_declared_tie_sensitive_feature_relation(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_tie_sensitive_object_database(
        reference / "analysis.db",
        location=10.0,
        intensity=0.5,
    )
    _write_tie_sensitive_object_database(
        candidate / "analysis.db",
        location=12.0,
        intensity=0.5,
    )
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=cellprofiler_runtime_equivalence_policy(
            allow_tie_sensitive_location_mismatches=True,
            allow_unstable_shape_descriptors=False,
            allow_sparse_object_boundary_jitter=False,
        ),
    )

    assert report.is_equivalent


def test_database_export_equivalence_rejects_object_location_drift_when_value_drifts(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_tie_sensitive_object_database(
        reference / "analysis.db",
        location=10.0,
        intensity=0.5,
    )
    _write_tie_sensitive_object_database(
        candidate / "analysis.db",
        location=12.0,
        intensity=0.7,
    )
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert not report.is_equivalent
    assert any("values differ" in message for message in report.failure_messages())


def test_database_export_equivalence_compares_aggregate_columns_strictly(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_tie_sensitive_image_database(
        reference / "analysis.db",
        location=10.0,
        intensity=0.5,
    )
    _write_tie_sensitive_image_database(
        candidate / "analysis.db",
        location=12.0,
        intensity=0.5,
    )
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert not report.is_equivalent


def test_database_export_equivalence_compares_relationship_rows_exactly(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_relationship_database(reference / "analysis.db", target_object=2)
    _write_relationship_database(candidate / "analysis.db", target_object=3)
    _write_properties(reference / "analysis.properties", reference / "analysis.db")
    _write_properties(candidate / "analysis.properties", candidate / "analysis.db")

    report = cellprofiler_database_export_equivalence(
        reference,
        RuntimeExportObservation.from_output_root(candidate),
        policy=RuntimeEquivalencePolicy(),
    )

    assert not report.is_equivalent
    assert any(
        "relationship_type_id" in message and "values differ" in message
        for message in report.failure_messages()
    )


def test_database_column_dialect_constructs_prefixed_field_once() -> None:
    dialect = CellProfilerDatabaseColumnDialect()
    projected = dialect.measurement_field(
        MeasurementSubject(MeasurementScope.OBJECT, "Cytoplasm"),
        FieldSpec("Cytoplasm_CustomFeature", float),
    )

    assert projected == FieldSpec("Cytoplasm_Cytoplasm_CustomFeature", float)


def test_database_column_dialect_inverts_declared_table_and_field_projection() -> None:
    dialect = CellProfilerDatabaseColumnDialect()
    subject = dialect.object_subject(
        "Per_Cytoplasm",
        "Cytoplasm_Number_Object_Number",
    )

    assert subject == MeasurementSubject(
        MeasurementScope.OBJECT,
        "Cytoplasm",
        id_field="Number_Object_Number",
    )
    assert dialect.source_measurement_field(
        subject,
        FieldSpec("Cytoplasm_Location_MaxIntensity_X_rawGFP", float),
    ) == FieldSpec("Location_MaxIntensity_X_rawGFP", float)


def test_database_column_dialect_inverts_prefixed_image_and_combined_object() -> None:
    dialect = CellProfilerDatabaseColumnDialect.from_image_table(
        "BBBC022QC_Per_Image"
    )

    assert dialect.table_prefix == "BBBC022QC_"
    assert dialect.combined_object_table_name() == "BBBC022QC_Per_Object"
    assert dialect.image_subject(
        "BBBC022QC_Per_Image",
        "ImageNumber",
    ) == MeasurementSubject(MeasurementScope.IMAGE, "Image")
    assert dialect.object_subject(
        "BBBC022QC_Per_Object",
        "ObjectNumber",
    ) is None


def test_database_column_dialect_inverts_declared_image_object_aggregate() -> None:
    dialect = CellProfilerDatabaseColumnDialect()
    external_field = dialect.image_aggregate_field(
        CellProfilerImageAggregateStatistic.MEAN,
        FieldSpec("Cells_Children_Cytoplasm_Count", float),
    )

    assert external_field.name == "Mean_Cells_Children_Cytoplasm_Count"
    assert dialect.source_measurement_field(
        MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        external_field,
    ) == external_field


def test_database_column_declarations_own_export_and_equivalence_names() -> None:
    group_family = CellProfilerImageStructuralFieldFamily.GROUP

    assert CellProfilerDatabaseColumnDialect.group_field("Plate").name == (
        f"Image_{group_family.qualified_name('Plate')}"
    )
    assert CELLPROFILER_MEASUREMENT_DIALECT.non_measurement_field_prefixes == tuple(
        normalize_runtime_identifier(family.field_prefix).rstrip("_") + "_"
        for family in CellProfilerImageStructuralFieldFamily
    )
