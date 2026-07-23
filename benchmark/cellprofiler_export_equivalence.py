"""Semantic parity checks for CellProfiler SQLite and CPA exports."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from openhcs.core.equivalence.comparison import runtime_table_differences
from openhcs.core.equivalence.outputs import RuntimeOutputSnapshot
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.equivalence.report import (
    RuntimeEquivalenceDifference,
    RuntimeEquivalenceDifferenceKind,
    RuntimeEquivalenceReport,
)
from openhcs.core.equivalence.tables import RuntimeTableSnapshot
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.runtime_equivalence import (
    RuntimeMeasurementSnapshot,
    runtime_measurement_equivalence,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    ); from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.interop.cellprofiler.analyst_export import (
    CPAExperimentPropertyColumn,
    CPAPropertyName,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerDatabaseColumnDialect,
    CellProfilerExperimentProjectionField,
)


def cellprofiler_database_export_equivalence(
    reference_output_root: Path,
    candidate_exports: RuntimeExportObservation,
    *,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeEquivalenceReport:
    """Compare SQLite databases and CPA properties emitted by CellProfiler."""

    reference_root = Path(reference_output_root)
    reference_properties = tuple(sorted(reference_root.rglob("*.properties")))
    candidate_properties = _outputs_with_suffix(candidate_exports, ".properties")
    differences = [
        *_sqlite_export_differences(
            _declared_sqlite_paths(
                tuple(sorted(reference_root.rglob("*.db"))),
                reference_properties,
            ),
            _declared_sqlite_paths(
                _outputs_with_suffix(candidate_exports, ".db"),
                candidate_properties,
            ),
            _declared_sqlite_table_subjects(reference_properties),
            _declared_sqlite_table_subjects(candidate_properties),
            policy,
        ),
        *_properties_export_differences(
            reference_properties,
            candidate_properties,
        ),
    ]
    return RuntimeEquivalenceReport(tuple(differences))


def _outputs_with_suffix(
    exports: RuntimeExportObservation,
    suffix: str,
) -> tuple[Path, ...]:
    return tuple(path for path in exports.output_files if path.suffix.lower() == suffix)


def _declared_sqlite_paths(
    database_paths: Sequence[Path],
    properties_paths: Sequence[Path],
) -> tuple[Path, ...]:
    if not properties_paths:
        return tuple(database_paths)
    declared_names = frozenset(
        Path(properties[CPAPropertyName.SQLITE_FILE.value]).name
        for path in properties_paths
        for properties in (_read_cpa_properties(path),)
    )
    return tuple(path for path in database_paths if path.name in declared_names)


def _sqlite_export_differences(
    reference_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
    reference_subjects: Mapping[str, Mapping[str, MeasurementSubject]],
    candidate_subjects: Mapping[str, Mapping[str, MeasurementSubject]],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    differences, reference_by_name, candidate_by_name = _named_output_differences(
        reference_paths,
        candidate_paths,
        output_label="SQLite database",
    )
    for name in sorted(reference_by_name.keys() & candidate_by_name.keys()):
        differences.extend(
            _sqlite_database_differences(
                reference_by_name[name],
                candidate_by_name[name],
                reference_subjects.get(name, {}),
                candidate_subjects.get(name, {}),
                policy,
            )
        )
    return tuple(differences)


def _sqlite_database_differences(
    reference_path: Path,
    candidate_path: Path,
    reference_subjects: Mapping[str, MeasurementSubject],
    candidate_subjects: Mapping[str, MeasurementSubject],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    reference_tables = _sqlite_tables(reference_path, reference_subjects, policy)
    candidate_tables = _sqlite_tables(candidate_path, candidate_subjects, policy)
    differences: list[RuntimeEquivalenceDifference] = []
    reference_names = set(reference_tables)
    candidate_names = set(candidate_tables)
    if reference_names != candidate_names:
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_SCHEMA,
                f"SQLite database {reference_path.name!r} table names differ: "
                f"reference={tuple(sorted(reference_names))!r}, "
                f"candidate={tuple(sorted(candidate_names))!r}",
            )
        )
    for table_name in sorted(reference_names & candidate_names):
        reference_schema, reference_table = reference_tables[table_name]
        candidate_schema, candidate_table = candidate_tables[table_name]
        if reference_schema != candidate_schema:
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.TABLE_SCHEMA,
                    f"SQLite database {reference_path.name!r} table "
                    f"{table_name!r} schema differs: "
                    f"reference={reference_schema!r}, candidate={candidate_schema!r}",
                )
            )
            continue
        if table_name in reference_subjects and table_name in candidate_subjects:
            table_report = runtime_measurement_equivalence(
                RuntimeMeasurementSnapshot.from_output_snapshot(
                    RuntimeOutputSnapshot(tables=(reference_table,)),
                    policy=policy,
                ),
                RuntimeMeasurementSnapshot.from_output_snapshot(
                    RuntimeOutputSnapshot(tables=(candidate_table,)),
                    policy=policy,
                ),
                policy=policy,
            ).differences
        else:
            table_report = runtime_table_differences(
                (reference_table,),
                (candidate_table,),
                policy,
            )
        differences.extend(
            RuntimeEquivalenceDifference(
                difference.kind,
                f"SQLite database {reference_path.name!r} table "
                f"{table_name!r}: {difference.message}",
            )
            for difference in table_report
        )
    return tuple(differences)


def _sqlite_tables(
    path: Path,
    subjects: Mapping[str, MeasurementSubject],
    policy: RuntimeEquivalencePolicy,
) -> dict[
    str,
    tuple[
        tuple[str, tuple[tuple[int, str, str, int, str | None, int], ...]],
        RuntimeTableSnapshot,
    ],
]:
    tables = {}
    database_dialect = CellProfilerDatabaseColumnDialect()
    image_subject = MeasurementSubject(
        MeasurementScope.IMAGE,
        MeasurementScope.IMAGE.value,
    )
    structural_field_prefixes = tuple(
        normalize_runtime_identifier(
            database_dialect.measurement_field(
                image_subject,
                FieldSpec(prefix),
            ).name
        )
        for prefix in policy.measurement_dialect.non_measurement_field_prefixes
    )
    with sqlite3.connect(path) as connection:
        objects = connection.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for table_name, object_type in objects:
            quoted_name = _quote_sqlite_identifier(table_name)
            raw_schema_rows = tuple(
                tuple(row)
                for row in connection.execute(f"PRAGMA table_info({quoted_name})")
            )
            retained_indices = tuple(
                index
                for index, row in enumerate(raw_schema_rows)
                if not normalize_runtime_identifier(str(row[1])).startswith(
                    structural_field_prefixes
                )
            )
            retained_schema_rows = tuple(
                raw_schema_rows[index] for index in retained_indices
            )
            schema_rows = tuple(
                (retained_index, *row[1:])
                for retained_index, row in enumerate(retained_schema_rows)
            )
            external_header = tuple(str(row[1]) for row in schema_rows)
            subject = subjects.get(str(table_name))
            semantic_header = (
                external_header
                if subject is None
                else tuple(
                    database_dialect.source_measurement_field(
                        subject,
                        FieldSpec(field_name),
                    ).name
                    for field_name in external_header
                )
            )
            rows = tuple(
                _normalized_sqlite_row(
                    external_header,
                    tuple(row[index] for index in retained_indices),
                )
                for row in connection.execute(f"SELECT * FROM {quoted_name}")
            )
            tables[str(table_name)] = (
                (str(object_type), schema_rows),
                RuntimeTableSnapshot(
                    path=Path(f"{table_name}.csv"),
                    header=semantic_header,
                    rows=rows,
                    column_context=(
                        ()
                        if subject is None
                        else (subject.name,) * len(semantic_header)
                    ),
                ),
            )
    return tables


def _declared_sqlite_table_subjects(
    properties_paths: Sequence[Path],
) -> Mapping[str, Mapping[str, MeasurementSubject]]:
    """Return exact CPA-declared measurement subjects by database and table."""

    subjects: dict[str, dict[str, MeasurementSubject]] = {}
    for properties_path in properties_paths:
        properties = _read_cpa_properties(properties_path)
        database_name = Path(properties[CPAPropertyName.SQLITE_FILE.value]).name
        image_table = properties[CPAPropertyName.IMAGE_TABLE.value]
        database_dialect = CellProfilerDatabaseColumnDialect.from_image_table(
            image_table
        )
        image_subject = database_dialect.image_subject(
            image_table,
            properties[CPAPropertyName.IMAGE_ID.value],
        )
        object_table = properties[CPAPropertyName.OBJECT_TABLE.value]
        object_subject = database_dialect.object_subject(
            object_table,
            properties[CPAPropertyName.OBJECT_ID.value],
        )
        database_subjects = subjects.setdefault(database_name, {})
        for table_name, subject in (
            (image_table, image_subject),
            (object_table, object_subject),
        ):
            if subject is None:
                continue
            existing = database_subjects.get(table_name)
            if existing is not None and existing != subject:
                raise ValueError(
                    f"CPA database {database_name!r} table {table_name!r} has "
                    f"conflicting declared subjects {existing!r} and {subject!r}."
                )
            database_subjects[table_name] = subject
    return subjects


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sqlite_cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _normalized_sqlite_row(
    header: tuple[str, ...],
    row: tuple[object, ...],
) -> tuple[str, ...]:
    values = [_sqlite_cell_text(value) for value in row]
    volatile_columns = frozenset(
        field.field_name
        for field in CellProfilerExperimentProjectionField
        if field.volatile_value
    )
    for index, column_name in enumerate(header):
        if column_name in volatile_columns:
            values[index] = ""

    property_columns = frozenset(column.value for column in CPAExperimentPropertyColumn)
    if property_columns.issubset(header):
        field_index = header.index(CPAExperimentPropertyColumn.FIELD.value)
        value_index = header.index(CPAExperimentPropertyColumn.VALUE.value)
        try:
            property_name = CPAPropertyName(values[field_index])
        except ValueError:
            pass
        else:
            values[value_index] = property_name.normalized_value(values[value_index])
    return tuple(values)


def _properties_export_differences(
    reference_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
) -> tuple[RuntimeEquivalenceDifference, ...]:
    differences, reference_by_name, candidate_by_name = _named_output_differences(
        reference_paths,
        candidate_paths,
        output_label="CPA properties file",
    )
    for name in sorted(reference_by_name.keys() & candidate_by_name.keys()):
        reference_properties = _read_cpa_properties(reference_by_name[name])
        candidate_properties = _read_cpa_properties(candidate_by_name[name])
        reference_keys = set(reference_properties)
        candidate_keys = set(candidate_properties)
        if reference_keys != candidate_keys:
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.TABLE_SCHEMA,
                    f"CPA properties file {name!r} keys differ: "
                    f"reference={tuple(sorted(reference_keys))!r}, "
                    f"candidate={tuple(sorted(candidate_keys))!r}",
                )
            )
        mismatched_values = tuple(
            key
            for key in sorted(reference_keys & candidate_keys)
            if _normalized_cpa_property_value(key, reference_properties[key])
            != _normalized_cpa_property_value(key, candidate_properties[key])
        )
        if mismatched_values:
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.TABLE_CONTENT,
                    f"CPA properties file {name!r} values differ for keys "
                    f"{mismatched_values!r}",
                )
            )
    return tuple(differences)


def _read_cpa_properties(path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        properties[key.strip()] = value.strip()
    return properties


def _normalized_cpa_property_value(key: str, value: str) -> str:
    try:
        property_name = CPAPropertyName(key)
    except ValueError:
        return value.strip()
    return property_name.normalized_value(value)


def _named_output_differences(
    reference_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
    *,
    output_label: str,
) -> tuple[
    list[RuntimeEquivalenceDifference],
    Mapping[str, Path],
    Mapping[str, Path],
]:
    reference_by_name = _unique_paths_by_name(reference_paths, output_label)
    candidate_by_name = _unique_paths_by_name(candidate_paths, output_label)
    reference_names = set(reference_by_name)
    candidate_names = set(candidate_by_name)
    differences: list[RuntimeEquivalenceDifference] = []
    if reference_names != candidate_names:
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_COUNT,
                f"{output_label} names differ: "
                f"reference={tuple(sorted(reference_names))!r}, "
                f"candidate={tuple(sorted(candidate_names))!r}",
            )
        )
    return differences, reference_by_name, candidate_by_name


def _unique_paths_by_name(
    paths: Sequence[Path],
    output_label: str,
) -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    for path in paths:
        if path.name in by_name:
            raise ValueError(
                f"{output_label} output name {path.name!r} is ambiguous between "
                f"{by_name[path.name]} and {path}."
            )
        by_name[path.name] = path
    return by_name
