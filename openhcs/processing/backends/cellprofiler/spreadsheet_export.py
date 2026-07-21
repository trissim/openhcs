"""CellProfiler-compatible spreadsheet projection over runtime artifacts."""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import PurePosixPath
from typing import cast, TYPE_CHECKING, ClassVar, TypeVar

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.equivalence import (
    measurement_qualifier_field_names,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
    MeasurementRowsAxisProjection,
    WideMeasurementRowAccumulator,
)
from openhcs.core.pipeline.function_contracts import (
    execution_scope,
    runtime_bound_parameters,
)
from openhcs.core.runtime_artifact_queries import MeasurementTableUnion
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    measurement_axis_integer_value,
)
from openhcs.core.runtime_identifier import (
    normalize_runtime_identifier,
)
from openhcs.core.runtime_stores import RuntimeArtifactBatch, StoredRuntimeValue
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.source_image_provenance import (
    source_component_metadata_consensus,
)
from openhcs.core.source_metadata import (
    SourceMetadataRoleView,
)
from openhcs.interop.cellprofiler.image_set_numbering import (
    CellProfilerImageSetNumbering,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ArtifactExportModule,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
    cellprofiler_projected_measurement_feature_name,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    is_blank_symbol_name,
    repeating_setting_blocks,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
)
from openhcs.processing.materialization import (
    FileBundleOptions,
    MaterializationSpec,
    WriteMode,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext


_METADATA_TEMPLATE = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
_CELLPROFILER_METADATA_TEMPLATE = re.compile(r"\\g<(?P<name>[A-Za-z_][A-Za-z0-9_]*)>")
_EnumT = TypeVar("_EnumT", bound=Enum)


class SpreadsheetDelimiter(str, Enum):
    """Supported CellProfiler spreadsheet delimiters."""

    COMMA = ","
    TAB = "\t"

    @classmethod
    def from_cellprofiler(cls, value: str) -> "SpreadsheetDelimiter":
        """Parse one CellProfiler delimiter setting."""

        normalized = value.strip().casefold()
        if normalized in {'comma (",")', "comma", ","}:
            return cls.COMMA
        if normalized in {"tab", "\\t"}:
            return cls.TAB
        raise ValueError(f"Unsupported spreadsheet delimiter {value!r}.")

    @property
    def default_suffix(self) -> str:
        """Return the conventional suffix for this delimiter."""

        return ".csv" if self is SpreadsheetDelimiter.COMMA else ".txt"


class SpreadsheetNanRepresentation(str, Enum):
    """How non-finite numeric values appear in spreadsheet cells."""

    NAN = "nan"
    NULL = "null"

    @classmethod
    def from_cellprofiler(cls, value: str) -> "SpreadsheetNanRepresentation":
        """Parse one CellProfiler NaN/Inf representation setting."""

        normalized = value.strip().casefold()
        if normalized == "nan":
            return cls.NAN
        if normalized in {"null", "nulls"}:
            return cls.NULL
        raise ValueError(f"Unsupported spreadsheet NaN representation {value!r}.")


class CellProfilerSpreadsheetRowField(str, Enum):
    """CellProfiler-owned spreadsheet row fields."""

    IMAGE_NUMBER = "image_number"


@dataclass(frozen=True, slots=True)
class SpreadsheetColumnSelection:
    """One CellProfiler subject/measurement column selection."""

    subject: str
    feature: str

    def __post_init__(self) -> None:
        if not self.subject.strip() or not self.feature.strip():
            raise ValueError(
                "SpreadsheetColumnSelection subject and feature must be non-empty."
            )

    @classmethod
    def from_cellprofiler(
        cls,
        value: str,
    ) -> tuple["SpreadsheetColumnSelection", ...]:
        """Parse CellProfiler's comma-separated ``subject|feature`` value."""

        selections: list[SpreadsheetColumnSelection] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            parts = tuple(part.strip() for part in token.split("|", 1))
            if len(parts) != 2:
                raise ValueError(
                    "Spreadsheet measurement selections must use "
                    f"'subject|feature', got {token!r}."
                )
            subject, feature = parts
            if subject.casefold() == "none" and feature.casefold() == "none":
                continue
            selections.append(cls(subject, feature))
        return tuple(selections)

    def matches(self, subject: str, feature: str) -> bool:
        """Return whether this selection addresses one projected column."""

        return normalize_runtime_identifier(
            self.subject
        ) == normalize_runtime_identifier(subject) and normalize_runtime_identifier(
            self.feature
        ) == normalize_runtime_identifier(feature)


@dataclass(frozen=True, slots=True)
class SpreadsheetFileSelection:
    """One output file and the measurement subjects rendered into it."""

    subjects: tuple[str, ...]
    file_name: str

    def __post_init__(self) -> None:
        subjects = tuple(dict.fromkeys(subject.strip() for subject in self.subjects))
        if not subjects or any(not subject for subject in subjects):
            raise ValueError(
                "SpreadsheetFileSelection.subjects must contain non-empty names."
            )
        file_name = self.file_name.strip()
        if not file_name:
            raise ValueError("SpreadsheetFileSelection.file_name cannot be empty.")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "file_name", file_name)


def cellprofiler_metadata_template(value: str) -> str:
    """Translate CellProfiler metadata references into a public string template."""

    return _CELLPROFILER_METADATA_TEMPLATE.sub(
        lambda match: "{" + match.group("name") + "}",
        value.strip(),
    )


def cellprofiler_output_directory(value: str) -> str:
    """Parse a CellProfiler output-folder setting as a relative bundle directory."""

    location, separator, relative = value.partition("|")
    if not separator:
        raise ValueError(
            "Spreadsheet output location must contain a CellProfiler folder choice."
        )
    if location.strip().casefold() not in {
        "default output folder",
        "default output folder sub-folder",
    }:
        raise ValueError(
            "Spreadsheet export supports only Default Output Folder locations, got "
            f"{location!r}."
        )
    normalized = cellprofiler_metadata_template(relative).replace("\\", "/")
    if normalized in {"", "."}:
        return ""
    return normalized.strip("/")


def render_spreadsheet_bundle(
    artifact_batch: RuntimeArtifactBatch,
    *,
    delimiter: SpreadsheetDelimiter = SpreadsheetDelimiter.COMMA,
    add_image_metadata: bool = False,
    add_image_file_names: bool = False,
    select_measurements: bool = False,
    selected_columns: tuple[SpreadsheetColumnSelection, ...] = (),
    calculate_aggregate_means: bool = False,
    calculate_aggregate_medians: bool = False,
    calculate_aggregate_standard_deviations: bool = False,
    output_directory: str = "",
    export_all_measurement_types: bool = True,
    file_selections: tuple[SpreadsheetFileSelection, ...] = (),
    nan_representation: SpreadsheetNanRepresentation = (
        SpreadsheetNanRepresentation.NAN
    ),
    add_filename_prefix: bool = True,
    filename_prefix: str = "MyExpt_",
) -> dict[str, str | bytes]:
    """Render exactly the measurement records selected by ``artifact_batch``."""

    if not isinstance(artifact_batch, RuntimeArtifactBatch):
        raise TypeError("artifact_batch must be RuntimeArtifactBatch.")
    delimiter = _coerce_enum(SpreadsheetDelimiter, delimiter)
    nan_representation = _coerce_enum(
        SpreadsheetNanRepresentation,
        nan_representation,
    )
    selected_columns = tuple(selected_columns)
    if any(
        not isinstance(selection, SpreadsheetColumnSelection)
        for selection in selected_columns
    ):
        raise TypeError(
            "selected_columns must contain SpreadsheetColumnSelection values."
        )
    file_selections = tuple(file_selections)
    if any(
        not isinstance(selection, SpreadsheetFileSelection)
        for selection in file_selections
    ):
        raise TypeError("file_selections must contain SpreadsheetFileSelection values.")

    image_numbers = CellProfilerImageSetNumbering(
        artifact_batch.source_image_set_identity_policy
    )
    tables, object_subjects = _measurement_tables(artifact_batch, image_numbers)
    relationship_rows = _relationship_rows(artifact_batch, image_numbers)
    if relationship_rows:
        tables["Object relationships"] = relationship_rows
    source_image_rows = tables.get("Image", ())

    tables = _selected_table_columns(
        tables,
        selected_columns=selected_columns,
        enabled=bool(select_measurements),
    )
    tables = _with_requested_aggregates(
        tables,
        object_subjects=object_subjects,
        mean=bool(calculate_aggregate_means),
        median=bool(calculate_aggregate_medians),
        standard_deviation=bool(calculate_aggregate_standard_deviations),
    )
    tables = _with_image_columns_on_objects(
        tables,
        object_subjects=object_subjects,
        image_rows=source_image_rows,
        add_metadata=bool(add_image_metadata),
        add_file_names=bool(add_image_file_names),
    )

    selections = (
        _automatic_file_selections(tables, delimiter)
        if export_all_measurement_types
        else file_selections
    )
    prefix = filename_prefix if add_filename_prefix else ""
    bundle: dict[str, str | bytes] = {}
    for selection in selections:
        selected_tables = tuple(
            (subject, tables[subject])
            for subject in selection.subjects
            if subject in tables
        )
        if not selected_tables:
            continue
        rows = _combined_rows(selected_tables)
        path_template = _bundle_path_template(
            output_directory=output_directory,
            prefix=prefix,
            file_name=selection.file_name,
        )
        for relative_path, selected_rows in _rows_by_resolved_path(
            path_template,
            rows,
            image_rows=source_image_rows,
        ):
            if relative_path in bundle:
                raise ValueError(
                    f"Spreadsheet export produced duplicate path {relative_path!r}."
                )
            bundle[relative_path] = _render_csv(
                selected_rows,
                delimiter=delimiter,
                nan_representation=nan_representation,
            )
    return bundle


def _measurement_tables(
    artifact_batch: RuntimeArtifactBatch,
    image_numbers: CellProfilerImageSetNumbering,
) -> tuple[
    OrderedDict[str, tuple[Mapping[str, object], ...]],
    tuple[str, ...],
]:
    accumulator = WideMeasurementRowAccumulator(
        CELLPROFILER_MEASUREMENT_DIALECT.row_identity_contract
    )
    source_metadata_by_image_number: OrderedDict[
        int,
        list[Mapping[str, object]],
    ] = OrderedDict()
    all_tables: list[MeasurementTable] = []
    for spec in artifact_batch.specs_of_type(MeasurementsArtifactType):
        records_by_axis = artifact_batch.records(spec.ref())
        records = tuple(
            record
            for axis_records in records_by_axis.values()
            for record in axis_records
        )
        tables = tuple(cast(MeasurementTable, record.value.data) for record in records)
        all_tables.extend(tables)
        slice_axis = MeasurementRowAxisField.SLICE_INDEX
        row_domains = tuple(
            MeasurementRowsAxisProjection.from_rows(table.rows) for table in tables
        )
        MeasurementTableUnion(spec.name, tables).row_axis_domain(slice_axis)
        for record, table, row_domain in zip(
            records,
            tables,
            row_domains,
            strict=True,
        ):
            image_numbers_by_slice = image_numbers.for_source_slices(
                scope=record.key.scope,
                provenance=table.source_provenance,
                slice_indices=row_domain.present_axis_values(slice_axis.value),
                owner=table.name,
            )
            accumulator.add(
                image_numbers.project_measurement_rows(
                    scope=record.key.scope,
                    table=table,
                ),
                cellprofiler_projected_measurement_feature_name,
                default_subject=_measurement_subject_name(table),
                default_scope=table.subject.scope,
                source_image_name=table.source_image_name,
                object_id_field=table.subject.object_id_field,
                qualifier_field_names=measurement_qualifier_field_names(
                    CELLPROFILER_MEASUREMENT_DIALECT
                ),
            )
            for image_number, metadata in _source_metadata_measurement_rows(
                table,
                row_domain,
                image_numbers_by_slice,
            ):
                source_metadata_by_image_number.setdefault(
                    image_number,
                    [],
                ).append(metadata)
    for table in CellProfilerModule.derive_experiment_measurement_tables(all_tables):
        accumulator.add(
            table.rows,
            cellprofiler_projected_measurement_feature_name,
            default_subject=_measurement_subject_name(table),
            default_scope=table.subject.scope,
            source_image_name=table.source_image_name,
            object_id_field=table.subject.object_id_field,
            qualifier_field_names=measurement_qualifier_field_names(
                CELLPROFILER_MEASUREMENT_DIALECT
            ),
        )
    source_metadata_rows = tuple(
        {
            MeasurementRowAxisField.SLICE_INDEX.value: image_number,
            **{
                f"Metadata_{field_name}": value
                for field_name, value in consensus.items()
            },
        }
        for image_number, metadata_rows in source_metadata_by_image_number.items()
        for consensus in (source_component_metadata_consensus(metadata_rows),)
        if consensus is not None
    )
    if source_metadata_rows:
        metadata_field_names = tuple(
            dict.fromkeys(
                field_name
                for row in source_metadata_rows
                for field_name in row
                if field_name != MeasurementRowAxisField.SLICE_INDEX.value
            )
        )
        accumulator.add(
            MeasurementSparseColumnarRows.from_rows(
                source_metadata_rows,
                fields=(
                    FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
                    *(
                        FieldSpec(field_name, str, required=False)
                        for field_name in metadata_field_names
                    ),
                ),
            ),
            cellprofiler_projected_measurement_feature_name,
            default_subject="Image",
            default_scope=MeasurementScope.IMAGE,
        )
    return OrderedDict(
        (subject, _cellprofiler_rows(rows))
        for subject, rows in accumulator.row_mappings_by_subject().items()
    ), accumulator.object_subjects()


def _source_metadata_measurement_rows(
    table: MeasurementTable,
    row_domain: MeasurementRowsAxisProjection,
    image_numbers_by_slice: Mapping[int, int],
) -> tuple[tuple[int, Mapping[str, object]], ...]:
    """Project producer-owned source metadata into CellProfiler Image rows."""

    rows: list[tuple[int, Mapping[str, object]]] = []
    for slice_index in row_domain.present_axis_values(
        MeasurementRowAxisField.SLICE_INDEX.value
    ):
        metadata = table.source_provenance.for_source_plane(
            slice_index
        ).source_component_metadata
        if metadata is None:
            continue
        role_view = SourceMetadataRoleView(metadata)
        original_metadata = dict(role_view.original_items())
        if original_metadata:
            rows.append((image_numbers_by_slice[slice_index], original_metadata))
    return tuple(rows)


def _relationship_rows(
    artifact_batch: RuntimeArtifactBatch,
    image_numbers: CellProfilerImageSetNumbering,
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for record in _records_in_contract_order(
        artifact_batch,
        RelationshipsArtifactType,
    ):
        relationship = cast(ObjectRelationship, record.value.data)
        image_numbers_by_slice = image_numbers.for_source_slices(
            scope=record.key.scope,
            provenance=relationship.source_provenance,
            slice_indices=relationship.payload.slice_indices,
            owner=relationship.name,
        )
        rows.extend(
            _cellprofiler_rows(
                MeasurementRowsAxisProjection.from_rows(
                    relationship.row_mappings()
                ).remap_runtime_slice_indices(image_numbers_by_slice)
            )
        )
    return tuple(rows)


def _cellprofiler_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Expose canonical image-number coordinates in CellProfiler rows."""

    slice_field = MeasurementRowAxisField.SLICE_INDEX.value
    image_number_field = CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value
    projected_rows: list[Mapping[str, object]] = []
    for row in rows:
        if slice_field not in row:
            projected_rows.append(row)
            continue
        slice_index = measurement_axis_integer_value(
            row[slice_field],
            MeasurementRowAxisField.SLICE_INDEX,
        )
        if slice_index is None:
            raise ValueError(
                "CellProfiler spreadsheet export requires an integer "
                f"{slice_field!r}, got {row[slice_field]!r}."
            )
        projected_rows.append(
            {
                (image_number_field if field_name == slice_field else field_name): (
                    slice_index if field_name == slice_field else value
                )
                for field_name, value in row.items()
            }
        )
    return tuple(projected_rows)


def _records_in_contract_order(
    artifact_batch: RuntimeArtifactBatch,
    artifact_type: type[ArtifactType],
) -> tuple[StoredRuntimeValue, ...]:
    return tuple(
        record
        for spec in artifact_batch.specs_of_type(artifact_type)
        for records in artifact_batch.records(spec.ref()).values()
        for record in records
    )


def _measurement_subject_name(table: MeasurementTable) -> str:
    subject = table.subject
    if subject is None:
        return table.name
    if subject.scope is MeasurementScope.IMAGE:
        return "Image"
    if subject.scope is MeasurementScope.EXPERIMENT:
        return "Experiment"
    if subject.scope is MeasurementScope.OBJECT:
        if subject.name is None:
            raise ValueError(f"Object measurement table {table.name!r} has no subject.")
        return subject.name
    return subject.name or table.name


def _selected_table_columns(
    tables: OrderedDict[str, tuple[Mapping[str, object], ...]],
    *,
    selected_columns: tuple[SpreadsheetColumnSelection, ...],
    enabled: bool,
) -> OrderedDict[str, tuple[Mapping[str, object], ...]]:
    if not enabled:
        return tables
    axis_fields = (
        *MeasurementRowAxisField.field_names(),
        CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value,
    )
    return OrderedDict(
        (
            subject,
            (
                rows
                if subject == "Object relationships"
                else tuple(
                    {
                        field_name: value
                        for field_name, value in row.items()
                        if field_name in axis_fields
                        or any(
                            selection.matches(subject, field_name)
                            for selection in selected_columns
                        )
                    }
                    for row in rows
                )
            ),
        )
        for subject, rows in tables.items()
    )


def _with_requested_aggregates(
    tables: OrderedDict[str, tuple[Mapping[str, object], ...]],
    *,
    object_subjects: tuple[str, ...],
    mean: bool,
    median: bool,
    standard_deviation: bool,
) -> OrderedDict[str, tuple[Mapping[str, object], ...]]:
    if not (mean or median or standard_deviation):
        return tables
    image_field = CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value
    image_rows = [dict(row) for row in tables.get("Image", ())]
    image_rows_by_number = {
        row[image_field]: row for row in image_rows if image_field in row
    }
    for subject in object_subjects:
        rows = tables.get(subject, ())
        rows_by_image: OrderedDict[object, list[Mapping[str, object]]] = OrderedDict()
        for row in rows:
            if image_field not in row:
                raise ValueError(
                    "Object measurement aggregation requires image_number in every "
                    f"{subject!r} row."
                )
            rows_by_image.setdefault(row[image_field], []).append(row)
        for image_number, subject_rows in rows_by_image.items():
            image_row = image_rows_by_number.get(image_number)
            if image_row is None:
                raise ValueError(
                    "Object measurement aggregation requires a producer-declared "
                    f"Image measurement row for {image_field}={image_number!r}."
                )
            for feature in _numeric_features(subject_rows):
                values = tuple(
                    float(row[feature]) for row in subject_rows if feature in row
                )
                if mean:
                    image_row[f"Mean_{subject}_{feature}"] = statistics.fmean(values)
                if median:
                    image_row[f"Median_{subject}_{feature}"] = statistics.median(values)
                if standard_deviation:
                    image_row[f"StDev_{subject}_{feature}"] = statistics.pstdev(values)
    updated = OrderedDict(tables)
    updated["Image"] = tuple(image_rows)
    return updated


def _numeric_features(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    axis_fields = frozenset(
        (
            *MeasurementRowAxisField.field_names(),
            CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value,
        )
    )
    candidates = tuple(
        dict.fromkeys(
            field_name
            for row in rows
            for field_name, value in row.items()
            if field_name not in axis_fields
            and isinstance(value, Real)
            and not isinstance(value, bool)
        )
    )
    return tuple(
        field_name
        for field_name in candidates
        if all(
            field_name not in row
            or (
                isinstance(row[field_name], Real)
                and not isinstance(row[field_name], bool)
            )
            for row in rows
        )
    )


def _with_image_columns_on_objects(
    tables: OrderedDict[str, tuple[Mapping[str, object], ...]],
    *,
    object_subjects: tuple[str, ...],
    image_rows: tuple[Mapping[str, object], ...],
    add_metadata: bool,
    add_file_names: bool,
) -> OrderedDict[str, tuple[Mapping[str, object], ...]]:
    if not (add_metadata or add_file_names):
        return tables
    image_field = CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value
    image_rows_by_number = {
        row[image_field]: row for row in image_rows if image_field in row
    }
    updated = OrderedDict(tables)
    for subject in object_subjects:
        updated[subject] = tuple(
            _row_with_image_columns(
                row,
                image_rows_by_number.get(row[image_field], {}),
                add_metadata=add_metadata,
                add_file_names=add_file_names,
            )
            for row in tables.get(subject, ())
        )
    return updated


def _row_with_image_columns(
    row: Mapping[str, object],
    image_row: Mapping[str, object],
    *,
    add_metadata: bool,
    add_file_names: bool,
) -> Mapping[str, object]:
    result = dict(row)
    for field_name, value in image_row.items():
        normalized = normalize_runtime_identifier(field_name)
        selected = (
            add_metadata and normalized.startswith(("metadata_", "image_metadata_"))
        ) or (
            add_file_names
            and normalized.startswith(
                (
                    "filename_",
                    "pathname_",
                    "url_",
                    "image_filename_",
                    "image_pathname_",
                    "image_url_",
                )
            )
        )
        if not selected:
            continue
        output_name = (
            field_name
            if normalize_runtime_identifier(field_name).startswith("image_")
            else f"Image_{field_name}"
        )
        result.setdefault(output_name, value)
    return result


def _automatic_file_selections(
    tables: Mapping[str, tuple[Mapping[str, object], ...]],
    delimiter: SpreadsheetDelimiter,
) -> tuple[SpreadsheetFileSelection, ...]:
    return tuple(
        SpreadsheetFileSelection(
            subjects=(subject,),
            file_name=f"{subject}{delimiter.default_suffix}",
        )
        for subject in tables
    )


def _combined_rows(
    selected_tables: tuple[tuple[str, tuple[Mapping[str, object], ...]], ...],
) -> tuple[Mapping[str, object], ...]:
    if len(selected_tables) == 1:
        return selected_tables[0][1]
    image_field = CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value
    grouped: list[tuple[str, OrderedDict[object, list[Mapping[str, object]]]]] = []
    image_order: list[object] = []
    for subject, rows in selected_tables:
        rows_by_image: OrderedDict[object, list[Mapping[str, object]]] = OrderedDict()
        for row in rows:
            if image_field not in row:
                raise ValueError(
                    "Combined spreadsheet subjects require a producer-declared "
                    f"{image_field!r} on every {subject!r} row."
                )
            image_number = row[image_field]
            rows_by_image.setdefault(image_number, []).append(row)
            if image_number not in image_order:
                image_order.append(image_number)
        grouped.append((subject, rows_by_image))

    combined: list[Mapping[str, object]] = []
    for image_number in image_order:
        row_count = max(
            (len(rows_by_image.get(image_number, ())) for _, rows_by_image in grouped),
            default=0,
        )
        for row_index in range(row_count):
            row: dict[str, object] = {image_field: image_number}
            for subject, rows_by_image in grouped:
                subject_rows = rows_by_image.get(image_number, ())
                if row_index >= len(subject_rows):
                    continue
                for field_name, value in subject_rows[row_index].items():
                    if field_name == image_field:
                        continue
                    row[f"{subject}_{field_name}"] = value
            combined.append(row)
    return tuple(combined)


def _bundle_path_template(
    *,
    output_directory: str,
    prefix: str,
    file_name: str,
) -> str:
    relative_directory = output_directory.strip().replace("\\", "/").strip("/")
    path = PurePosixPath(relative_directory) / f"{prefix}{file_name}"
    return str(path)


def _rows_by_resolved_path(
    path_template: str,
    rows: tuple[Mapping[str, object], ...],
    *,
    image_rows: tuple[Mapping[str, object], ...],
) -> tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]:
    tokens = tuple(
        match.group("name") for match in _METADATA_TEMPLATE.finditer(path_template)
    )
    if not tokens:
        return ((path_template, rows),)
    image_field = CellProfilerSpreadsheetRowField.IMAGE_NUMBER.value
    image_rows_by_number = {
        row[image_field]: row for row in image_rows if image_field in row
    }
    grouped: OrderedDict[str, list[Mapping[str, object]]] = OrderedDict()
    for row in rows:
        metadata_row = dict(image_rows_by_number.get(row.get(image_field), {}))
        metadata_row.update(row)
        replacements = {}
        for token in tokens:
            value = _optional_metadata_value(metadata_row, token)
            if value is None:
                plate_values = tuple(
                    dict.fromkeys(
                        candidate
                        for image_row in image_rows
                        for candidate in (_optional_metadata_value(image_row, token),)
                        if candidate is not None
                    )
                )
                if len(plate_values) != 1:
                    raise ValueError(
                        "Spreadsheet path template cannot resolve metadata field "
                        f"{token!r} for an unscoped row; plate values are "
                        f"{plate_values!r}."
                    )
                value = plate_values[0]
            replacements[token] = value
        relative_path = path_template.format_map(replacements)
        grouped.setdefault(relative_path, []).append(row)
    return tuple((path, tuple(path_rows)) for path, path_rows in grouped.items())


def _optional_metadata_value(
    row: Mapping[str, object],
    token: str,
) -> str | None:
    candidates = {
        normalize_runtime_identifier(token),
        normalize_runtime_identifier(f"Metadata_{token}"),
        normalize_runtime_identifier(f"Image_Metadata_{token}"),
    }
    for field_name, value in row.items():
        if normalize_runtime_identifier(field_name) in candidates:
            return str(value)
    return None


def _render_csv(
    rows: tuple[Mapping[str, object], ...],
    *,
    delimiter: SpreadsheetDelimiter,
    nan_representation: SpreadsheetNanRepresentation,
) -> str:
    columns = tuple(dict.fromkeys(field_name for row in rows for field_name in row))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter.value, lineterminator="\n")
    if columns:
        writer.writerow(columns)
        writer.writerows(
            tuple(
                _csv_cell(row.get(column, ""), nan_representation) for column in columns
            )
            for row in rows
        )
    return stream.getvalue()


def _csv_cell(value: object, mode: SpreadsheetNanRepresentation) -> object:
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            if mode is SpreadsheetNanRepresentation.NULL:
                return ""
            if math.isnan(numeric):
                return "NaN"
            return "Inf" if numeric > 0 else "-Inf"
    return value


def _coerce_enum(enum_type: type[_EnumT], value: object) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


@execution_scope(FunctionStepExecutionScope.PLATE)
@runtime_bound_parameters(RuntimeArtifactBatch)
def export_to_spreadsheet(
    *,
    delimiter: SpreadsheetDelimiter = SpreadsheetDelimiter.COMMA,
    add_image_metadata: bool = False,
    add_image_file_names: bool = False,
    select_measurements: bool = False,
    selected_columns: tuple[SpreadsheetColumnSelection, ...] = (),
    calculate_aggregate_means: bool = False,
    calculate_aggregate_medians: bool = False,
    calculate_aggregate_standard_deviations: bool = False,
    output_directory: str = "",
    export_all_measurement_types: bool = True,
    file_selections: tuple[SpreadsheetFileSelection, ...] = (),
    nan_representation: SpreadsheetNanRepresentation = (
        SpreadsheetNanRepresentation.NAN
    ),
    add_filename_prefix: bool = True,
    filename_prefix: str = "MyExpt_",
    artifact_batch: RuntimeArtifactBatch,
) -> dict[str, str | bytes]:
    """Render one plate's exact contract-selected spreadsheet file bundle."""

    return render_spreadsheet_bundle(
        artifact_batch,
        delimiter=delimiter,
        add_image_metadata=add_image_metadata,
        add_image_file_names=add_image_file_names,
        select_measurements=select_measurements,
        selected_columns=selected_columns,
        calculate_aggregate_means=calculate_aggregate_means,
        calculate_aggregate_medians=calculate_aggregate_medians,
        calculate_aggregate_standard_deviations=(
            calculate_aggregate_standard_deviations
        ),
        output_directory=output_directory,
        export_all_measurement_types=export_all_measurement_types,
        file_selections=file_selections,
        nan_representation=nan_representation,
        add_filename_prefix=add_filename_prefix,
        filename_prefix=filename_prefix,
    )


class ExportToSpreadsheetModule(ArtifactExportModule):
    """Executable plate-scoped CellProfiler spreadsheet export declaration."""

    module_name = "ExportToSpreadsheet"
    function_name = "export_to_spreadsheet"
    validated = True
    confidence = 1.0

    delimiter_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the column delimiter"
    )
    add_metadata_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Add image metadata columns to your object data file?"
    )
    add_file_names_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Add image file and folder names to your object data file?"
    )
    select_measurements_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the measurements to export",
        ("Select measurements to export",),
    )
    aggregate_means_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate the per-image mean values for object measurements?"
    )
    aggregate_medians_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate the per-image median values for object measurements?"
    )
    aggregate_standard_deviations_setting: ClassVar[SettingNameFamily] = (
        SettingNameFamily(
            "Calculate the per-image standard deviation values for object measurements?"
        )
    )
    output_directory_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Output file location"
    )
    gene_pattern_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Create a GenePattern GCT file?"
    )
    gene_name_source_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select source of sample row name"
    )
    gene_image_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the image to use as the identifier"
    )
    gene_metadata_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the metadata to use as the identifier"
    )
    export_all_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Export all measurement types?"
    )
    selected_columns_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Press button to select measurements"
    )
    nan_representation_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Representation of Nan/Inf"
    )
    add_prefix_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Add a prefix to file names?"
    )
    prefix_setting: ClassVar[SettingNameFamily] = SettingNameFamily("Filename prefix")
    overwrite_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Overwrite existing files without warning?"
    )
    excel_size_limit_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Limit output to a size that is allowed in Excel"
    )
    data_to_export_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Data to export"
    )
    combine_with_previous_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Combine these object measurements with those of the previous object?"
    )
    file_name_setting: ClassVar[SettingNameFamily] = SettingNameFamily("File name")
    automatic_file_name_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Use the object name for the file name?"
    )

    setting_bindings = (
        SettingToKeywordBinding(
            delimiter_setting,
            "delimiter",
            SpreadsheetDelimiter.from_cellprofiler,
        ),
        SettingToKeywordBinding(
            add_metadata_setting,
            "add_image_metadata",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            add_file_names_setting,
            "add_image_file_names",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            select_measurements_setting,
            "select_measurements",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            selected_columns_setting,
            "selected_columns",
            SpreadsheetColumnSelection.from_cellprofiler,
        ),
        SettingToKeywordBinding(
            aggregate_means_setting,
            "calculate_aggregate_means",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_medians_setting,
            "calculate_aggregate_medians",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_standard_deviations_setting,
            "calculate_aggregate_standard_deviations",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            output_directory_setting,
            "output_directory",
            cellprofiler_output_directory,
        ),
        SettingToKeywordBinding(
            export_all_setting,
            "export_all_measurement_types",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            nan_representation_setting,
            "nan_representation",
            SpreadsheetNanRepresentation.from_cellprofiler,
        ),
        SettingToKeywordBinding(
            add_prefix_setting,
            "add_filename_prefix",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(prefix_setting, "filename_prefix", str),
        SettingToKeywordBinding(
            overwrite_setting,
            parse=parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def uses_cellprofiler_runtime_adapter(cls) -> bool:
        """Spreadsheet rendering runs through the generic plate executor."""

        return False

    @classmethod
    def ignored_settings_for(
        cls,
        module: ModuleBlock,
    ) -> tuple[str | SettingNameFamily, ...]:
        """Return inactive CP rows that do not participate in CSV rendering."""

        ignored: list[str | SettingNameFamily] = []
        gene_pattern = module.get_setting(cls.gene_pattern_setting.canonical, "")
        if not gene_pattern or not parse_cellprofiler_bool(gene_pattern):
            ignored.extend(
                (
                    cls.gene_pattern_setting,
                    cls.gene_name_source_setting,
                    cls.gene_image_setting,
                    cls.gene_metadata_setting,
                )
            )
        excel_size_limit = module.get_setting(
            cls.excel_size_limit_setting.canonical,
            "No",
        )
        if not parse_cellprofiler_bool(excel_size_limit):
            ignored.append(cls.excel_size_limit_setting)
        return tuple(ignored)

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: ModuleBlock,
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        """Parse repeated export rows and reject unsupported active behavior."""

        gene_pattern = module.get_setting(cls.gene_pattern_setting.canonical, "No")
        if parse_cellprofiler_bool(gene_pattern):
            raise ValueError(
                "ExportToSpreadsheet GenePattern GCT output is not declared by the "
                "OpenHCS file-bundle contract."
            )
        excel_size_limit = module.get_setting(
            cls.excel_size_limit_setting.canonical,
            "No",
        )
        if parse_cellprofiler_bool(excel_size_limit):
            raise ValueError(
                "ExportToSpreadsheet Excel row and column truncation is not "
                "declared by the OpenHCS file-bundle contract."
            )
        kwargs = dict(bound.kwargs)
        kwargs["file_selections"] = cls._file_selections(
            module,
            delimiter=kwargs.get("delimiter", SpreadsheetDelimiter.COMMA),
            export_all=bool(kwargs.get("export_all_measurement_types", True)),
        )
        unmapped = dict(bound.unmapped_kwargs)
        for setting in (
            cls.data_to_export_setting,
            cls.combine_with_previous_setting,
            cls.file_name_setting,
            cls.automatic_file_name_setting,
        ):
            unmapped.pop(cls.normalize_setting_name(setting.canonical), None)
        return BoundModuleSettings(
            kwargs,
            unmapped,
            bound.setting_coverage,
        )

    @classmethod
    def _file_selections(
        cls,
        module: ModuleBlock,
        *,
        delimiter: SpreadsheetDelimiter,
        export_all: bool,
    ) -> tuple[SpreadsheetFileSelection, ...]:
        """Collapse CP's repeated output rows into typed file selections."""

        if export_all:
            return ()
        selections: list[SpreadsheetFileSelection] = []
        for block in repeating_setting_blocks(
            module.iter_settings(),
            start_name=cls.data_to_export_setting,
        ):
            subject = block_setting_value(block, cls.data_to_export_setting).strip()
            if not subject or is_blank_symbol_name(subject):
                continue
            combine = parse_cellprofiler_bool(
                block_setting_value(
                    block,
                    cls.combine_with_previous_setting,
                    default="No",
                )
            )
            automatic = parse_cellprofiler_bool(
                block_setting_value(
                    block,
                    cls.automatic_file_name_setting,
                    default="Yes",
                )
            )
            file_name = (
                f"{subject}{delimiter.default_suffix}"
                if automatic
                else cellprofiler_metadata_template(
                    block_setting_value(block, cls.file_name_setting)
                )
            )
            if not file_name:
                raise ValueError(
                    f"ExportToSpreadsheet subject {subject!r} has no output file name."
                )
            if combine:
                if not selections:
                    raise ValueError(
                        "ExportToSpreadsheet cannot combine its first data row with "
                        "a previous output."
                    )
                previous = selections[-1]
                selections[-1] = SpreadsheetFileSelection(
                    subjects=(*previous.subjects, subject),
                    file_name=previous.file_name,
                )
            else:
                selections.append(SpreadsheetFileSelection((subject,), file_name))
        return tuple(selections)

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Select the exact ordered tables exported by this plate module."""

        del (
            module,
            invocation_key,
        )
        return ArtifactSpecCollection(
            spec.for_plan_type(ArtifactInputPlan)
            for spec in step_context.available_artifacts.specs
            if spec.artifact_type
            in (MeasurementsArtifactType, RelationshipsArtifactType)
        ).unique(conflict_context="ExportToSpreadsheet input")

    @classmethod
    def artifact_contract_outputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Declare the materialized spreadsheet bundle."""
        return (
            ArtifactSpec.output(
                cls._file_bundle_artifact_name(
                    module,
                    invocation_key=invocation_key,
                    step_context=step_context,
                ),
                SpecialArtifactType,
                materialization=MaterializationSpec(
                    FileBundleOptions(),
                    write_mode=(
                        WriteMode.OVERWRITE
                        if parse_cellprofiler_bool(
                            module.get_setting(cls.overwrite_setting.canonical, "Yes")
                        )
                        else WriteMode.ERROR
                    ),
                ),
            ),
        )

    @classmethod
    def _file_bundle_artifact_name(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> str:
        step_index = step_context.step_index
        if not isinstance(step_index, int):
            raise TypeError(
                "ExportToSpreadsheet requires an integer step index for its file "
                "bundle identity."
            )
        suffix = str(step_index + 1)
        if invocation_key.position:
            suffix = f"{suffix}_{invocation_key.position + 1}"
        return f"{module.name}_{suffix}_files"
