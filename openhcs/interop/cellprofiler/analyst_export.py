"""CellProfiler Analyst export projections.

This module intentionally builds render-only CPA views from existing OpenHCS
runtime artifacts. It does not make CPA tables a new semantic authority.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_semantics import MeasurementScope, measurement_row_mapping
from openhcs.core.runtime_stores import RuntimeValueStore, StoredRuntimeValue
from openhcs.core.runtime_table_projection import (
    RuntimeProjectedColumnIdentity,
    RuntimeProjectedColumnRole,
    RuntimeTableProjectionDialect,
)
from openhcs.core.runtime_values import ColumnarRows, MeasurementTable, ObjectRelationship

from .database_column_dialect import CellProfilerDatabaseColumnDialect


class CellProfilerObjectTableMode(str, Enum):
    """ExportToDatabase object-table layout requested by a CP pipeline."""

    PER_OBJECT = "per_object"
    COMBINED = "combined"
    VIEW = "view"


@dataclass(frozen=True, slots=True)
class CellProfilerExecutionExportContext:
    """Nominal execution boundary required to derive a CPA export."""

    prepared: object
    execution: object
    runtime_stores_by_axis: Mapping[str, RuntimeValueStore]
    output_roots: tuple[Path, ...]
    source_workspace_root: Path
    export_root: Path

    def __post_init__(self) -> None:
        if not self.runtime_stores_by_axis:
            raise ValueError(
                "CellProfilerExecutionExportContext.runtime_stores_by_axis "
                "cannot be empty."
            )
        for axis_id, store in self.runtime_stores_by_axis.items():
            if not axis_id:
                raise ValueError(
                    "CellProfilerExecutionExportContext axis ids cannot be empty."
                )
            if not isinstance(store, RuntimeValueStore):
                raise TypeError(
                    "CellProfilerExecutionExportContext.runtime_stores_by_axis "
                    f"values must be RuntimeValueStore, got {type(store).__name__}."
                )
        object.__setattr__(
            self,
            "output_roots",
            tuple(Path(output_root) for output_root in self.output_roots),
        )
        object.__setattr__(
            self,
            "source_workspace_root",
            Path(self.source_workspace_root),
        )
        object.__setattr__(self, "export_root", Path(self.export_root))


@dataclass(frozen=True, slots=True)
class CellProfilerDatabaseExportSettings:
    """Subset of CellProfiler ExportToDatabase settings needed for CPA projection."""

    database_type: Literal["sqlite"]
    sqlite_file: str
    experiment_name: str
    table_prefix: str
    object_table_mode: CellProfilerObjectTableMode
    selected_objects: tuple[str, ...] | None
    wants_properties_file: bool
    wants_relationship_tables: bool

    def __post_init__(self) -> None:
        if self.database_type != "sqlite":
            raise ValueError(
                "CellProfilerDatabaseExportSettings.database_type must be 'sqlite'."
            )
        if not self.sqlite_file:
            raise ValueError("CellProfilerDatabaseExportSettings.sqlite_file is required.")
        if not self.experiment_name:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.experiment_name is required."
            )
        object.__setattr__(
            self,
            "object_table_mode",
            (
                self.object_table_mode
                if isinstance(self.object_table_mode, CellProfilerObjectTableMode)
                else CellProfilerObjectTableMode(self.object_table_mode)
            ),
        )
        if self.selected_objects is not None:
            normalized = tuple(str(name) for name in self.selected_objects)
            if any(not name for name in normalized):
                raise ValueError(
                    "CellProfilerDatabaseExportSettings.selected_objects cannot "
                    "contain empty object names."
                )
            object.__setattr__(self, "selected_objects", normalized)


@dataclass(frozen=True, slots=True)
class CPAImageChannelSpec:
    """One CPA image channel backed by a monochrome OpenHCS image identity."""

    alias: str
    image_name: str
    channel_color: str
    channels_per_image: int = 1

    def __post_init__(self) -> None:
        for field_name, value in (
            ("alias", self.alias),
            ("image_name", self.image_name),
            ("channel_color", self.channel_color),
        ):
            if not value:
                raise ValueError(f"CPAImageChannelSpec.{field_name} is required.")
        if self.channels_per_image <= 0:
            raise ValueError("CPAImageChannelSpec.channels_per_image must be positive.")


@dataclass(frozen=True, slots=True)
class CellProfilerAnalystExportRequest:
    """Complete nominal request for a CPA export projection."""

    settings: CellProfilerDatabaseExportSettings
    context: CellProfilerExecutionExportContext
    image_channels: tuple[CPAImageChannelSpec, ...]

    def __post_init__(self) -> None:
        if not self.image_channels:
            raise ValueError(
                "CellProfilerAnalystExportRequest.image_channels cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class CPAImageRow:
    """One row for CPA's image table."""

    image_number: int
    measurements: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CPAObjectTable:
    """Rows for one CPA object table."""

    object_name: str
    table_name: str
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CPARelationshipTable:
    """Rows for one CPA relationship table."""

    table_name: str
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CellProfilerAnalystProjection:
    """Render-only CPA database view derived from runtime stores."""

    image_table_name: str
    image_rows: tuple[CPAImageRow, ...]
    object_tables: tuple[CPAObjectTable, ...]
    relationship_tables: tuple[CPARelationshipTable, ...]


@dataclass(frozen=True, slots=True)
class CPAPropertiesFile:
    """One CellProfiler Analyst properties file render."""

    object_name: str | None
    file_name: str
    properties: Mapping[str, str]
    text: str


@dataclass(frozen=True, slots=True)
class CPATableRowProjection:
    """Authoritative row projection for CPA-compatible table records."""

    dialect: RuntimeTableProjectionDialect

    def measurement_rows(self, rows: Any) -> tuple[Mapping[str, Any], ...]:
        if isinstance(rows, ColumnarRows):
            return tuple(dict(row) for row in rows.row_mappings())
        if isinstance(rows, Mapping):
            return self.mapping_columns_to_rows(rows)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            return tuple(dict(measurement_row_mapping(row)) for row in rows)
        return (dict(measurement_row_mapping(rows)),)

    def relationship_rows(
        self,
        record: StoredRuntimeValue,
        relationship: ObjectRelationship,
    ) -> tuple[Mapping[str, Any], ...]:
        rows = self.mapping_columns_to_rows(record.value.data)
        for row in rows:
            self.required_int(row, relationship.source.id_field, record.key.name)
            self.required_int(row, relationship.target.id_field, record.key.name)
        return rows

    def mapping_columns_to_rows(
        self,
        rows: Mapping[Any, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        columns = {str(column): value for column, value in rows.items()}
        lengths = tuple(
            len(value)
            for value in columns.values()
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        )
        if not lengths:
            return (columns,)
        row_count = max(lengths)
        return tuple(
            {
                column: (
                    value[row_index]
                    if isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    and row_index < len(value)
                    else value
                )
                for column, value in columns.items()
            }
            for row_index in range(row_count)
        )

    def require_object_rows(
        self,
        *,
        table: MeasurementTable,
        rows: tuple[Mapping[str, Any], ...],
    ) -> None:
        image_id_column = self.dialect.column_name(
            RuntimeProjectedColumnIdentity(RuntimeProjectedColumnRole.IMAGE_ID)
        )
        object_id_column = self.dialect.column_name(
            RuntimeProjectedColumnIdentity(RuntimeProjectedColumnRole.OBJECT_ID)
        )
        for row in rows:
            self.required_int(row, image_id_column, table.name)
            self.required_int(row, object_id_column, table.name)

    def required_int(
        self,
        row: Mapping[str, Any],
        field_name: str,
        table_name: str,
    ) -> int:
        if field_name not in row:
            raise ValueError(
                f"CPA export requires field '{field_name}' in table '{table_name}'."
            )
        value = row[field_name]
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CPA export requires integer field '{field_name}' in table "
                f"'{table_name}', got {value!r}."
            ) from exc


@dataclass(frozen=True, slots=True)
class CellProfilerAnalystProjectionBuilder:
    """Build CPA projection records from typed OpenHCS runtime values."""

    dialect: RuntimeTableProjectionDialect | None = None

    def __post_init__(self) -> None:
        if self.dialect is not None and not isinstance(
            self.dialect,
            RuntimeTableProjectionDialect,
        ):
            raise TypeError(
                "CellProfilerAnalystProjectionBuilder.dialect must be "
                f"RuntimeTableProjectionDialect, got {type(self.dialect).__name__}."
            )

    def build(
        self,
        request: CellProfilerAnalystExportRequest,
    ) -> CellProfilerAnalystProjection:
        dialect = self._dialect(request)
        row_projection = CPATableRowProjection(dialect)
        image_rows_by_number: dict[int, dict[str, Any]] = {}
        object_rows_by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        relationship_rows_by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

        for axis_id, store in request.context.runtime_stores_by_axis.items():
            self._collect_measurements(
                store=store,
                axis_id=axis_id,
                settings=request.settings,
                row_projection=row_projection,
                image_rows_by_number=image_rows_by_number,
                object_rows_by_name=object_rows_by_name,
            )
            if request.settings.wants_relationship_tables:
                self._collect_relationships(
                    store=store,
                    axis_id=axis_id,
                    row_projection=row_projection,
                    relationship_rows_by_name=relationship_rows_by_name,
                )

        object_tables = tuple(
            CPAObjectTable(
                object_name=object_name,
                table_name=dialect.object_table_name(object_name),
                rows=tuple(rows),
            )
            for object_name, rows in sorted(object_rows_by_name.items())
        )
        relationship_tables = tuple(
            CPARelationshipTable(
                table_name=dialect.relationship_table_name(table_name),
                rows=tuple(rows),
            )
            for table_name, rows in sorted(relationship_rows_by_name.items())
        )
        return CellProfilerAnalystProjection(
            image_table_name=dialect.image_table_name(),
            image_rows=tuple(
                CPAImageRow(image_number=image_number, measurements=row)
                for image_number, row in sorted(image_rows_by_number.items())
            ),
            object_tables=object_tables,
            relationship_tables=relationship_tables,
        )

    def _collect_measurements(
        self,
        *,
        store: RuntimeValueStore,
        axis_id: str,
        settings: CellProfilerDatabaseExportSettings,
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
        object_rows_by_name: dict[str, list[Mapping[str, Any]]],
    ) -> None:
        for record in store.find(kind=ArtifactKind.MEASUREMENTS, axis_id=axis_id):
            table = MeasurementTable.from_runtime_value(record.value)
            if table.subject is None:
                raise ValueError(
                    f"Measurement table '{table.name}' has no measurement subject."
                )
            rows = row_projection.measurement_rows(table.rows)
            if table.subject.scope is MeasurementScope.IMAGE:
                self._collect_image_rows(
                    table=table,
                    rows=rows,
                    row_projection=row_projection,
                    target=image_rows_by_number,
                )
                continue
            if table.subject.scope is MeasurementScope.OBJECT:
                object_name = table.subject.name
                if object_name is None:
                    raise ValueError(
                        f"Object measurement table '{table.name}' has no object name."
                    )
                if (
                    settings.selected_objects is not None
                    and object_name not in settings.selected_objects
                ):
                    continue
                row_projection.require_object_rows(table=table, rows=rows)
                object_rows_by_name[object_name].extend(rows)

    def _collect_image_rows(
        self,
        *,
        table: MeasurementTable,
        rows: tuple[Mapping[str, Any], ...],
        row_projection: CPATableRowProjection,
        target: dict[int, dict[str, Any]],
    ) -> None:
        image_id_column = row_projection.dialect.column_name(
            RuntimeProjectedColumnIdentity(RuntimeProjectedColumnRole.IMAGE_ID)
        )
        for row in rows:
            image_number = row_projection.required_int(
                row,
                image_id_column,
                table.name,
            )
            target.setdefault(image_number, {image_id_column: image_number}).update(row)

    def _collect_relationships(
        self,
        *,
        store: RuntimeValueStore,
        axis_id: str,
        row_projection: CPATableRowProjection,
        relationship_rows_by_name: dict[str, list[Mapping[str, Any]]],
    ) -> None:
        for record in store.find(kind=ArtifactKind.RELATIONSHIPS, axis_id=axis_id):
            relationship = ObjectRelationship.from_runtime_value(record.value)
            rows = row_projection.relationship_rows(record, relationship)
            relationship_rows_by_name[record.key.name].extend(rows)

    def _dialect(
        self,
        request: CellProfilerAnalystExportRequest,
    ) -> RuntimeTableProjectionDialect:
        if self.dialect is not None:
            return self.dialect
        return CellProfilerDatabaseColumnDialect(request.settings.table_prefix)


@dataclass(frozen=True, slots=True)
class CPAPropertiesRenderer:
    """Render CPA properties text from a projection and export request."""

    dialect: RuntimeTableProjectionDialect | None = None

    def render(
        self,
        request: CellProfilerAnalystExportRequest,
        projection: CellProfilerAnalystProjection,
    ) -> tuple[CPAPropertiesFile, ...]:
        if not request.settings.wants_properties_file:
            return ()
        object_tables: tuple[CPAObjectTable | None, ...]
        object_tables = projection.object_tables or (None,)
        return tuple(
            self._render_for_object_table(
                request=request,
                projection=projection,
                dialect=self._dialect(request),
                object_table=object_table,
            )
            for object_table in object_tables
        )

    def _render_for_object_table(
        self,
        *,
        request: CellProfilerAnalystExportRequest,
        projection: CellProfilerAnalystProjection,
        dialect: RuntimeTableProjectionDialect,
        object_table: CPAObjectTable | None,
    ) -> CPAPropertiesFile:
        properties = self._properties(
            request=request,
            projection=projection,
            dialect=dialect,
            object_table=object_table,
        )
        object_name = None if object_table is None else object_table.object_name
        return CPAPropertiesFile(
            object_name=object_name,
            file_name=self._file_name(request=request, object_name=object_name),
            properties=properties,
            text="\n".join(
                f"{key} = {value}" for key, value in properties.items()
            )
            + "\n",
        )

    def _properties(
        self,
        *,
        request: CellProfilerAnalystExportRequest,
        projection: CellProfilerAnalystProjection,
        dialect: RuntimeTableProjectionDialect,
        object_table: CPAObjectTable | None,
    ) -> Mapping[str, str]:
        object_table_name = "" if object_table is None else object_table.table_name
        object_id = (
            ""
            if object_table is None
            else self._object_id_column(dialect, object_table.object_name)
        )
        return {
            "db_type": request.settings.database_type,
            "db_sqlite_file": str(
                request.context.export_root / request.settings.sqlite_file
            ),
            "image_table": projection.image_table_name,
            "object_table": object_table_name,
            "image_id": dialect.column_name(
                RuntimeProjectedColumnIdentity(RuntimeProjectedColumnRole.IMAGE_ID)
            ),
            "object_id": object_id,
            "plate_id": dialect.column_name(
                RuntimeProjectedColumnIdentity(
                    RuntimeProjectedColumnRole.METADATA,
                    metadata_key="Plate",
                )
            ),
            "well_id": dialect.column_name(
                RuntimeProjectedColumnIdentity(
                    RuntimeProjectedColumnRole.METADATA,
                    metadata_key="Well",
                )
            ),
            "series_id": dialect.column_name(
                RuntimeProjectedColumnIdentity(
                    RuntimeProjectedColumnRole.GROUP,
                    metadata_key="Number",
                )
            ),
            "group_id": dialect.column_name(
                RuntimeProjectedColumnIdentity(
                    RuntimeProjectedColumnRole.GROUP,
                    metadata_key="Number",
                )
            ),
            "timepoint_id": dialect.column_name(
                RuntimeProjectedColumnIdentity(
                    RuntimeProjectedColumnRole.GROUP,
                    metadata_key="Index",
                )
            ),
            "cell_x_loc": self._location_column(dialect, object_table, "X"),
            "cell_y_loc": self._location_column(dialect, object_table, "Y"),
            "cell_z_loc": self._location_column(dialect, object_table, "Z"),
            "image_path_cols": ",".join(
                dialect.column_name(
                    RuntimeProjectedColumnIdentity(
                        RuntimeProjectedColumnRole.SOURCE_IMAGE_PATH,
                        source_image_name=channel.image_name,
                    )
                )
                for channel in request.image_channels
            ),
            "image_file_cols": ",".join(
                dialect.column_name(
                    RuntimeProjectedColumnIdentity(
                        RuntimeProjectedColumnRole.SOURCE_IMAGE_FILE,
                        source_image_name=channel.image_name,
                    )
                )
                for channel in request.image_channels
            ),
            "image_names": ",".join(
                channel.image_name for channel in request.image_channels
            ),
            "image_channel_colors": ",".join(
                channel.channel_color for channel in request.image_channels
            ),
            "channels_per_image": ",".join(
                str(channel.channels_per_image)
                for channel in request.image_channels
            ),
        }

    def _file_name(
        self,
        *,
        request: CellProfilerAnalystExportRequest,
        object_name: str | None,
    ) -> str:
        sqlite_stem = Path(request.settings.sqlite_file).stem
        prefix = request.settings.table_prefix.rstrip("_")
        base_name = "_".join(part for part in (sqlite_stem, prefix) if part)
        if object_name is None:
            return f"{base_name}.properties"
        return f"{base_name}_{object_name}.properties"

    def _dialect(
        self,
        request: CellProfilerAnalystExportRequest,
    ) -> RuntimeTableProjectionDialect:
        if self.dialect is not None:
            return self.dialect
        return CellProfilerDatabaseColumnDialect(request.settings.table_prefix)

    @staticmethod
    def _object_id_column(
        dialect: RuntimeTableProjectionDialect,
        object_name: str,
    ) -> str:
        if isinstance(dialect, CellProfilerDatabaseColumnDialect):
            return dialect.object_id_column(object_name, qualified=True)
        return dialect.column_name(
            RuntimeProjectedColumnIdentity(
                RuntimeProjectedColumnRole.OBJECT_ID,
                object_name=object_name,
            )
        )

    @staticmethod
    def _location_column(
        dialect: RuntimeTableProjectionDialect,
        object_table: CPAObjectTable | None,
        axis_name: str,
    ) -> str:
        if object_table is None:
            return ""
        return dialect.column_name(
            RuntimeProjectedColumnIdentity(
                RuntimeProjectedColumnRole.OBJECT_LOCATION,
                object_name=object_table.object_name,
                axis_name=axis_name,
            )
        )
