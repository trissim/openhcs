"""CellProfiler Analyst export projections.

This module intentionally builds render-only CPA views from existing OpenHCS
runtime artifacts. It does not make CPA tables a new semantic authority.
"""

from __future__ import annotations

from base64 import b64encode
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import md5
from io import BytesIO
from numbers import Integral, Real
from pathlib import Path
import sqlite3
from typing import cast, Any, ClassVar, Literal

import numpy as np
from PIL import Image

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.measurement_row_materialization import (
    WideMeasurementRowAccumulator,
)
from openhcs.core.equivalence import measurement_qualifier_field_names
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementTableRowLayout,
    measurement_table_row_layout_from_fields,
)
from openhcs.core.runtime_identifier import (
    normalize_runtime_identifier,
)
from openhcs.core.runtime_stores import RuntimeArtifactBatch, StoredRuntimeValue
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_image_values import image_payload_data, image_payload_metadata
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.source_image_provenance import (
    source_component_metadata_consensus,
    SourceImageProvenance,
)
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.source_metadata import SourceMetadataRoleView, SourceMetadataScalar
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjectionAuthority,
)

from .database_column_dialect import (
    CellProfilerDatabaseColumnDialect,
    CellProfilerExperimentProjectionField,
    CellProfilerImageAggregateStatistic,
    CellProfilerProjectedTable,
    CellProfilerSourceImageProjectionField,
)
from .image_set_numbering import CellProfilerImageSetNumbering
from .measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
    cellprofiler_projected_measurement_feature_name,
)
from .module_declarations import CellProfilerModule
from .source_metadata import CellProfilerSourceMetadataField


class CellProfilerObjectTableMode(str, Enum):
    """ExportToDatabase object-table layout requested by a CP pipeline."""

    def __new__(cls, value: str, cellprofiler_literal: str):
        member = str.__new__(cls, value)
        member._value_ = value
        member._cellprofiler_literal = cellprofiler_literal
        return member

    PER_OBJECT = ("per_object", "one table per object type")
    COMBINED = ("combined", "single object table")
    VIEW = ("view", "single object view")

    @property
    def cellprofiler_literal(self) -> str:
        return self._cellprofiler_literal

    @classmethod
    def from_cellprofiler(cls, value: str) -> "CellProfilerObjectTableMode":
        """Parse a public enum value or CellProfiler UI literal."""

        normalized = value.strip().casefold()
        matches = tuple(
            member
            for member in cls
            if normalized in {member.value, member.cellprofiler_literal}
        )
        if len(matches) != 1:
            raise ValueError(
                f"Unsupported ExportToDatabase object table mode {value!r}."
            )
        return matches[0]


class CellProfilerRelationshipProjectionName(str, Enum):
    """Canonical SQLite object names owned by CellProfiler relationships."""

    TYPES = "Per_RelationshipTypes"
    ROWS = "Per_Relationships"
    VIEW = "Per_RelationshipsView"


class CPAPropertyName(str, Enum):
    """Canonical static keys in a CellProfiler Analyst properties export."""

    DATABASE_TYPE = "db_type"
    SQLITE_FILE = "db_sqlite_file"
    IMAGE_TABLE = "image_table"
    OBJECT_TABLE = "object_table"
    IMAGE_ID = "image_id"
    OBJECT_ID = "object_id"
    PLATE_ID = "plate_id"
    WELL_ID = "well_id"
    SERIES_ID = "series_id"
    GROUP_ID = "group_id"
    TIMEPOINT_ID = "timepoint_id"
    CELL_X_LOCATION = "cell_x_loc"
    CELL_Y_LOCATION = "cell_y_loc"
    CELL_Z_LOCATION = "cell_z_loc"
    IMAGE_PATH_COLUMNS = "image_path_cols"
    IMAGE_FILE_COLUMNS = "image_file_cols"
    IMAGE_NAMES = "image_names"
    IMAGE_CHANNEL_COLORS = "image_channel_colors"
    CHANNELS_PER_IMAGE = "channels_per_image"
    IMAGE_THUMBNAIL_COLUMNS = "image_thumbnail_cols"
    IMAGE_CHANNEL_BLEND_MODES = "image_channel_blend_modes"
    IMAGE_URL_PREPEND = "image_url_prepend"
    OBJECT_NAME = "object_name"
    PLATE_TYPE = "plate_type"
    CLASSIFIER_IGNORE_COLUMNS = "classifier_ignore_columns"
    IMAGE_TILE_SIZE = "image_tile_size"
    IMAGE_SIZE = "image_size"
    CLASSIFICATION_TYPE = "classification_type"
    TRAINING_SET = "training_set"
    AREA_SCORING_COLUMN = "area_scoring_column"
    CLASS_TABLE = "class_table"
    CHECK_TABLES = "check_tables"
    FORCE_BIOFORMATS = "force_bioformats"
    USE_LEGACY_FETCHER = "use_legacy_fetcher"
    PROCESS_3D = "process_3D"

    def normalized_value(self, value: str) -> str:
        if self is CPAPropertyName.SQLITE_FILE:
            return Path(value).name
        return ",".join(part.strip() for part in value.split(","))


class CPAExperimentPropertyColumn(str, Enum):
    """Canonical columns in CellProfiler's Experiment_Properties table."""

    EXPERIMENT_ID = "experiment_id"
    OBJECT_NAME = "object_name"
    FIELD = "field"
    VALUE = "value"


def _source_image_projection_values(
    source_path: Path,
    source_image_name: str,
    dialect: CellProfilerDatabaseColumnDialect,
) -> Mapping[str, Any]:
    with Image.open(source_path) as source_image:
        width, height = source_image.size
        source_dtype = np.asarray(source_image).dtype
    if np.issubdtype(source_dtype, np.integer):
        scaling = float(np.iinfo(source_dtype).max)
    elif np.issubdtype(source_dtype, np.bool_):
        scaling = 1.0
    else:
        scaling = 1.0
    values_by_field = {
        CellProfilerSourceImageProjectionField.FRAME: 0,
        CellProfilerSourceImageProjectionField.HEIGHT: height,
        CellProfilerSourceImageProjectionField.MD5_DIGEST: md5(
            source_path.read_bytes(),
            usedforsecurity=False,
        ).hexdigest(),
        CellProfilerSourceImageProjectionField.SCALING: scaling,
        CellProfilerSourceImageProjectionField.SERIES: 0,
        CellProfilerSourceImageProjectionField.URL: source_path.resolve().as_uri(),
        CellProfilerSourceImageProjectionField.WIDTH: width,
    }
    return {
        dialect.source_image_feature_field(
            source_image_name,
            field.field_spec,
        ).name: value
        for field, value in values_by_field.items()
    }


def _merge_projected_row_values(
    target: dict[str, Any],
    additions: Mapping[str, Any],
    *,
    owner: str,
) -> None:
    for field_name, value in additions.items():
        if field_name in target and target[field_name] != value:
            raise ValueError(
                f"{owner} has conflicting values for field {field_name!r}: "
                f"{target[field_name]!r} != {value!r}."
            )
        target[field_name] = value


def _thumbnail_png_base64(pixels: np.ndarray, *, auto_scale: bool) -> str:
    image = np.asarray(pixels)
    if image.dtype == bool:
        normalized = image.astype(np.float64)
    elif np.issubdtype(image.dtype, np.integer):
        normalized = image.astype(np.float64) / float(np.iinfo(image.dtype).max)
    elif np.issubdtype(image.dtype, np.floating):
        normalized = image.astype(np.float64, copy=False)
    else:
        raise TypeError(
            "CPA thumbnails require numeric or boolean image payloads, "
            f"got {image.dtype}."
        )
    if auto_scale and image.dtype != bool:
        normalized = (normalized - normalized.min()) / normalized.max()
    if normalized.ndim == 2:
        pil_image = Image.fromarray((normalized * 255).astype("uint8"), "L")
    elif normalized.ndim == 3 and normalized.shape[-1] == 3:
        pil_image = Image.fromarray((normalized * 255).astype("uint8"), "RGB")
    else:
        raise ValueError(
            "CPA thumbnails require a 2D grayscale or three-channel RGB image, "
            f"got shape {normalized.shape!r}."
        )
    major_axis = max(pil_image.size)
    minor_axis = 200 * min(pil_image.size) // major_axis
    thumbnail_size = (
        (200, minor_axis) if pil_image.size[0] == major_axis else (minor_axis, 200)
    )
    pil_image = pil_image.resize(thumbnail_size)
    output = BytesIO()
    pil_image.save(output, format="PNG")
    return b64encode(output.getvalue()).decode()


@dataclass(frozen=True, slots=True)
class CellProfilerDatabaseExportSettings:
    """Subset of CellProfiler ExportToDatabase settings needed for CPA projection."""

    sqlite_file: str
    experiment_name: str
    table_prefix: str
    object_table_mode: CellProfilerObjectTableMode
    selected_objects: tuple[str, ...] | None
    wants_properties_file: bool
    wants_relationship_tables: bool
    maximum_column_name_length: int = 64
    location_object: str | None = None
    plate_type: str | None = None
    plate_metadata: str = "Plate"
    well_metadata: str = "Well"
    image_url_prepend: str = ""
    group_fields: tuple[tuple[str, str], ...] = ()
    classification_type: Literal["object", "image"] = "object"
    phenotype_class_table: str = ""
    calculate_per_image_mean: bool = False
    calculate_per_image_median: bool = False
    calculate_per_image_standard_deviation: bool = False
    write_image_thumbnails: bool = False
    thumbnail_image_names: tuple[str, ...] = ()
    auto_scale_thumbnail_intensities: bool = True

    def __post_init__(self) -> None:
        if not self.sqlite_file:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.sqlite_file is required."
            )
        if not self.experiment_name:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.experiment_name is required."
            )
        if isinstance(self.maximum_column_name_length, bool) or not isinstance(
            self.maximum_column_name_length,
            int,
        ):
            raise TypeError(
                "CellProfilerDatabaseExportSettings.maximum_column_name_length "
                "must be an integer."
            )
        if not 10 <= self.maximum_column_name_length <= 64:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.maximum_column_name_length "
                "must be between 10 and 64."
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
        if self.location_object is not None:
            normalized_location = str(self.location_object).strip()
            object.__setattr__(
                self,
                "location_object",
                normalized_location or None,
            )
        plate_type = (
            None
            if self.plate_type is None
            else str(self.plate_type).strip() or None
        )
        plate_metadata = str(self.plate_metadata).strip()
        well_metadata = str(self.well_metadata).strip()
        if not plate_metadata:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.plate_metadata is required."
            )
        if not well_metadata:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.well_metadata is required."
            )
        object.__setattr__(self, "plate_type", plate_type)
        object.__setattr__(self, "plate_metadata", plate_metadata)
        object.__setattr__(self, "well_metadata", well_metadata)
        object.__setattr__(self, "image_url_prepend", str(self.image_url_prepend))
        normalized_groups = tuple(
            (str(name).strip(), str(columns).strip())
            for name, columns in self.group_fields
        )
        if any(not name or not columns for name, columns in normalized_groups):
            raise ValueError(
                "CellProfilerDatabaseExportSettings.group_fields requires non-empty "
                "group names and column expressions."
            )
        object.__setattr__(self, "group_fields", normalized_groups)
        if self.classification_type not in {"object", "image"}:
            raise ValueError(
                "CellProfilerDatabaseExportSettings.classification_type must be "
                "'object' or 'image'."
            )
        object.__setattr__(
            self,
            "phenotype_class_table",
            str(self.phenotype_class_table).strip(),
        )
        thumbnail_names = tuple(
            dict.fromkeys(str(name).strip() for name in self.thumbnail_image_names)
        )
        if any(not name for name in thumbnail_names):
            raise ValueError(
                "CellProfilerDatabaseExportSettings.thumbnail_image_names cannot "
                "contain empty names."
            )
        object.__setattr__(self, "thumbnail_image_names", thumbnail_names)

    def exports_object(self, object_name: str) -> bool:
        """Return whether this export includes one declared object subject."""

        return self.selected_objects is None or object_name in self.selected_objects


@dataclass(frozen=True, slots=True)
class CPAImageChannelSpec:
    """One CPA image channel backed by a monochrome OpenHCS image identity."""

    alias: str
    image_name: str
    channel_color: str
    channels_per_image: int = 1

    DEFAULT_CHANNEL_COLORS: ClassVar[tuple[str, ...]] = (
        "red",
        "green",
        "blue",
        "cyan",
        "magenta",
        "yellow",
        "gray",
    )

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

    @classmethod
    def defaults_for_artifacts(
        cls,
        image_specs: Sequence[ArtifactSpec],
        *,
        source_binding_plan: CompiledSourceBindingPlan,
    ) -> tuple["CPAImageChannelSpec", ...]:
        """Derive CellProfiler's default CPA channel declarations in source order."""

        if not isinstance(source_binding_plan, CompiledSourceBindingPlan):
            raise TypeError(
                "CPAImageChannelSpec.defaults_for_artifacts requires a "
                "CompiledSourceBindingPlan."
            )
        specs = tuple(image_specs)
        for spec in specs:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "CPAImageChannelSpec.defaults_for_artifacts requires "
                    f"ArtifactSpec values, got {type(spec).__name__}."
                )
            if spec.artifact_type is not ImageArtifactType:
                raise TypeError(
                    "CPAImageChannelSpec.defaults_for_artifacts requires image "
                    f"artifacts, got {spec.artifact_type.require_value()}:{spec.name}."
                )
        specs_by_ref = {
            spec.ref().for_plan_type(ArtifactInputPlan): spec for spec in specs
        }
        if len(specs_by_ref) != len(specs):
            raise ValueError(
                "CPAImageChannelSpec.defaults_for_artifacts requires unique image "
                "artifact references."
            )
        source_refs = tuple(
            binding.input_spec().ref()
            for binding in source_binding_plan.binding_declarations
            if binding.input_spec().ref() in specs_by_ref
        )
        ordered_specs = tuple(specs_by_ref[ref] for ref in source_refs) + tuple(
            spec
            for spec in specs
            if spec.ref().for_plan_type(ArtifactInputPlan) not in source_refs
        )
        return tuple(
            cls(
                alias=spec.name,
                image_name=spec.name,
                channel_color=(
                    cls.DEFAULT_CHANNEL_COLORS[index]
                    if index < len(cls.DEFAULT_CHANNEL_COLORS)
                    else "none"
                ),
            )
            for index, spec in enumerate(ordered_specs)
        )


@dataclass(frozen=True, slots=True)
class CPARelationshipTable:
    """Rows for one CPA relationship table."""

    table_name: str
    rows: tuple[Mapping[str, Any], ...]
    relationship: str
    object_name1: str
    object_name2: str
    module_number: int | None = None


@dataclass(frozen=True, slots=True)
class CellProfilerAnalystProjection:
    """Render-only CPA database view derived from runtime stores."""

    image_table: CellProfilerProjectedTable
    object_tables: tuple[CellProfilerProjectedTable, ...]
    relationship_tables: tuple[CPARelationshipTable, ...]
    experiment_table: CellProfilerProjectedTable
    image_channels: tuple[CPAImageChannelSpec, ...] = ()

    def database_column_names(
        self,
        settings: CellProfilerDatabaseExportSettings,
    ) -> tuple[str, ...]:
        """Return the complete raw field universe used by tables and properties."""

        names = [
            field_spec.name
            for table in (self.image_table, self.experiment_table, *self.object_tables)
            for field_spec in table.columns
        ]
        if settings.object_table_mode is not CellProfilerObjectTableMode.PER_OBJECT:
            raw_dialect = CellProfilerDatabaseColumnDialect(settings.table_prefix)
            selected_object_tables = tuple(
                table
                for table in self.object_tables
                if settings.exports_object(
                    _required_table_subject_name(table, MeasurementScope.OBJECT)
                )
            )
            combined_table = CPASQLiteRenderer._combined_object_table(
                selected_object_tables,
                raw_dialect.combined_object_table_name(),
                raw_dialect,
            )
            names.extend(field_spec.name for field_spec in combined_table.columns)
        names.extend(
            field_spec.name
            for field_spec in CPAPropertiesRenderer.property_fields(
                settings,
                self.image_channels,
                self,
            )
        )
        return tuple(dict.fromkeys(names))

    def database_dialect(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        settings: CellProfilerDatabaseExportSettings,
    ) -> CellProfilerDatabaseColumnDialect:
        """Resolve this projection's single external column-name dialect."""

        mapping = dialect.column_name_mapping
        if mapping is not None:
            if mapping.maximum_length != settings.maximum_column_name_length:
                raise ValueError(
                    "CellProfiler projection dialect maximum-column length does "
                    "not match its export settings."
                )
            return dialect
        return dialect.with_column_names(
            self.database_column_names(settings),
            settings.maximum_column_name_length,
        )


def _required_table_subject(
    table: CellProfilerProjectedTable,
    scope: MeasurementScope,
) -> MeasurementSubject:
    subject = table.subject
    if subject is None or subject.scope is not scope:
        raise ValueError(
            f"Projected table {table.table_name!r} requires a {scope.value} subject."
        )
    return subject


def _required_table_subject_name(
    table: CellProfilerProjectedTable,
    scope: MeasurementScope,
) -> str:
    subject = _required_table_subject(table, scope)
    if subject.name is None:
        raise ValueError(
            f"Projected table {table.table_name!r} requires a named subject."
        )
    return subject.name


@dataclass(frozen=True, slots=True)
class CPAPropertiesFile:
    """One CellProfiler Analyst properties file render."""

    object_name: str | None
    file_name: str
    properties: Mapping[str, str]
    text: str


@dataclass(slots=True)
class CPATableRowProjection:
    """Authoritative row projection for CPA-compatible table records."""

    dialect: CellProfilerDatabaseColumnDialect
    image_set_numbering: CellProfilerImageSetNumbering
    context: ProcessingContext | None = None

    def measurement_rows_by_subject(
        self,
        table: MeasurementTable,
        *,
        scope: RuntimeExecutionAxisScope | None,
    ) -> Mapping[
        MeasurementSubject,
        tuple[Mapping[str, Any], ...],
    ]:
        projected_rows = table.rows
        if table.subject.scope is not MeasurementScope.EXPERIMENT:
            if scope is None:
                raise ValueError(
                    f"Measurement table {table.name!r} requires an execution scope."
                )
            projected_rows = self.image_set_numbering.project_measurement_rows(
                scope=scope,
                table=table,
            )

        default_subject = table.subject.name or table.subject.scope.value.title()
        if (
            table.subject.scope is MeasurementScope.EXPERIMENT
            and measurement_table_row_layout_from_fields(table.rows.fields)
            is MeasurementTableRowLayout.WIDE
        ):
            return {
                table.subject: tuple(
                    self._project_runtime_row(table, row, subject=table.subject)
                    for row in projected_rows
                )
            }
        accumulator = WideMeasurementRowAccumulator(
            CELLPROFILER_MEASUREMENT_DIALECT.row_identity_contract
        )
        accumulator.add(
            projected_rows,
            cellprofiler_projected_measurement_feature_name,
            default_subject=default_subject,
            default_scope=table.subject.scope,
            source_image_name=table.source_image_name,
            object_id_field=table.subject.object_id_field,
            qualifier_field_names=measurement_qualifier_field_names(
                CELLPROFILER_MEASUREMENT_DIALECT
            ),
        )
        object_subjects = accumulator.object_subjects()
        object_subject_set = frozenset(object_subjects)
        rows_by_subject: dict[
            MeasurementSubject,
            tuple[Mapping[str, Any], ...],
        ] = {}
        for subject_name, subject_rows in accumulator.row_mappings_by_subject().items():
            subject = (
                MeasurementSubject(MeasurementScope.OBJECT, subject_name)
                if subject_name in object_subject_set
                else MeasurementSubject(table.subject.scope, subject_name)
            )
            rows_by_subject[subject] = tuple(
                self._project_runtime_row(table, row, subject=subject)
                for row in subject_rows
            )
        return rows_by_subject

    def measurement_columns(
        self,
        table: MeasurementTable,
        rows: Sequence[Mapping[str, Any]],
        *,
        subject: MeasurementSubject,
    ) -> tuple[FieldSpec, ...]:
        """Project declared table fields while retaining row-owned feature names."""

        layout = measurement_table_row_layout_from_fields(table.rows.fields)
        structural_fields = (
            (self.image_id_field(), self.object_id_field(subject))
            if subject.scope is MeasurementScope.OBJECT
            else (
                (self.image_id_field(),)
                if subject.scope is MeasurementScope.IMAGE
                else ()
            )
        )
        projected_table_fields = tuple(
            projected
            for field in table.rows.fields
            for projected in (
                self._project_measurement_field(
                    table,
                    field.name,
                    subject=subject,
                ),
            )
            if projected is not None
        )
        declared_fields = FieldSpec.merge_exact(
            (
                structural_fields,
                (
                    ()
                    if layout is MeasurementTableRowLayout.LONG
                    else projected_table_fields
                ),
            ),
            context=f"CPA table {table.name!r} declared fields",
        )
        declared_by_name = {
            field_spec.name: field_spec
            for field_spec in (*structural_fields, *projected_table_fields)
        }
        value_fields = tuple(
            field_spec
            for field_spec in table.rows.fields
            if field_spec.name in MeasurementRowValueField.field_names()
        )
        dynamic_dtype = value_fields[0].dtype if value_fields else None
        row_fields = tuple(
            declared_by_name.get(
                field_name,
                FieldSpec(field_name, dynamic_dtype, required=False),
            )
            for field_name in dict.fromkeys(
                field_name for row in rows for field_name in row
            )
        )
        return FieldSpec.merge_exact(
            (
                declared_fields,
                row_fields,
            ),
            context=f"CPA table {table.name!r} fields",
        )

    def _project_measurement_field(
        self,
        table: MeasurementTable,
        field_name: str,
        *,
        subject: MeasurementSubject,
    ) -> FieldSpec | None:
        normalized_field_name = normalize_runtime_identifier(field_name)
        row_identity = CELLPROFILER_MEASUREMENT_DIALECT.row_identity_contract
        if row_identity.selected_image_identity_fields(
            frozenset((normalized_field_name,))
        ):
            return self.image_id_field()
        if subject.scope is MeasurementScope.OBJECT and (
            field_name == table.subject.object_id_field
            or row_identity.selected_object_identity_field(
                frozenset((normalized_field_name,))
            )
            is not None
        ):
            return self.object_id_field(subject)
        if field_name in MeasurementRowAxisField.field_names():
            return None
        fields_by_name = {field.name: field for field in table.rows.fields}
        source_field = fields_by_name.get(field_name)
        if source_field is None:
            value_fields = tuple(
                field
                for field in table.rows.fields
                if field.name in MeasurementRowValueField.field_names()
            )
            value_field = value_fields[0] if value_fields else None
            source_field = FieldSpec(
                field_name,
                dtype=None if value_field is None else value_field.dtype,
                required=False,
            )
        feature_name = cellprofiler_projected_measurement_feature_name(
            field_name,
            (),
        )
        projected_field = FieldSpec(
            feature_name,
            dtype=source_field.dtype,
            required=source_field.required,
        )
        owner = table.measurement_feature_owner
        if owner is not None and issubclass(owner, CellProfilerModule):
            projected_field = owner.database_measurement_field(projected_field)
        return self.dialect.measurement_field(
            subject,
            projected_field,
        )

    def _project_runtime_row(
        self,
        table: MeasurementTable,
        row: Mapping[str, Any],
        *,
        subject: MeasurementSubject,
    ) -> Mapping[str, Any]:
        projected_row: dict[str, Any] = {}
        for field_name, value in row.items():
            field_spec = self._project_measurement_field(
                table,
                field_name,
                subject=subject,
            )
            if field_spec is None:
                continue
            if field_spec.name in projected_row:
                raise ValueError(
                    f"CPA row projection for table '{table.name}' would overwrite "
                    f"field {field_spec.name!r}."
                )
            projected_row[field_spec.name] = value
        return projected_row

    def image_id_field(self) -> FieldSpec:
        return self.dialect.image_id_field()

    def object_id_field(self, subject: MeasurementSubject) -> FieldSpec:
        return self.dialect.object_id_field(subject)

    def relationship_rows(
        self,
        record: StoredRuntimeValue,
        relationship: ObjectRelationship,
    ) -> tuple[Mapping[str, Any], ...]:
        slice_indices = relationship.payload.slice_indices or (0,) * len(
            relationship.payload.source_ids
        )
        image_numbers_by_slice = self.image_set_numbering.for_source_slices(
            scope=record.key.scope,
            provenance=relationship.source_provenance,
            slice_indices=tuple(dict.fromkeys(slice_indices)),
            owner=relationship.name,
        )
        return tuple(
            {
                "image_number1": image_numbers_by_slice[slice_index],
                "object_number1": int(source_id),
                "image_number2": image_numbers_by_slice[slice_index],
                "object_number2": int(target_id),
            }
            for source_id, target_id, slice_index in zip(
                relationship.payload.source_ids,
                relationship.payload.target_ids,
                slice_indices,
                strict=True,
            )
        )

    def collect_image_provenance(
        self,
        provenance: SourceImageProvenance,
        *,
        scope: RuntimeExecutionAxisScope,
        source_image_name: str,
        image_rows_by_number: dict[int, dict[str, Any]],
        source_metadata_by_image_number: dict[
            int,
            list[Mapping[str, SourceMetadataScalar]],
        ],
        thumbnail_field: FieldSpec | None = None,
        auto_scale_thumbnail_intensities: bool = True,
    ) -> None:
        """Fold typed source provenance directly into projected image rows."""

        plane_count = provenance.source_plane_count
        plane_indices = range(plane_count) if plane_count > 0 else range(1)
        for plane_index in plane_indices:
            plane_provenance = provenance.for_source_plane(plane_index)
            values: dict[str, Any] = {}
            source_path = self._resolved_source_path(plane_provenance.source_path)
            if source_path is not None:
                values[self.dialect.source_image_path_field(source_image_name).name] = (
                    str(source_path.parent)
                )
                values[self.dialect.source_image_file_field(source_image_name).name] = (
                    source_path.name
                )
                if source_path.is_file():
                    values.update(
                        _source_image_projection_values(
                            source_path,
                            source_image_name,
                            self.dialect,
                        )
                    )
                    if thumbnail_field is not None:
                        with Image.open(source_path) as source_image:
                            pixels = np.asarray(source_image)
                        values[thumbnail_field.name] = _thumbnail_png_base64(
                            pixels,
                            auto_scale=auto_scale_thumbnail_intensities,
                        )
            component_metadata = plane_provenance.source_component_metadata
            metadata_items = {
                field_name: value
                for field_name, value in self.dialect.source_metadata_defaults().items()
            }
            if source_path is not None:
                metadata_items[
                    CellProfilerSourceMetadataField.FILE_LOCATION.field_name
                ] = source_path.as_uri()
            if component_metadata is not None:
                metadata_items.update(
                    SourceMetadataRoleView(component_metadata).original_items()
                )
            image_number = self.image_set_numbering.for_source_slice(
                scope=scope,
                provenance=provenance,
                slice_index=plane_index,
                owner=source_image_name,
            )
            image_id = self.image_id_field()
            target = image_rows_by_number.setdefault(
                image_number,
                {image_id.name: image_number},
            )
            _merge_projected_row_values(
                target,
                values,
                owner=f"CPA image {image_number}",
            )
            source_metadata_by_image_number[image_number].append(metadata_items)

    def image_numbers_for_provenance(
        self,
        provenance: SourceImageProvenance,
        *,
        scope: RuntimeExecutionAxisScope,
        owner: str,
    ) -> tuple[int, ...]:
        plane_count = provenance.source_plane_count
        plane_indices = range(plane_count) if plane_count > 0 else range(1)
        return tuple(
            self.image_set_numbering.for_source_slice(
                scope=scope,
                provenance=provenance,
                slice_index=plane_index,
                owner=owner,
            )
            for plane_index in plane_indices
        )

    def _resolved_source_path(self, source_path: str | None) -> Path | None:
        if source_path is None:
            return None
        if self.context is None:
            return Path(source_path).resolve(strict=False)
        if self.context.filemanager is None:
            raise RuntimeError(
                "Virtual workspace source resolution requires a runtime FileManager."
            )
        projection = VirtualWorkspaceSourceProjectionAuthority.from_context(
            self.context,
            cache=self.context.runtime_source_workspace_projection_cache,
        ).projection_if_available()
        if projection is None:
            return Path(source_path).resolve(strict=False)
        return Path(
            projection.resolved_source_path_for(
                VirtualWorkspacePathLookup.from_paths(source_path, source_path),
                self.context.filemanager,
            )
        ).resolve(strict=False)

    def required_int(
        self,
        row: Mapping[str, Any],
        field_spec: FieldSpec,
        table_name: str,
    ) -> int:
        if field_spec.name not in row:
            raise ValueError(
                f"CPA export requires field {field_spec.name!r} in table "
                f"'{table_name}'."
            )
        value = row[field_spec.name]
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CPA export requires integer field {field_spec.name!r} in "
                f"table '{table_name}', got {value!r}."
            ) from exc


@dataclass(frozen=True, slots=True)
class CellProfilerAnalystProjectionBuilder:
    """Build CPA projection records from typed OpenHCS runtime values."""

    source_binding_plan: CompiledSourceBindingPlan
    dialect: CellProfilerDatabaseColumnDialect | None = None
    context: ProcessingContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_binding_plan, CompiledSourceBindingPlan):
            raise TypeError(
                "CellProfilerAnalystProjectionBuilder.source_binding_plan must be "
                "CompiledSourceBindingPlan."
            )
        if self.dialect is not None and not isinstance(
            self.dialect,
            CellProfilerDatabaseColumnDialect,
        ):
            raise TypeError(
                "CellProfilerAnalystProjectionBuilder.dialect must be "
                "CellProfilerDatabaseColumnDialect, got "
                f"{type(self.dialect).__name__}."
            )

    def build(
        self,
        artifact_batch: RuntimeArtifactBatch,
        settings: CellProfilerDatabaseExportSettings,
        image_channels: Sequence[CPAImageChannelSpec],
    ) -> CellProfilerAnalystProjection:
        if not isinstance(artifact_batch, RuntimeArtifactBatch):
            raise TypeError(
                "CellProfilerAnalystProjectionBuilder.build requires "
                "RuntimeArtifactBatch."
            )
        if not isinstance(settings, CellProfilerDatabaseExportSettings):
            raise TypeError(
                "CellProfilerAnalystProjectionBuilder.build requires "
                "CellProfilerDatabaseExportSettings."
            )
        self._validate_image_channels(artifact_batch, image_channels)

        dialect = self._dialect(settings)
        row_projection = CPATableRowProjection(
            dialect,
            CellProfilerImageSetNumbering(
                artifact_batch.source_image_set_identity_policy
            ),
            self.context,
        )
        image_subject = MeasurementSubject(MeasurementScope.IMAGE, "Image")
        image_rows_by_number: dict[
            int,
            dict[str, Any],
        ] = {}
        source_metadata_by_image_number: dict[
            int,
            list[Mapping[str, SourceMetadataScalar]],
        ] = defaultdict(list)
        image_columns: tuple[FieldSpec, ...] = (
            row_projection.image_id_field(),
            *(
                field_spec
                for channel in image_channels
                for field_spec in (
                    dialect.source_image_path_field(channel.alias),
                    dialect.source_image_file_field(channel.alias),
                    *(
                        dialect.source_image_feature_field(
                            channel.alias,
                            feature.field_spec,
                        )
                        for feature in CellProfilerSourceImageProjectionField
                    ),
                )
            ),
            *(
                dialect.group_field(group_name)
                for group_name in ("Index", "Length", "Number")
            ),
            *(
                dialect.thumbnail_field(image_name)
                for image_name in settings.thumbnail_image_names
            ),
        )
        experiment_rows: list[Mapping[str, Any]] = []
        experiment_columns = tuple(
            field.field_spec for field in CellProfilerExperimentProjectionField
        )
        object_rows_by_subject: dict[
            MeasurementSubject,
            dict[tuple[int, int], dict[str, Any]],
        ] = {}
        object_columns_by_subject: dict[
            MeasurementSubject,
            tuple[FieldSpec, ...],
        ] = {}
        relationship_rows_by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(
            list
        )
        relationship_declarations_by_name: dict[
            str,
            tuple[str, str, str, int | None],
        ] = {}

        measurement_records = artifact_batch.records_of_type(MeasurementsArtifactType)
        image_records = artifact_batch.records_of_type(ImageArtifactType)
        relationship_records = artifact_batch.records_of_type(RelationshipsArtifactType)
        for axis_id in artifact_batch.records_by_axis:
            image_columns, experiment_columns = self._collect_measurements(
                records=measurement_records[axis_id],
                row_projection=row_projection,
                image_rows_by_number=image_rows_by_number,
                image_columns=image_columns,
                experiment_rows=experiment_rows,
                experiment_columns=experiment_columns,
                object_rows_by_subject=object_rows_by_subject,
                object_columns_by_subject=object_columns_by_subject,
            )
            self._collect_measurement_provenance(
                records=measurement_records[axis_id],
                image_channels=image_channels,
                settings=settings,
                row_projection=row_projection,
                image_rows_by_number=image_rows_by_number,
                source_metadata_by_image_number=source_metadata_by_image_number,
            )
            self._collect_image_provenance(
                records=image_records[axis_id],
                image_channels=image_channels,
                settings=settings,
                row_projection=row_projection,
                image_rows_by_number=image_rows_by_number,
                source_metadata_by_image_number=source_metadata_by_image_number,
            )
            if settings.write_image_thumbnails:
                self._collect_image_thumbnails(
                    records=image_records[axis_id],
                    settings=settings,
                    row_projection=row_projection,
                    image_rows_by_number=image_rows_by_number,
                )
            if settings.wants_relationship_tables:
                self._collect_relationships(
                    records=relationship_records[axis_id],
                    row_projection=row_projection,
                    relationship_rows_by_name=relationship_rows_by_name,
                    relationship_declarations_by_name=(
                        relationship_declarations_by_name
                    ),
                )
        all_measurement_tables = tuple(
            cast(MeasurementTable, record.value.data)
            for records in measurement_records.values()
            for record in records
        )
        for table in CellProfilerModule.derive_experiment_measurement_tables(
            all_measurement_tables
        ):
            image_columns, experiment_columns = self._collect_measurement_table(
                table=table,
                scope=None,
                row_projection=row_projection,
                image_rows_by_number=image_rows_by_number,
                image_columns=image_columns,
                experiment_rows=experiment_rows,
                experiment_columns=experiment_columns,
                object_rows_by_subject=object_rows_by_subject,
                object_columns_by_subject=object_columns_by_subject,
            )
        source_metadata_columns = self._collect_common_source_metadata(
            source_metadata_by_image_number=source_metadata_by_image_number,
            declared_fields=self.source_binding_plan.metadata_fields,
            row_projection=row_projection,
            image_rows_by_number=image_rows_by_number,
        )
        image_columns = FieldSpec.merge_exact(
            (image_columns, source_metadata_columns),
            context=f"CPA table {dialect.image_table_name()!r} fields",
        )
        self._collect_image_group_values(
            image_rows_by_number,
            row_projection=row_projection,
        )

        object_table_values: list[CellProfilerProjectedTable] = []
        for subject, rows in object_rows_by_subject.items():
            object_name = subject.object_name
            if object_name is None:
                raise ValueError("CPA object table requires an object subject.")
            object_table_values.append(
                CellProfilerProjectedTable(
                    table_name=dialect.object_table_name(object_name),
                    rows=tuple(rows.values()),
                    columns=object_columns_by_subject.get(subject, ()),
                    subject=subject,
                )
            )
        object_tables = tuple(object_table_values)
        image_columns = self._collect_image_aggregates(
            settings=settings,
            object_tables=tuple(
                table
                for table in object_tables
                if table.subject is not None
                and table.subject.object_name is not None
                and settings.exports_object(table.subject.object_name)
            ),
            image_rows_by_number=image_rows_by_number,
            image_columns=image_columns,
            row_projection=row_projection,
        )
        relationship_tables = tuple(
            CPARelationshipTable(
                table_name=dialect.relationship_table_name(table_name),
                rows=tuple(rows),
                relationship=relationship_declarations_by_name[table_name][0],
                object_name1=relationship_declarations_by_name[table_name][1],
                object_name2=relationship_declarations_by_name[table_name][2],
                module_number=relationship_declarations_by_name[table_name][3],
            )
            for table_name, rows in relationship_rows_by_name.items()
        )
        return CellProfilerAnalystProjection(
            image_table=CellProfilerProjectedTable(
                table_name=dialect.image_table_name(),
                rows=tuple(
                    row for _image_number, row in sorted(image_rows_by_number.items())
                ),
                columns=image_columns,
                subject=image_subject,
            ),
            object_tables=object_tables,
            relationship_tables=relationship_tables,
            experiment_table=CellProfilerProjectedTable(
                table_name=dialect.object_table_name("Experiment"),
                rows=tuple(experiment_rows or ({},)),
                columns=experiment_columns,
                subject=MeasurementSubject(
                    MeasurementScope.EXPERIMENT,
                    "Experiment",
                ),
            ),
            image_channels=tuple(image_channels),
        )

    def _collect_measurements(
        self,
        *,
        records: Sequence[StoredRuntimeValue],
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
        object_rows_by_subject: dict[
            MeasurementSubject,
            dict[tuple[int, int], dict[str, Any]],
        ],
        image_columns: tuple[FieldSpec, ...],
        experiment_rows: list[Mapping[str, Any]],
        experiment_columns: tuple[FieldSpec, ...],
        object_columns_by_subject: dict[
            MeasurementSubject,
            tuple[FieldSpec, ...],
        ],
    ) -> tuple[tuple[FieldSpec, ...], tuple[FieldSpec, ...]]:
        for record in records:
            table = cast(MeasurementTable, record.value.data)
            image_columns, experiment_columns = self._collect_measurement_table(
                table=table,
                scope=record.key.scope,
                row_projection=row_projection,
                image_rows_by_number=image_rows_by_number,
                image_columns=image_columns,
                experiment_rows=experiment_rows,
                experiment_columns=experiment_columns,
                object_rows_by_subject=object_rows_by_subject,
                object_columns_by_subject=object_columns_by_subject,
            )
        return image_columns, experiment_columns

    def _collect_measurement_table(
        self,
        *,
        table: MeasurementTable,
        scope: RuntimeExecutionAxisScope | None,
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
        image_columns: tuple[FieldSpec, ...],
        experiment_rows: list[Mapping[str, Any]],
        experiment_columns: tuple[FieldSpec, ...],
        object_rows_by_subject: dict[
            MeasurementSubject,
            dict[tuple[int, int], dict[str, Any]],
        ],
        object_columns_by_subject: dict[
            MeasurementSubject,
            tuple[FieldSpec, ...],
        ],
    ) -> tuple[tuple[FieldSpec, ...], tuple[FieldSpec, ...]]:
        if table.subject.scope not in {
            MeasurementScope.IMAGE,
            MeasurementScope.OBJECT,
            MeasurementScope.EXPERIMENT,
        }:
            return image_columns, experiment_columns
        rows_by_subject = row_projection.measurement_rows_by_subject(
            table,
            scope=scope,
        )
        if not rows_by_subject:
            rows_by_subject = {table.subject: ()}
        for subject, rows in rows_by_subject.items():
            columns = row_projection.measurement_columns(
                table,
                rows,
                subject=subject,
            )
            if subject.scope is MeasurementScope.IMAGE:
                image_table_name = row_projection.dialect.image_table_name()
                image_columns = FieldSpec.merge_exact(
                    (image_columns, columns),
                    context=f"CPA table {image_table_name!r} fields",
                )
                self._collect_image_rows(
                    table=table,
                    rows=rows,
                    row_projection=row_projection,
                    target=image_rows_by_number,
                )
                continue
            if subject.scope is MeasurementScope.EXPERIMENT:
                experiment_table_name = row_projection.dialect.object_table_name(
                    "Experiment"
                )
                experiment_columns = FieldSpec.merge_exact(
                    (experiment_columns, columns),
                    context=f"CPA table {experiment_table_name!r} fields",
                )
                self._merge_experiment_rows(
                    table=table,
                    rows=rows,
                    target=experiment_rows,
                )
                continue
            object_name = subject.object_name
            if object_name is None:
                raise ValueError("CPA object table requires an object subject.")
            object_rows = object_rows_by_subject.setdefault(subject, {})
            object_table_name = row_projection.dialect.object_table_name(object_name)
            object_columns_by_subject[subject] = FieldSpec.merge_exact(
                (object_columns_by_subject.get(subject, ()), columns),
                context=f"CPA table {object_table_name!r} fields",
            )
            self._merge_object_rows(
                table=table,
                rows=rows,
                subject=subject,
                row_projection=row_projection,
                target=object_rows,
            )
        return image_columns, experiment_columns

    def _collect_image_rows(
        self,
        *,
        table: MeasurementTable,
        rows: tuple[Mapping[str, Any], ...],
        row_projection: CPATableRowProjection,
        target: dict[int, dict[str, Any]],
    ) -> None:
        image_id_field = row_projection.image_id_field()
        for row in rows:
            image_number = row_projection.required_int(
                row,
                image_id_field,
                table.name,
            )
            projected_row = target.setdefault(
                image_number,
                {image_id_field.name: image_number},
            )
            _merge_projected_row_values(
                projected_row,
                row,
                owner=f"CPA image {image_number}",
            )

    @staticmethod
    def _merge_experiment_rows(
        *,
        table: MeasurementTable,
        rows: Sequence[Mapping[str, Any]],
        target: list[Mapping[str, Any]],
    ) -> None:
        if len(rows) > 1:
            raise ValueError(
                f"CPA experiment measurement table '{table.name}' must contain "
                f"at most one row, got {len(rows)}."
            )
        if not rows:
            return
        if not target:
            target.append(dict(rows[0]))
            return
        merged = dict(target[0])
        for field_name, value in rows[0].items():
            if field_name in merged and merged[field_name] != value:
                raise ValueError(
                    "CPA experiment measurements have conflicting values for "
                    f"field {field_name!r}: {merged[field_name]!r} != {value!r}."
                )
            merged[field_name] = value
        target[0] = merged

    def _collect_relationships(
        self,
        *,
        records: Sequence[StoredRuntimeValue],
        row_projection: CPATableRowProjection,
        relationship_rows_by_name: dict[str, list[Mapping[str, Any]]],
        relationship_declarations_by_name: dict[
            str,
            tuple[str, str, str, int | None],
        ],
    ) -> None:
        for record in records:
            relationship = cast(ObjectRelationship, record.value.data)
            relationship_declaration = relationship.declaration
            declaration = (
                relationship_declaration.relationship_type,
                relationship_declaration.source.name,
                relationship_declaration.target.name,
                relationship_declaration.producer_module_number,
            )
            existing = relationship_declarations_by_name.get(relationship.name)
            if existing is not None and existing != declaration:
                raise ValueError(
                    f"CPA relationship artifact '{relationship.name}' has "
                    f"conflicting declarations: {existing!r} != {declaration!r}."
                )
            relationship_declarations_by_name[relationship.name] = declaration
            rows = row_projection.relationship_rows(record, relationship)
            relationship_rows_by_name[relationship.name].extend(rows)

    @staticmethod
    def _collect_image_provenance(
        *,
        records: Sequence[StoredRuntimeValue],
        image_channels: Sequence[CPAImageChannelSpec],
        settings: CellProfilerDatabaseExportSettings,
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
        source_metadata_by_image_number: dict[
            int,
            list[Mapping[str, SourceMetadataScalar]],
        ],
    ) -> None:
        for channel in image_channels:
            channel_records = tuple(
                record for record in records if record.key.name == channel.alias
            )
            for record in channel_records:
                provenance = image_payload_metadata(record.value.data).source_provenance
                row_projection.collect_image_provenance(
                    provenance,
                    scope=record.key.scope,
                    source_image_name=channel.alias,
                    image_rows_by_number=image_rows_by_number,
                    source_metadata_by_image_number=source_metadata_by_image_number,
                )

    @staticmethod
    def _collect_measurement_provenance(
        *,
        records: Sequence[StoredRuntimeValue],
        image_channels: Sequence[CPAImageChannelSpec],
        settings: CellProfilerDatabaseExportSettings,
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
        source_metadata_by_image_number: dict[
            int,
            list[Mapping[str, SourceMetadataScalar]],
        ],
    ) -> None:
        channel_aliases = frozenset(channel.alias for channel in image_channels)
        for record in records:
            table = cast(MeasurementTable, record.value.data)
            provenance = table.source_provenance
            for source_image_name in provenance.represented_source_image_names:
                if source_image_name not in channel_aliases:
                    continue
                selected_provenance = provenance.for_source_image(source_image_name)
                thumbnail_field = (
                    row_projection.dialect.thumbnail_field(source_image_name)
                    if source_image_name in settings.thumbnail_image_names
                    else None
                )
                row_projection.collect_image_provenance(
                    selected_provenance,
                    scope=record.key.scope,
                    source_image_name=source_image_name,
                    image_rows_by_number=image_rows_by_number,
                    source_metadata_by_image_number=source_metadata_by_image_number,
                    thumbnail_field=thumbnail_field,
                    auto_scale_thumbnail_intensities=(
                        settings.auto_scale_thumbnail_intensities
                    ),
                )

    @staticmethod
    def _collect_image_group_values(
        image_rows_by_number: dict[int, dict[str, Any]],
        *,
        row_projection: CPATableRowProjection,
    ) -> None:
        group_fields = tuple(
            row_projection.dialect.group_field(group_name)
            for group_name in ("Index", "Length", "Number")
        )
        image_numbers = tuple(sorted(image_rows_by_number))
        image_count = len(image_numbers)
        for index, image_number in enumerate(image_numbers, start=1):
            row = image_rows_by_number[image_number]
            row[group_fields[0].name] = index
            row[group_fields[1].name] = image_count
            row[group_fields[2].name] = 1

    @classmethod
    def _collect_image_thumbnails(
        cls,
        *,
        records: Sequence[StoredRuntimeValue],
        settings: CellProfilerDatabaseExportSettings,
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
    ) -> None:
        image_id_field = row_projection.image_id_field()
        for image_name in settings.thumbnail_image_names:
            for record in records:
                if record.key.name != image_name:
                    continue
                payload = record.value.data
                provenance = image_payload_metadata(payload).source_provenance
                image_numbers = row_projection.image_numbers_for_provenance(
                    provenance,
                    scope=record.key.scope,
                    owner=image_name,
                )
                pixels = np.asarray(image_payload_data(payload))
                plane_pixels = cls._thumbnail_planes(pixels, len(image_numbers))
                thumbnail_field = row_projection.dialect.thumbnail_field(image_name)
                for image_number, plane in zip(
                    image_numbers,
                    plane_pixels,
                    strict=True,
                ):
                    target = image_rows_by_number.setdefault(
                        image_number,
                        {image_id_field.name: image_number},
                    )
                    _merge_projected_row_values(
                        target,
                        {
                            thumbnail_field.name: _thumbnail_png_base64(
                                plane,
                                auto_scale=settings.auto_scale_thumbnail_intensities,
                            )
                        },
                        owner=f"CPA image {image_number}",
                    )

    @staticmethod
    def _thumbnail_planes(
        pixels: np.ndarray,
        plane_count: int,
    ) -> tuple[np.ndarray, ...]:
        if plane_count <= 1:
            return (pixels,)
        if pixels.ndim < 3 or pixels.shape[0] != plane_count:
            raise ValueError(
                "CPA thumbnail image stack does not match its source-plane count: "
                f"shape={pixels.shape!r}, planes={plane_count}."
            )
        return tuple(pixels[index] for index in range(plane_count))

    @classmethod
    def _collect_image_aggregates(
        cls,
        *,
        settings: CellProfilerDatabaseExportSettings,
        object_tables: Sequence[CellProfilerProjectedTable],
        image_rows_by_number: dict[int, dict[str, Any]],
        image_columns: tuple[FieldSpec, ...],
        row_projection: CPATableRowProjection,
    ) -> tuple[FieldSpec, ...]:
        statistics = tuple(
            (name, reducer)
            for enabled, name, reducer in (
                (
                    settings.calculate_per_image_mean,
                    CellProfilerImageAggregateStatistic.MEAN,
                    np.nanmean,
                ),
                (
                    settings.calculate_per_image_median,
                    CellProfilerImageAggregateStatistic.MEDIAN,
                    np.nanmedian,
                ),
                (
                    settings.calculate_per_image_standard_deviation,
                    CellProfilerImageAggregateStatistic.STANDARD_DEVIATION,
                    np.nanstd,
                ),
            )
            if enabled
        )
        if not statistics:
            return image_columns
        image_id_field = row_projection.image_id_field()
        aggregate_fields: list[FieldSpec] = []
        for object_table in object_tables:
            subject = object_table.subject
            if subject is None or subject.object_name is None:
                raise ValueError("CPA object aggregate requires an object subject.")
            object_id_field = row_projection.object_id_field(subject)
            feature_fields = tuple(
                field_spec
                for field_spec in object_table.columns
                if field_spec.name not in {image_id_field.name, object_id_field.name}
                and cls._aggregate_field_is_numeric(
                    field_spec,
                    object_table.rows,
                )
            )
            rows_by_image: dict[
                int,
                list[Mapping[str, Any]],
            ] = defaultdict(list)
            for row in object_table.rows:
                rows_by_image[int(row[image_id_field.name])].append(row)
            for statistic_name, reducer in statistics:
                for feature_field in feature_fields:
                    aggregate_field = row_projection.dialect.image_aggregate_field(
                        statistic_name,
                        feature_field,
                    )
                    aggregate_fields.append(aggregate_field)
                    for image_number, rows in rows_by_image.items():
                        values = np.asarray(
                            [
                                float(row[feature_field.name])
                                for row in rows
                                if row.get(feature_field.name) is not None
                            ],
                            dtype=float,
                        )
                        value = None if not values.size else float(reducer(values))
                        image_rows_by_number.setdefault(
                            image_number,
                            {image_id_field.name: image_number},
                        )[aggregate_field.name] = value
        return FieldSpec.merge_exact(
            (image_columns, tuple(aggregate_fields)),
            context=f"CPA table {row_projection.dialect.image_table_name()!r} fields",
        )

    @staticmethod
    def _aggregate_field_is_numeric(
        field_spec: FieldSpec,
        rows: Sequence[Mapping[str, Any]],
    ) -> bool:
        if field_spec.dtype is not None:
            return (
                field_spec.dtype is int
                or field_spec.dtype is float
                or field_spec.dtype
                in {
                    "int",
                    "integer",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float",
                    "float16",
                    "float32",
                    "float64",
                    "real",
                    "double",
                }
            )
        values = tuple(
            row[field_spec.name] for row in rows if row.get(field_spec.name) is not None
        )
        return bool(values) and all(
            isinstance(value, Real) and not isinstance(value, bool) for value in values
        )

    @classmethod
    def _collect_common_source_metadata(
        cls,
        *,
        source_metadata_by_image_number: Mapping[
            int,
            Sequence[Mapping[str, SourceMetadataScalar]],
        ],
        declared_fields: Sequence[FieldSpec],
        row_projection: CPATableRowProjection,
        image_rows_by_number: dict[int, dict[str, Any]],
    ) -> tuple[FieldSpec, ...]:
        source_fields = FieldSpec.merge_exact(
            (tuple(declared_fields),),
            context="CPA declared source metadata fields",
        )
        declared_by_name = {field.name: field for field in source_fields}
        for image_number, metadata_rows in source_metadata_by_image_number.items():
            if not metadata_rows:
                continue
            consensus = source_component_metadata_consensus(metadata_rows)
            common_metadata = {} if consensus is None else consensus
            projected_metadata = {
                row_projection.dialect.metadata_field(declared_by_name[key]).name: (
                    declared_by_name[key].coerce_scalar(value)
                )
                for key, value in common_metadata.items()
                if key in declared_by_name
            }
            _merge_projected_row_values(
                image_rows_by_number[image_number],
                projected_metadata,
                owner=f"CPA image {image_number}",
            )
        return tuple(
            row_projection.dialect.metadata_field(field_spec)
            for field_spec in source_fields
        )

    @staticmethod
    def _merge_object_rows(
        *,
        table: MeasurementTable,
        rows: tuple[Mapping[str, Any], ...],
        subject: MeasurementSubject,
        row_projection: CPATableRowProjection,
        target: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        image_id_field = row_projection.image_id_field()
        object_id_field = row_projection.object_id_field(subject)
        for row in rows:
            image_number = row_projection.required_int(
                row,
                image_id_field,
                table.name,
            )
            object_number = row_projection.required_int(
                row,
                object_id_field,
                table.name,
            )
            projected_row = target.setdefault(
                (image_number, object_number),
                {
                    image_id_field.name: image_number,
                    object_id_field.name: object_number,
                },
            )
            _merge_projected_row_values(
                projected_row,
                row,
                owner=f"CPA object row {(image_number, object_number)!r}",
            )

    @staticmethod
    def _validate_image_channels(
        artifact_batch: RuntimeArtifactBatch,
        image_channels: Sequence[CPAImageChannelSpec],
    ) -> None:
        declared_image_names = frozenset(
            spec.name for spec in artifact_batch.specs_of_type(ImageArtifactType)
        )
        seen_aliases: set[str] = set()
        for channel in image_channels:
            if not isinstance(channel, CPAImageChannelSpec):
                raise TypeError(
                    "CellProfilerAnalystProjectionBuilder image channels must be "
                    f"CPAImageChannelSpec values, got {type(channel).__name__}."
                )
            if channel.alias not in declared_image_names:
                raise ValueError(
                    "CPA image channel references an image absent from the exact "
                    f"runtime artifact contract: {channel.alias!r}."
                )
            if channel.alias in seen_aliases:
                raise ValueError(
                    f"CPA image channel alias {channel.alias!r} is declared twice."
                )
            seen_aliases.add(channel.alias)

    def _dialect(
        self,
        settings: CellProfilerDatabaseExportSettings,
    ) -> CellProfilerDatabaseColumnDialect:
        if self.dialect is not None:
            return self.dialect
        return CellProfilerDatabaseColumnDialect(settings.table_prefix)


@dataclass(frozen=True, slots=True)
class CPAPropertiesRenderer:
    """Render CPA properties text from a projection and export request."""

    dialect: CellProfilerDatabaseColumnDialect | None = None

    def render(
        self,
        settings: CellProfilerDatabaseExportSettings,
        image_channels: Sequence[CPAImageChannelSpec],
        projection: CellProfilerAnalystProjection,
    ) -> tuple[CPAPropertiesFile, ...]:
        if not settings.wants_properties_file:
            return ()
        object_tables: tuple[CellProfilerProjectedTable | None, ...]
        if settings.object_table_mode is CellProfilerObjectTableMode.PER_OBJECT:
            object_tables = tuple(
                table
                for table in projection.object_tables
                if settings.exports_object(
                    _required_table_subject_name(
                        table,
                        MeasurementScope.OBJECT,
                    )
                )
            ) or (None,)
        else:
            object_tables = (None,)
        dialect = self._dialect(settings, projection)
        return tuple(
            self._render_for_object_table(
                settings=settings,
                image_channels=image_channels,
                projection=projection,
                dialect=dialect,
                object_table=object_table,
            )
            for object_table in object_tables
        )

    def _render_for_object_table(
        self,
        *,
        settings: CellProfilerDatabaseExportSettings,
        image_channels: Sequence[CPAImageChannelSpec],
        projection: CellProfilerAnalystProjection,
        dialect: CellProfilerDatabaseColumnDialect,
        object_table: CellProfilerProjectedTable | None,
    ) -> CPAPropertiesFile:
        properties = self._properties(
            settings=settings,
            image_channels=image_channels,
            projection=projection,
            dialect=dialect,
            object_table=object_table,
        )
        object_name = (
            None
            if object_table is None
            else _required_table_subject_name(
                object_table,
                MeasurementScope.OBJECT,
            )
        )
        return CPAPropertiesFile(
            object_name=object_name,
            file_name=self._file_name(settings=settings, object_name=object_name),
            properties=properties,
            text="\n".join(f"{key} = {value}" for key, value in properties.items())
            + "\n",
        )

    def _properties(
        self,
        *,
        settings: CellProfilerDatabaseExportSettings,
        image_channels: Sequence[CPAImageChannelSpec],
        projection: CellProfilerAnalystProjection,
        dialect: CellProfilerDatabaseColumnDialect,
        object_table: CellProfilerProjectedTable | None,
    ) -> Mapping[str, str]:
        combined_objects = (
            settings.object_table_mode is not CellProfilerObjectTableMode.PER_OBJECT
        )
        object_table_name = (
            dialect.combined_object_table_name()
            if combined_objects
            else ""
            if object_table is None
            else object_table.table_name
        )
        property_fields = self._property_fields_for_table(
            settings,
            image_channels,
            projection,
            object_table,
            dialect,
        )
        rendered_columns = {
            property_name: ",".join(
                dialect.render_name(field_spec.name) for field_spec in fields
            )
            for property_name, fields in property_fields.items()
        }
        properties = {
            CPAPropertyName.DATABASE_TYPE.value: "sqlite",
            CPAPropertyName.SQLITE_FILE.value: settings.sqlite_file,
            CPAPropertyName.IMAGE_TABLE.value: projection.image_table.table_name,
            CPAPropertyName.OBJECT_TABLE.value: object_table_name,
            CPAPropertyName.IMAGE_ID.value: rendered_columns[CPAPropertyName.IMAGE_ID],
            CPAPropertyName.OBJECT_ID.value: rendered_columns[
                CPAPropertyName.OBJECT_ID
            ],
            CPAPropertyName.PLATE_ID.value: rendered_columns[CPAPropertyName.PLATE_ID],
            CPAPropertyName.WELL_ID.value: rendered_columns[CPAPropertyName.WELL_ID],
            CPAPropertyName.SERIES_ID.value: rendered_columns[
                CPAPropertyName.SERIES_ID
            ],
            CPAPropertyName.GROUP_ID.value: rendered_columns[CPAPropertyName.GROUP_ID],
            CPAPropertyName.TIMEPOINT_ID.value: rendered_columns[
                CPAPropertyName.TIMEPOINT_ID
            ],
            CPAPropertyName.CELL_X_LOCATION.value: rendered_columns[
                CPAPropertyName.CELL_X_LOCATION
            ],
            CPAPropertyName.CELL_Y_LOCATION.value: rendered_columns[
                CPAPropertyName.CELL_Y_LOCATION
            ],
            CPAPropertyName.CELL_Z_LOCATION.value: rendered_columns[
                CPAPropertyName.CELL_Z_LOCATION
            ],
            CPAPropertyName.IMAGE_PATH_COLUMNS.value: rendered_columns[
                CPAPropertyName.IMAGE_PATH_COLUMNS
            ],
            CPAPropertyName.IMAGE_FILE_COLUMNS.value: rendered_columns[
                CPAPropertyName.IMAGE_FILE_COLUMNS
            ],
            CPAPropertyName.IMAGE_NAMES.value: ",".join(
                channel.image_name for channel in image_channels
            ),
            CPAPropertyName.IMAGE_CHANNEL_COLORS.value: ",".join(
                channel.channel_color for channel in image_channels
            ),
            CPAPropertyName.CHANNELS_PER_IMAGE.value: ",".join(
                str(channel.channels_per_image) for channel in image_channels
            ),
            CPAPropertyName.IMAGE_THUMBNAIL_COLUMNS.value: rendered_columns[
                CPAPropertyName.IMAGE_THUMBNAIL_COLUMNS
            ],
            CPAPropertyName.IMAGE_CHANNEL_BLEND_MODES.value: "",
            CPAPropertyName.IMAGE_URL_PREPEND.value: settings.image_url_prepend,
            CPAPropertyName.OBJECT_NAME.value: "cell, cells,",
            CPAPropertyName.PLATE_TYPE.value: settings.plate_type or "None",
            CPAPropertyName.CLASSIFIER_IGNORE_COLUMNS.value: (
                "table_number_key_column, image_number_key_column, "
                "object_number_key_column"
            ),
            CPAPropertyName.IMAGE_TILE_SIZE.value: "50",
            CPAPropertyName.IMAGE_SIZE.value: "",
            CPAPropertyName.CLASSIFICATION_TYPE.value: (
                "image" if settings.classification_type == "image" else ""
            ),
            CPAPropertyName.TRAINING_SET.value: "",
            CPAPropertyName.AREA_SCORING_COLUMN.value: "",
            CPAPropertyName.CLASS_TABLE.value: (
                f"{settings.table_prefix}{settings.phenotype_class_table}"
                if settings.phenotype_class_table
                else ""
            ),
            CPAPropertyName.CHECK_TABLES.value: "no",
            CPAPropertyName.FORCE_BIOFORMATS.value: "no",
            CPAPropertyName.USE_LEGACY_FETCHER.value: "no",
            CPAPropertyName.PROCESS_3D.value: "False",
        }
        properties.update(
            {
                f"group_SQL_{group_name}": (
                    f"SELECT {columns} FROM {projection.image_table.table_name}"
                )
                for group_name, columns in settings.group_fields
            }
        )
        return properties

    @classmethod
    def property_fields(
        cls,
        settings: CellProfilerDatabaseExportSettings,
        image_channels: Sequence[CPAImageChannelSpec],
        projection: CellProfilerAnalystProjection,
    ) -> tuple[FieldSpec, ...]:
        """Return actual projected fields referenced by generated properties."""

        object_tables: tuple[CellProfilerProjectedTable | None, ...]
        if settings.object_table_mode is CellProfilerObjectTableMode.PER_OBJECT:
            object_tables = tuple(
                table
                for table in projection.object_tables
                if settings.exports_object(
                    _required_table_subject_name(
                        table,
                        MeasurementScope.OBJECT,
                    )
                )
            ) or (None,)
        else:
            object_tables = (None,)
        dialect = CellProfilerDatabaseColumnDialect(settings.table_prefix)
        return tuple(
            dict.fromkeys(
                field_spec
                for object_table in object_tables
                for fields in cls._property_fields_for_table(
                    settings,
                    image_channels,
                    projection,
                    object_table,
                    dialect,
                ).values()
                for field_spec in fields
            )
        )

    @classmethod
    def _property_fields_for_table(
        cls,
        settings: CellProfilerDatabaseExportSettings,
        image_channels: Sequence[CPAImageChannelSpec],
        projection: CellProfilerAnalystProjection,
        object_table: CellProfilerProjectedTable | None,
        dialect: CellProfilerDatabaseColumnDialect,
    ) -> Mapping[CPAPropertyName, tuple[FieldSpec, ...]]:
        combined_objects = (
            settings.object_table_mode is not CellProfilerObjectTableMode.PER_OBJECT
        )
        selected_object_tables = tuple(
            table
            for table in projection.object_tables
            if settings.exports_object(
                _required_table_subject_name(table, MeasurementScope.OBJECT)
            )
        )
        projected_object_table = object_table
        if combined_objects:
            projected_object_table = CPASQLiteRenderer._combined_object_table(
                selected_object_tables,
                dialect.combined_object_table_name(),
                dialect,
            )
        location_subject = (
            MeasurementSubject(MeasurementScope.OBJECT, settings.location_object)
            if combined_objects and settings.location_object is not None
            else (
                None
                if object_table is None
                else _required_table_subject(
                    object_table,
                    MeasurementScope.OBJECT,
                )
            )
        )
        object_id_name = (
            dialect.object_id_field().name
            if combined_objects
            else (
                None
                if object_table is None
                else dialect.object_id_field(
                    _required_table_subject(
                        object_table,
                        MeasurementScope.OBJECT,
                    )
                ).name
            )
        )
        location_axes = (
            (CPAPropertyName.CELL_X_LOCATION, "X"),
            (CPAPropertyName.CELL_Y_LOCATION, "Y"),
            (CPAPropertyName.CELL_Z_LOCATION, "Z"),
        )
        locations: dict[CPAPropertyName, tuple[FieldSpec, ...]] = {
            property_name: () for property_name, _axis_name in location_axes
        }
        if location_subject is not None:
            for property_name, axis_name in location_axes:
                location_field = dialect.object_location_field(
                    location_subject,
                    axis_name,
                )
                locations[property_name] = (location_field,)
        image_table = projection.image_table
        return {
            CPAPropertyName.IMAGE_ID: cls._matching_fields(
                image_table,
                dialect.image_id_field().name,
            ),
            CPAPropertyName.OBJECT_ID: cls._matching_fields(
                projected_object_table,
                object_id_name,
            ),
            CPAPropertyName.PLATE_ID: cls._matching_fields(
                image_table,
                dialect.metadata_field(FieldSpec(settings.plate_metadata)).name,
            ),
            CPAPropertyName.WELL_ID: cls._matching_fields(
                image_table,
                dialect.metadata_field(FieldSpec(settings.well_metadata)).name,
            ),
            CPAPropertyName.SERIES_ID: cls._matching_fields(
                image_table,
                dialect.group_field("Number").name,
            ),
            CPAPropertyName.GROUP_ID: cls._matching_fields(
                image_table,
                dialect.group_field("Number").name,
            ),
            CPAPropertyName.TIMEPOINT_ID: cls._matching_fields(
                image_table,
                dialect.group_field("Index").name,
            ),
            **locations,
            CPAPropertyName.IMAGE_PATH_COLUMNS: tuple(
                field_spec
                for channel in image_channels
                for field_spec in cls._matching_fields(
                    image_table,
                    dialect.source_image_path_field(channel.alias).name,
                )
            ),
            CPAPropertyName.IMAGE_FILE_COLUMNS: tuple(
                field_spec
                for channel in image_channels
                for field_spec in cls._matching_fields(
                    image_table,
                    dialect.source_image_file_field(channel.alias).name,
                )
            ),
            CPAPropertyName.IMAGE_THUMBNAIL_COLUMNS: tuple(
                field_spec
                for image_name in settings.thumbnail_image_names
                for field_spec in cls._matching_fields(
                    image_table,
                    dialect.thumbnail_field(image_name).name,
                )
            ),
        }

    @staticmethod
    def _matching_fields(
        table: CellProfilerProjectedTable | None,
        field_name: str | None,
    ) -> tuple[FieldSpec, ...]:
        if table is None or field_name is None:
            return ()
        return tuple(
            field_spec for field_spec in table.columns if field_spec.name == field_name
        )

    def _file_name(
        self,
        *,
        settings: CellProfilerDatabaseExportSettings,
        object_name: str | None,
    ) -> str:
        sqlite_stem = Path(settings.sqlite_file).stem
        prefix = settings.table_prefix.rstrip("_")
        base_name = "_".join(part for part in (sqlite_stem, prefix) if part)
        if object_name is None:
            return f"{base_name}.properties"
        return f"{base_name}_{object_name}.properties"

    def _dialect(
        self,
        settings: CellProfilerDatabaseExportSettings,
        projection: CellProfilerAnalystProjection,
    ) -> CellProfilerDatabaseColumnDialect:
        dialect = self.dialect or CellProfilerDatabaseColumnDialect(
            settings.table_prefix
        )
        return projection.database_dialect(
            dialect,
            settings,
        )


@dataclass(frozen=True, slots=True)
class CPASQLiteRenderer:
    """Render a CPA projection as a self-contained SQLite database."""

    dialect: CellProfilerDatabaseColumnDialect | None = None

    def __post_init__(self) -> None:
        if self.dialect is not None and not isinstance(
            self.dialect,
            CellProfilerDatabaseColumnDialect,
        ):
            raise TypeError(
                "CPASQLiteRenderer.dialect must be "
                "CellProfilerDatabaseColumnDialect, "
                f"got {type(self.dialect).__name__}."
            )

    def render(
        self,
        projection: CellProfilerAnalystProjection,
        settings: CellProfilerDatabaseExportSettings,
    ) -> bytes:
        """Return SQLite bytes for the exact projected image/object/relationship rows."""

        if not isinstance(projection, CellProfilerAnalystProjection):
            raise TypeError(
                "CPASQLiteRenderer.render requires CellProfilerAnalystProjection."
            )
        if not isinstance(settings, CellProfilerDatabaseExportSettings):
            raise TypeError(
                "CPASQLiteRenderer.render requires CellProfilerDatabaseExportSettings."
            )

        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                'CREATE TABLE "Experiment" ('
                '"experiment_id" INTEGER PRIMARY KEY AUTOINCREMENT, '
                '"name" TEXT)'
            )
            connection.execute(
                'INSERT INTO "Experiment" ("name") VALUES (?)',
                (settings.experiment_name,),
            )
            connection.execute(
                'CREATE TABLE "Experiment_Properties" ('
                f'"{CPAExperimentPropertyColumn.EXPERIMENT_ID.value}" '
                "INTEGER NOT NULL, "
                f'"{CPAExperimentPropertyColumn.OBJECT_NAME.value}" TEXT NOT NULL, '
                f'"{CPAExperimentPropertyColumn.FIELD.value}" TEXT NOT NULL, '
                f'"{CPAExperimentPropertyColumn.VALUE.value}" longtext, '
                f'PRIMARY KEY ("{CPAExperimentPropertyColumn.EXPERIMENT_ID.value}", '
                f'"{CPAExperimentPropertyColumn.OBJECT_NAME.value}", '
                f'"{CPAExperimentPropertyColumn.FIELD.value}"))'
            )
            dialect = self._dialect(settings, projection)
            self._write_experiment_properties(
                connection,
                projection,
                settings,
                dialect,
            )
            if projection.image_table.columns:
                image_id_field = self._required_field(
                    projection.image_table,
                    dialect.image_id_field().name,
                )
                self._write_table(
                    connection,
                    dialect,
                    replace(
                        projection.image_table,
                        columns=dialect.ordered_measurement_fields(
                            projection.image_table.columns,
                            (image_id_field,),
                        ),
                    ),
                    primary_key=(image_id_field,),
                )
            if projection.experiment_table.columns or projection.experiment_table.rows:
                self._write_table(
                    connection,
                    dialect,
                    projection.experiment_table,
                )
            self._write_object_projection(
                connection,
                projection,
                settings,
                dialect,
            )
            if settings.wants_relationship_tables and projection.relationship_tables:
                self._write_relationship_projection(
                    connection,
                    projection,
                    settings,
                    dialect,
                )
            connection.commit()
            return bytes(connection.serialize())
        finally:
            connection.close()

    def _write_experiment_properties(
        self,
        connection: sqlite3.Connection,
        projection: CellProfilerAnalystProjection,
        settings: CellProfilerDatabaseExportSettings,
        dialect: CellProfilerDatabaseColumnDialect,
    ) -> None:
        if settings.object_table_mode is CellProfilerObjectTableMode.PER_OBJECT:
            return
        properties_files = CPAPropertiesRenderer(dialect).render(
            settings,
            projection.image_channels,
            projection,
        )
        connection.executemany(
            'INSERT INTO "Experiment_Properties" '
            f'("{CPAExperimentPropertyColumn.EXPERIMENT_ID.value}", '
            f'"{CPAExperimentPropertyColumn.OBJECT_NAME.value}", '
            f'"{CPAExperimentPropertyColumn.FIELD.value}", '
            f'"{CPAExperimentPropertyColumn.VALUE.value}") '
            "VALUES (1, ?, ?, ?)",
            tuple(
                (
                    properties_file.object_name or "Object",
                    field,
                    value,
                )
                for properties_file in properties_files
                for field, value in properties_file.properties.items()
            ),
        )

    def _write_object_projection(
        self,
        connection: sqlite3.Connection,
        projection: CellProfilerAnalystProjection,
        settings: CellProfilerDatabaseExportSettings,
        dialect: CellProfilerDatabaseColumnDialect,
    ) -> None:
        object_tables = tuple(
            table
            for table in projection.object_tables
            if settings.exports_object(
                _required_table_subject_name(
                    table,
                    MeasurementScope.OBJECT,
                )
            )
        )
        if settings.object_table_mode is CellProfilerObjectTableMode.PER_OBJECT:
            for object_table in object_tables:
                subject = _required_table_subject(
                    object_table,
                    MeasurementScope.OBJECT,
                )
                image_id_field = self._required_field(
                    object_table,
                    dialect.image_id_field().name,
                )
                object_id_field = self._required_field(
                    object_table,
                    dialect.object_id_field(subject).name,
                )
                self._write_table(
                    connection,
                    dialect,
                    replace(
                        object_table,
                        columns=dialect.ordered_measurement_fields(
                            object_table.columns,
                            (image_id_field, object_id_field),
                        ),
                    ),
                    primary_key=(image_id_field, object_id_field),
                )
            return

        combined_table_name = dialect.combined_object_table_name()
        if settings.object_table_mode is CellProfilerObjectTableMode.COMBINED:
            combined_table = self._combined_object_table(
                object_tables,
                combined_table_name,
                dialect,
            )
            image_id_field = self._required_field(
                combined_table,
                dialect.image_id_field().name,
            )
            object_id_field = self._required_field(
                combined_table,
                dialect.object_id_field().name,
            )
            self._write_table(
                connection,
                dialect,
                combined_table,
                primary_key=(image_id_field, object_id_field),
            )
            return

        source_tables: list[CellProfilerProjectedTable] = []
        for object_table in object_tables:
            subject = _required_table_subject(
                object_table,
                MeasurementScope.OBJECT,
            )
            image_id_field = self._required_field(
                object_table,
                dialect.image_id_field().name,
            )
            object_id_field = self._required_field(
                object_table,
                dialect.object_id_field(subject).name,
            )
            ordered_table = replace(
                object_table,
                columns=dialect.ordered_measurement_fields(
                    object_table.columns,
                    (image_id_field, object_id_field),
                ),
            )
            self._write_table(
                connection,
                dialect,
                ordered_table,
                primary_key=(image_id_field, object_id_field),
            )
            source_tables.append(ordered_table)
        self._write_combined_view(
            connection,
            dialect,
            combined_table_name,
            source_tables,
        )

    @classmethod
    def _combined_object_table(
        cls,
        object_tables: Sequence[CellProfilerProjectedTable],
        table_name: str,
        dialect: CellProfilerDatabaseColumnDialect,
    ) -> CellProfilerProjectedTable:
        image_id = dialect.image_id_field()
        object_id = dialect.object_id_field()
        combined_columns: tuple[FieldSpec, ...] = (image_id, object_id)
        combined: dict[
            tuple[int, int],
            dict[str, Any],
        ] = {}
        for object_table in sorted(
            object_tables,
            key=lambda table: _required_table_subject_name(
                table,
                MeasurementScope.OBJECT,
            ),
        ):
            subject = _required_table_subject(
                object_table,
                MeasurementScope.OBJECT,
            )
            source_image_id = cls._required_field(
                object_table,
                dialect.image_id_field().name,
            )
            source_object_id = cls._required_field(
                object_table,
                dialect.object_id_field(subject).name,
            )
            nullable_columns = tuple(
                replace(field_spec, required=False)
                for field_spec in dialect.ordered_measurement_fields(
                    object_table.columns,
                    (source_image_id, source_object_id),
                )
                if field_spec != source_image_id
            )
            combined_columns = FieldSpec.merge_exact(
                (combined_columns, nullable_columns),
                context=f"CPA table {table_name!r} fields",
            )
            nullable_names = frozenset(
                field_spec.name for field_spec in nullable_columns
            )
            for row in object_table.rows:
                key = (
                    int(row[source_image_id.name]),
                    int(row[source_object_id.name]),
                )
                additions = {
                    field_name: value
                    for field_name, value in row.items()
                    if field_name in nullable_names
                }
                projected_row = combined.setdefault(
                    key,
                    {image_id.name: key[0], object_id.name: key[1]},
                )
                _merge_projected_row_values(
                    projected_row,
                    additions,
                    owner=f"CPA combined object row {key!r}",
                )
        return CellProfilerProjectedTable(
            table_name=table_name,
            rows=tuple(combined.values()),
            columns=combined_columns,
        )

    def _write_table(
        self,
        connection: sqlite3.Connection,
        dialect: CellProfilerDatabaseColumnDialect,
        table: CellProfilerProjectedTable,
        *,
        primary_key: Sequence[FieldSpec] = (),
    ) -> None:
        columns = dialect.ordered_fields(
            table.columns,
            tuple(primary_key),
        )
        if not columns:
            return
        primary_key_columns = tuple(primary_key)
        missing_primary_key = tuple(
            column for column in primary_key_columns if column not in columns
        )
        if missing_primary_key:
            raise ValueError(
                f"CPA table '{table.table_name}' primary key columns are undeclared: "
                f"{missing_primary_key!r}."
            )
        rendered_columns = tuple(
            (field_spec, dialect.render_name(field_spec.name)) for field_spec in columns
        )
        rendered_by_column = dict(rendered_columns)
        column_definitions: list[str] = []
        for field_spec, column_name in rendered_columns:
            sql_type = (
                self._declared_sqlite_type(field_spec.dtype)
                if field_spec.dtype is not None
                else self._sqlite_type(table.rows, field_spec.name)
            )
            column_definitions.append(
                f"{self._quote_identifier(column_name)} {sql_type}"
            )
        if primary_key_columns:
            column_definitions.append(
                "PRIMARY KEY ("
                + ", ".join(
                    self._quote_identifier(rendered_by_column[column])
                    for column in primary_key_columns
                )
                + ")"
            )
        connection.execute(
            f"CREATE TABLE {self._quote_identifier(table.table_name)} "
            f"({', '.join(column_definitions)})"
        )
        placeholders = ", ".join("?" for _field in columns)
        insert_sql = (
            f"INSERT INTO {self._quote_identifier(table.table_name)} "
            f"({', '.join(self._quote_identifier(name) for _field, name in rendered_columns)}) "
            f"VALUES ({placeholders})"
        )
        connection.executemany(
            insert_sql,
            tuple(
                tuple(
                    self._sqlite_value(row.get(field_spec.name))
                    for field_spec in columns
                )
                for row in table.rows
            ),
        )

    @staticmethod
    def _required_field(
        table: CellProfilerProjectedTable,
        field_name: str,
    ) -> FieldSpec:
        matches = tuple(
            field_spec for field_spec in table.columns if field_spec.name == field_name
        )
        if len(matches) != 1:
            raise ValueError(
                f"Projected table {table.table_name!r} requires exactly one "
                f"field {field_name!r}, got {len(matches)}."
            )
        return matches[0]

    @classmethod
    def _write_combined_view(
        cls,
        connection: sqlite3.Connection,
        dialect: CellProfilerDatabaseColumnDialect,
        view_name: str,
        source_tables: Sequence[CellProfilerProjectedTable],
    ) -> None:
        if not source_tables:
            return
        first_table = source_tables[0]
        image_id = dialect.render_name(dialect.image_id_field().name)
        object_id = dialect.render_name(dialect.object_id_field().name)
        first_subject = _required_table_subject(
            first_table,
            MeasurementScope.OBJECT,
        )
        first_qualified_object_id = dialect.render_name(
            cls._required_field(
                first_table,
                dialect.object_id_field(first_subject).name,
            ).name
        )
        selected_columns = [
            f"{cls._quote_identifier(first_table.table_name)}.{cls._quote_identifier(image_id)}"
            f" AS {cls._quote_identifier(image_id)}",
            f"{cls._quote_identifier(first_table.table_name)}."
            f"{cls._quote_identifier(first_qualified_object_id)}"
            f" AS {cls._quote_identifier(object_id)}",
            *(
                f"{cls._quote_identifier(table.table_name)}."
                f"{cls._quote_identifier(dialect.render_name(field_spec.name))}"
                for table in source_tables
                for field_spec in table.columns
                if field_spec.name != dialect.image_id_field().name
            ),
        ]
        joins = []
        for table in source_tables[1:]:
            subject = _required_table_subject(table, MeasurementScope.OBJECT)
            qualified_object_id = dialect.render_name(
                cls._required_field(
                    table,
                    dialect.object_id_field(subject).name,
                ).name
            )
            joins.append(
                f"INNER JOIN {cls._quote_identifier(table.table_name)} ON "
                f"{cls._quote_identifier(first_table.table_name)}."
                f"{cls._quote_identifier(image_id)} = "
                f"{cls._quote_identifier(table.table_name)}."
                f"{cls._quote_identifier(image_id)} AND "
                f"{cls._quote_identifier(first_table.table_name)}."
                f"{cls._quote_identifier(first_qualified_object_id)} = "
                f"{cls._quote_identifier(table.table_name)}."
                f"{cls._quote_identifier(qualified_object_id)}"
            )
        connection.execute(
            f"CREATE VIEW {cls._quote_identifier(view_name)} AS "
            f"SELECT {', '.join(selected_columns)} "
            f"FROM {cls._quote_identifier(first_table.table_name)} " + " ".join(joins)
        )

    def _write_relationship_projection(
        self,
        connection: sqlite3.Connection,
        projection: CellProfilerAnalystProjection,
        settings: CellProfilerDatabaseExportSettings,
        dialect: CellProfilerDatabaseColumnDialect,
    ) -> None:
        type_table = dialect.relationship_table_name(
            CellProfilerRelationshipProjectionName.TYPES.value
        )
        relationship_table = dialect.relationship_table_name(
            CellProfilerRelationshipProjectionName.ROWS.value
        )
        relationship_view = dialect.relationship_table_name(
            CellProfilerRelationshipProjectionName.VIEW.value
        )
        type_unique = dialect.relationship_table_name("RelationshipTypesUnique")
        relationship_unique = dialect.relationship_table_name("RelationshipUnique")
        foreign_key = dialect.relationship_table_name("RRTypeIdFK")
        first_index = dialect.relationship_table_name("IRelationships1")
        second_index = dialect.relationship_table_name("IRelationships2")

        connection.execute(
            f"CREATE TABLE {self._quote_identifier(type_table)} ("
            '"relationship_type_id" INTEGER PRIMARY KEY, '
            '"module_number" INTEGER, '
            '"relationship" varchar(255), '
            '"object_name1" varchar(255), '
            '"object_name2" varchar(255), '
            f"CONSTRAINT {self._quote_identifier(type_unique)} UNIQUE ("
            '"relationship_type_id", "module_number", "relationship", '
            '"object_name1", "object_name2"))'
        )
        connection.execute(
            f"CREATE TABLE {self._quote_identifier(relationship_table)} ("
            '"relationship_type_id" INTEGER, '
            '"image_number1" INTEGER, '
            '"object_number1" INTEGER, '
            '"image_number2" INTEGER, '
            '"object_number2" INTEGER, '
            f"CONSTRAINT {self._quote_identifier(foreign_key)} FOREIGN KEY ("
            '"relationship_type_id") REFERENCES '
            f'{self._quote_identifier(type_table)} ("relationship_type_id"), '
            f"CONSTRAINT {self._quote_identifier(relationship_unique)} UNIQUE ("
            '"relationship_type_id", "image_number1", "object_number1", '
            '"image_number2", "object_number2"))'
        )

        for relationship_type_id, relationship in enumerate(
            projection.relationship_tables,
            start=1,
        ):
            connection.execute(
                f"INSERT INTO {self._quote_identifier(type_table)} ("
                '"relationship_type_id", "module_number", "relationship", '
                '"object_name1", "object_name2") VALUES (?, ?, ?, ?, ?)',
                (
                    relationship_type_id,
                    relationship.module_number,
                    relationship.relationship,
                    relationship.object_name1,
                    relationship.object_name2,
                ),
            )
            connection.executemany(
                f"INSERT OR IGNORE INTO {self._quote_identifier(relationship_table)} ("
                '"relationship_type_id", "image_number1", "object_number1", '
                '"image_number2", "object_number2") VALUES (?, ?, ?, ?, ?)',
                tuple(
                    (
                        relationship_type_id,
                        self._required_relationship_int(row, "image_number1"),
                        self._required_relationship_int(row, "object_number1"),
                        self._required_relationship_int(row, "image_number2"),
                        self._required_relationship_int(row, "object_number2"),
                    )
                    for row in relationship.rows
                ),
            )

        connection.execute(
            f"CREATE INDEX {self._quote_identifier(first_index)} ON "
            f"{self._quote_identifier(relationship_table)} ("
            '"image_number1", "object_number1", "relationship_type_id")'
        )
        connection.execute(
            f"CREATE INDEX {self._quote_identifier(second_index)} ON "
            f"{self._quote_identifier(relationship_table)} ("
            '"image_number2", "object_number2", "relationship_type_id")'
        )
        connection.execute(
            f"CREATE VIEW {self._quote_identifier(relationship_view)} AS SELECT "
            'T."module_number", T."relationship", T."object_name1", '
            'T."object_name2", R."image_number1", R."object_number1", '
            'R."image_number2", R."object_number2" FROM '
            f"{self._quote_identifier(type_table)} T JOIN "
            f"{self._quote_identifier(relationship_table)} R ON "
            'T."relationship_type_id" = R."relationship_type_id"'
        )

    @staticmethod
    def _required_relationship_int(row: Mapping[str, Any], field_name: str) -> int:
        if field_name not in row:
            raise ValueError(f"CPA relationship row requires field '{field_name}'.")
        value = row[field_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"CPA relationship field '{field_name}' must be an integer, "
                f"got {value!r}."
            )
        return int(value)

    @staticmethod
    def _declared_sqlite_type(dtype: object) -> str:
        if dtype is bool or dtype in {"bool", "boolean"}:
            return "INTEGER"
        if dtype is int or dtype in {
            "int",
            "integer",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        }:
            return "INTEGER"
        if dtype is float or dtype in {
            "float",
            "float16",
            "float32",
            "float64",
        }:
            return "float"
        if dtype == "real":
            return "REAL"
        if dtype == "double":
            return "double"
        if dtype is str or dtype in {"str", "string", "text", "varchar"}:
            return "TEXT"
        if dtype is bytes or dtype == "bytes":
            return "longblob"
        if dtype == "blob":
            return "BLOB"
        if dtype == "longblob":
            return "longblob"
        if isinstance(dtype, str) and dtype.strip():
            return dtype
        raise TypeError(f"Unsupported declared CPA SQLite dtype {dtype!r}.")

    @classmethod
    def _sqlite_type(
        cls,
        rows: Sequence[Mapping[str, Any]],
        field_name: str,
    ) -> str:
        values = tuple(
            cls._sqlite_value(row[field_name])
            for row in rows
            if field_name in row and row[field_name] is not None
        )
        if not values:
            return "TEXT"
        if all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            return "INTEGER"
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            return "REAL"
        if all(isinstance(value, bytes) for value in values):
            return "BLOB"
        return "TEXT"

    @staticmethod
    def _sqlite_value(value: Any) -> int | float | str | bytes | None:
        if isinstance(value, bool):
            return int(value)
        if value is None or isinstance(value, (str, bytes)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value)
        raise TypeError(
            f"CPA SQLite rows require scalar values, got {type(value).__name__}."
        )

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("SQLite identifiers must be non-empty strings.")
        return '"' + identifier.replace('"', '""') + '"'

    def _dialect(
        self,
        settings: CellProfilerDatabaseExportSettings,
        projection: CellProfilerAnalystProjection,
    ) -> CellProfilerDatabaseColumnDialect:
        dialect = self.dialect or CellProfilerDatabaseColumnDialect(
            settings.table_prefix
        )
        return projection.database_dialect(
            dialect,
            settings,
        )
