"""Measurement row projection and materialization for CellProfiler runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from inspect import unwrap
import logging
import time
from types import MappingProxyType
from typing import ClassVar, TYPE_CHECKING, get_args, get_origin

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute

from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MEASUREMENT_SPARSE_CELL,
    MeasurementProjectedColumnarRows,
    MeasurementRowsAxisProjection,
    MeasurementSparseColumnarRows,
    MeasurementSliceIndexImageNumberProjection,
    MeasurementSourceImageNumberProjection,
    ProjectedMeasurementRows,
    measurement_row_object_name,
)
from openhcs.core.pipeline.function_contracts import special_output_specs_from_callable
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementRowAxisState,
    ParentChildRelationshipPayload,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    ImagePayloadMetadataInput,
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
    ImagePayloadMetadataCompositionRequest,
    MeasurementTable,
    ObjectLabelValue,
    ObjectRelationship,
    SpatialGrid,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import _callable_type_hints
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
    MeasurementObjectName,
    MeasurementRowsInput,
)
from openhcs.interop.cellprofiler.runtime.source_candidates import (
    CellProfilerImageNumberResolver,
)
from openhcs.processing.materialization import tabular_field_names_from_materialization


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
        CellProfilerObjectMeasurementRowPolicy,
    )

class CellProfilerMeasurementFieldSchema:
    """Authoritative field inference for CellProfiler measurement records."""

    @classmethod
    def for_record(
        cls,
        spec: ArtifactSpec,
        rows: MeasurementRowsInput,
        func: CellProfilerFunction,
    ) -> tuple[FieldSpec, ...]:
        fields = cls.from_materialization(spec)
        if fields:
            return fields
        fields = cls.from_callable_materialization(func)
        if fields:
            return fields
        fields = cls.from_rows(rows)
        if fields:
            return fields
        return cls.from_callable(func)

    @staticmethod
    def from_materialization(spec: ArtifactSpec) -> tuple[FieldSpec, ...]:
        field_names = tabular_field_names_from_materialization(spec.materialization)
        return tuple(FieldSpec(name) for name in field_names)

    @staticmethod
    def from_callable_materialization(
        func: CellProfilerFunction,
    ) -> tuple[FieldSpec, ...]:
        raw_outputs = special_output_specs_from_callable(unwrap(func))
        field_sets = tuple(
            field_names
            for output_spec in raw_outputs
            if (
                isinstance(output_spec, tuple)
                and len(output_spec) == 2
                and (
                    field_names := tabular_field_names_from_materialization(
                        output_spec[1]
                    )
                )
            )
        )
        if len(field_sets) != 1:
            return ()
        return tuple(FieldSpec(name) for name in field_sets[0])

    @staticmethod
    def rows_have_inferable_fields(rows: MeasurementRowsInput) -> bool:
        if isinstance(rows, ColumnarRows):
            return True
        if not rows:
            return False
        row = rows[0]
        return bool(is_dataclass(row) or (isinstance(row, Mapping) and row))

    @staticmethod
    def rows_declare_object_name(rows: MeasurementRowsInput) -> bool:
        """Return whether rows carry explicit object ownership."""
        if isinstance(rows, ColumnarRows):
            return MeasurementRowAxisField.OBJECT_NAME.value in rows.columns
        return any(
            measurement_row_object_name(measurement_row_mapping(row)) is not None
            for row in rows
        )

    @classmethod
    def from_rows(cls, rows: MeasurementRowsInput) -> tuple[FieldSpec, ...]:
        """Infer table fields from concrete row carriers."""
        if isinstance(rows, ColumnarRows):
            return tuple(FieldSpec(str(field_name)) for field_name in rows.columns)
        if not cls.rows_have_inferable_fields(rows):
            return ()
        return cls.from_row_mappings(
            tuple(measurement_row_mapping(row) for row in rows)
        )

    @staticmethod
    def from_row_mappings(
        rows: Sequence[CellProfilerKwargs],
    ) -> tuple[FieldSpec, ...]:
        field_names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for field_name in row:
                if field_name in seen:
                    continue
                seen.add(field_name)
                field_names.append(str(field_name))
        return tuple(FieldSpec(field_name) for field_name in field_names)

    @classmethod
    def from_callable(cls, func: CellProfilerFunction) -> tuple[FieldSpec, ...]:
        return_type = _callable_type_hints(unwrap(func)).get("return")
        row_type = cls.row_type_from_annotation(return_type)
        if row_type is None:
            return ()
        return tuple(FieldSpec(field.name) for field in dataclass_fields(row_type))

    @classmethod
    def row_type_from_annotation(cls, annotation: CellProfilerRuntimeValue) -> type[CellProfilerRuntimeValue] | None:
        if isinstance(annotation, type) and is_dataclass(annotation):
            return annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in (list, tuple):
            return cls.row_type_from_sequence_args(args)
        return None

    @classmethod
    def row_type_from_sequence_args(
        cls,
        args: CellProfilerRuntimeValues,
    ) -> type[CellProfilerRuntimeValue] | None:
        for arg in args:
            if arg is Ellipsis:
                continue
            row_type = cls.row_type_from_annotation(arg)
            if row_type is not None:
                return row_type
        return None

@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementOutputProjection:
    """Project raw CellProfiler feature fields to canonical runtime table names."""

    fields: tuple[FieldSpec, ...]
    rows: MeasurementRowsInput

    def apply(self) -> tuple[tuple[FieldSpec, ...], MeasurementRowsInput]:
        fields = self.projected_fields()
        if isinstance(self.rows, ColumnarRows):
            return fields, self.project_columnar_rows(self.rows)
        return fields, self.project_rows(self.rows)

    def projected_fields(self) -> tuple[FieldSpec, ...]:
        projected: list[FieldSpec] = []
        seen: set[str] = set()
        for field in self.fields:
            self.add_projected_field(
                projected,
                seen,
                FieldSpec(
                    self.export_field_name(field.name),
                    dtype=field.dtype,
                    required=field.required,
                ),
            )
        return tuple(projected)

    @staticmethod
    def add_projected_field(
        projected: list[FieldSpec],
        seen: set[str],
        field: FieldSpec,
    ) -> None:
        if field.name in seen:
            return
        seen.add(field.name)
        projected.append(field)

    @classmethod
    def project_columnar_rows(cls, rows: ColumnarRows) -> ColumnarRows:
        columns = CellProfilerProjectedColumnarRowColumns(rows.columns)
        if isinstance(rows, MeasurementSparseColumnarRows):
            return MeasurementSparseColumnarRows(
                columns,
                missing_cell=rows.missing_cell,
                declared_object_measurement_domain_covered=(
                    rows.covers_declared_object_measurement_domain
                ),
            )
        return MeasurementProjectedColumnarRows(
            columns,
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
        )

    @classmethod
    def project_row(cls, row: CellProfilerRuntimeValue) -> "CellProfilerProjectedMeasurementRow":
        projected = {
            cls.export_field_name(name): value
            for name, value in measurement_row_mapping(row).items()
        }
        return CellProfilerProjectedMeasurementRow(MappingProxyType(projected))

    @classmethod
    def project_rows(
        cls,
        rows: CellProfilerRuntimeValueSequence,
    ) -> tuple["CellProfilerProjectedMeasurementRow", ...]:
        field_name_cache: dict[tuple[str, ...], tuple[str, ...]] = {}
        projected_rows: list[CellProfilerProjectedMeasurementRow] = []
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            field_names = tuple(str(name) for name in row_mapping)
            projected_field_names = field_name_cache.get(field_names)
            if projected_field_names is None:
                projected_field_names = tuple(
                    cls.export_field_name(name) for name in field_names
                )
                field_name_cache[field_names] = projected_field_names
            projected_rows.append(
                CellProfilerProjectedMeasurementRow(
                    MappingProxyType(
                        dict(zip(projected_field_names, row_mapping.values()))
                    )
                )
            )
        return tuple(projected_rows)

    @staticmethod
    def export_field_name(name: CellProfilerRuntimeValue) -> str:
        text = str(name)
        if text == text.lower():
            return text
        return normalize_runtime_identifier(text)


@dataclass(frozen=True, slots=True)
class CellProfilerProjectedColumnarRowColumns(Mapping[str, Sequence[object]]):
    """Lazy column-name projection for CellProfiler measurement tables."""

    source_columns: Mapping[str, Sequence[object]]
    source_column_by_projected_name: Mapping[str, str] = field(init=False)
    column_names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        projected_sources: dict[str, str] = {}
        for source_name in self.source_columns:
            projected_sources[
                CellProfilerMeasurementOutputProjection.export_field_name(source_name)
            ] = source_name
        object.__setattr__(
            self,
            "source_column_by_projected_name",
            MappingProxyType(projected_sources),
        )
        object.__setattr__(self, "column_names", tuple(projected_sources))

    def __getitem__(self, projected_name: str) -> Sequence[object]:
        return self.source_columns[self.source_column_by_projected_name[projected_name]]

    def __iter__(self):
        return iter(self.column_names)

    def __len__(self) -> int:
        return len(self.column_names)

@dataclass(frozen=True, slots=True)
class ProjectedMeasurementFieldDescriptor:
    """Generated-style descriptor for one projected measurement field."""

    field_name: str

    def __get__(
        self,
        instance: "CellProfilerProjectedMeasurementRow | None",
        owner: type["CellProfilerProjectedMeasurementRow"],
    ) -> CellProfilerRuntimeValue | "ProjectedMeasurementFieldDescriptor":
        del owner
        if instance is None:
            return self
        return instance.value_for_field(self.field_name)

@dataclass(slots=True)
class CellProfilerProjectedMeasurementRow(CellProfilerKwargs):
    """Projected measurement row supporting explicit mapping access."""

    area_shape_area: ClassVar[ProjectedMeasurementFieldDescriptor] = (
        ProjectedMeasurementFieldDescriptor("area_shape_area")
    )
    values: CellProfilerKwargs

    def __getitem__(self, key: str) -> CellProfilerRuntimeValue:
        return self.values[key]

    def __iter__(self) -> CellProfilerRuntimeValue:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def value_for_field(self, name: str) -> CellProfilerRuntimeValue:
        """Return a projected field value or fail with AttributeError semantics."""
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

_MISSING_MEASUREMENT_OBJECT_NAME = object()

CellProfilerMeasurementSourcePayload = ImagePayloadMetadataInput | ObjectLabelValue


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementSourceContext:
    """Source context carried by CellProfiler measurement rows."""

    source_image_name: str | None = None
    source_image_payload: CellProfilerMeasurementSourcePayload | None = None
    source_metadata: ImagePayloadMetadata | None = None

    def with_ownership(
        self,
        *,
        source_image_name: str | None,
        source_image_payload: CellProfilerMeasurementSourcePayload | None,
    ) -> "CellProfilerMeasurementSourceContext":
        return CellProfilerMeasurementSourceContext(
            source_image_name=source_image_name,
            source_image_payload=source_image_payload,
            source_metadata=self.source_metadata,
        )

    def without_source(self) -> "CellProfilerMeasurementSourceContext":
        return CellProfilerMeasurementSourceContext(source_metadata=self.source_metadata)

    def payload_metadata(self) -> ImagePayloadMetadata:
        if self.source_metadata is not None:
            return self.source_metadata
        return image_payload_metadata(self.source_image_payload)

    def record_context_with_output_value(
        self,
        *,
        object_name: MeasurementObjectName,
        output_values: CellProfilerMeasurementOutputValues,
    ) -> "CellProfilerMeasurementSourceContext":
        if object_name is _MISSING_MEASUREMENT_OBJECT_NAME or object_name is None:
            return self
        output_value = output_values.get(str(object_name))
        if (
            output_value is None
            or not image_payload_metadata(output_value).source_image_paths
        ):
            return self
        source_image_name = self.source_image_name
        if source_image_name is None and isinstance(output_value, ObjectLabelValue):
            source_image_name = output_value.source_image_name
        return CellProfilerMeasurementSourceContext(
            source_image_name=source_image_name,
            source_image_payload=output_value,
            source_metadata=self.source_metadata,
        )


@dataclass(slots=True)
class CellProfilerMeasurementRecord:
    """Rows and semantic owner for one CellProfiler measurement output."""

    rows: MeasurementRowsInput
    object_name: MeasurementObjectName = _MISSING_MEASUREMENT_OBJECT_NAME
    source_context: CellProfilerMeasurementSourceContext = field(
        default_factory=CellProfilerMeasurementSourceContext
    )
    fields: tuple[FieldSpec, ...] = ()
    owns_source_qualified_features: bool = False
    clear_source_when_rows_declare_object_name: bool = True

    def __post_init__(self) -> None:
        if (
            self.clear_source_when_rows_declare_object_name
            and
            self.object_name is None
            and CellProfilerMeasurementFieldSchema.rows_declare_object_name(self.rows)
        ):
            self.source_context = self.source_context.without_source()
        if self.fields or not CellProfilerMeasurementFieldSchema.rows_have_inferable_fields(
            self.rows
        ):
            return
        self.fields = CellProfilerMeasurementFieldSchema.from_rows(self.rows)

    @classmethod
    def shared_source_image_name(
        cls,
        records: tuple["CellProfilerMeasurementRecord", ...],
    ) -> str | None:
        """Return a table source only when every record declares the same one."""
        if (
            not records
            or any(record.owns_source_qualified_features for record in records)
            or any(record.source_context.source_image_name is None for record in records)
        ):
            return None
        unique_names = tuple(
            dict.fromkeys(
                record.source_context.source_image_name for record in records
            )
        )
        if len(unique_names) == 1:
            return unique_names[0]
        return None

    @classmethod
    def composed_source_metadata(
        cls,
        records: tuple["CellProfilerMeasurementRecord", ...],
        *,
        mode: ImagePayloadMetadataCompositionMode = (
            ImagePayloadMetadataCompositionMode.STACK
        ),
    ) -> ImagePayloadMetadata | None:
        """Return source metadata composed in measurement-record order."""
        if not records:
            return None
        source_metadata = tuple(
            record.source_context.payload_metadata() for record in records
        )
        if not any(metadata.has_values for metadata in source_metadata):
            return None
        composed = ImagePayloadMetadataCompositionRequest(
            tuple(metadata.payload_with((0,)) for metadata in source_metadata),
            mode=mode,
        ).metadata()
        if not composed.has_values:
            return None
        return composed

    def with_ownership(
        self,
        rows: MeasurementRowsInput,
        *,
        object_name: MeasurementObjectName,
        source_image_name: str | None,
        source_image_payload: CellProfilerMeasurementSourcePayload | None,
    ) -> "CellProfilerMeasurementRecord":
        """Return a partition preserving this record's field schema."""
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=object_name,
            source_context=self.source_context.with_ownership(
                source_image_name=source_image_name,
                source_image_payload=source_image_payload,
            ),
            fields=self.fields,
            clear_source_when_rows_declare_object_name=False,
        )

    def projection_request(
        self,
        *,
        adapter: CellProfilerRuntimeAdapter,
        output_values: CellProfilerMeasurementOutputValues | None = None,
        current_image: CellProfilerMeasurementSourcePayload | None = None,
        need_row_mappings: bool = False,
    ) -> "CellProfilerMeasurementProjectionRequest":
        """Return a projection request using this record's ownership semantics."""
        return CellProfilerMeasurementProjectionRequest(
            adapter=adapter,
            rows=self.rows,
            fields=self.fields,
            source_context=self.source_context,
            source_resolver=CellProfilerMeasurementSourceResolver(
                object_source_lookup=AdapterObjectLabelSourceLookup(
                    adapter=adapter,
                    current_source_payload=current_image,
                ),
                output_values=(
                    MappingProxyType({})
                    if output_values is None
                    else output_values
                ),
            ),
            object_name=self.object_name,
            need_row_mappings=need_row_mappings,
        )

    def materialization_request(
        self,
        *,
        adapter: CellProfilerRuntimeAdapter,
        name: str,
        output_values: CellProfilerMeasurementOutputValues | None = None,
        current_image: CellProfilerMeasurementSourcePayload | None = None,
        axis_state: MeasurementRowAxisState | None = None,
    ) -> "CellProfilerMeasurementMaterializationRequest":
        """Return an adapter materialization request for this record."""
        return CellProfilerMeasurementMaterializationRequest.for_rows(
            adapter=adapter,
            name=name,
            rows=self.rows,
            fields=self.fields,
            object_name=self.object_name,
            source_context=self.source_context,
            source_resolver=CellProfilerMeasurementSourceResolver(
                object_source_lookup=AdapterObjectLabelSourceLookup(
                    adapter=adapter,
                    current_source_payload=current_image,
                ),
                output_values=(
                    MappingProxyType({})
                    if output_values is None
                    else output_values
                ),
            ),
            axis_state=(
                CellProfilerMeasurementOutputAxisState.for_rows(self.rows)
                if axis_state is None
                else axis_state
            ),
        )

@dataclass(frozen=True, slots=True)
class MeasurementRowColumnarMaterialization:
    """Columnar view for mapping rows with sparse per-row feature fields."""

    rows: CellProfilerRuntimeValueSequence

    @classmethod
    def from_rows(cls, rows: CellProfilerRuntimeValueSequence) -> "MeasurementRowColumnarMaterialization":
        return cls(tuple(rows))

    def table(self) -> tuple[MeasurementRowsInput, tuple[FieldSpec, ...]]:
        row_mappings = tuple(measurement_row_mapping(row) for row in self.rows)
        fields = CellProfilerMeasurementFieldSchema.from_row_mappings(row_mappings)
        if not row_mappings:
            return list(self.rows), fields
        field_names = tuple(field.name for field in fields)
        columns = {
            field_name: tuple(
                row[field_name] if field_name in row else MEASUREMENT_SPARSE_CELL
                for row in row_mappings
            )
            for field_name in field_names
        }
        return (
            MeasurementSparseColumnarRows(MappingProxyType(columns)),
            fields,
        )

ProjectedMeasurementRowsResult = tuple[MeasurementRowsInput, ProjectedMeasurementRows]

CellProfilerMeasurementOutputValue = (
    CellProfilerMeasurementSourcePayload
    | MeasurementTable
    | ParentChildRelationshipPayload
    | ObjectRelationship
    | SpatialGrid
)

CellProfilerMeasurementOutputValues = Mapping[str, CellProfilerMeasurementOutputValue]


class ObjectLabelSourceLookup(ABC):
    """Lookup authority for object-label source context used by measurements."""

    def source_context_for(
        self,
        source_context: CellProfilerMeasurementSourceContext,
        object_name: MeasurementObjectName,
    ) -> CellProfilerMeasurementSourceContext:
        del object_name
        return source_context

    def source_metadata_for(
        self,
        object_name: MeasurementObjectName,
    ) -> ImagePayloadMetadata:
        del object_name
        return ImagePayloadMetadata()


@dataclass(frozen=True, slots=True)
class AdapterObjectLabelSourceLookup(ObjectLabelSourceLookup):
    """Resolve object-label source context through the runtime adapter."""

    adapter: CellProfilerRuntimeAdapter
    current_source_payload: CellProfilerMeasurementSourcePayload | None = None

    def source_context_for(
        self,
        source_context: CellProfilerMeasurementSourceContext,
        object_name: MeasurementObjectName,
    ) -> CellProfilerMeasurementSourceContext:
        object_labels = self.object_labels(object_name)
        if not image_payload_metadata(object_labels).source_image_paths:
            return source_context
        return CellProfilerMeasurementSourceContext(
            source_image_name=object_labels.source_image_name,
            source_image_payload=object_labels,
        )

    def source_metadata_for(
        self,
        object_name: MeasurementObjectName,
    ) -> ImagePayloadMetadata:
        return image_payload_metadata(self.object_labels(object_name))

    def object_labels(self, object_name: MeasurementObjectName) -> ObjectLabelValue:
        return self.adapter.get_objects(
            str(object_name),
            current_image=self.current_source_payload,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementSourceResolver:
    """Resolve measurement source provenance from direct, output, and object context."""

    object_source_lookup: ObjectLabelSourceLookup = field(
        default_factory=ObjectLabelSourceLookup
    )
    output_values: CellProfilerMeasurementOutputValues = field(
        default_factory=lambda: MappingProxyType({})
    )

    @staticmethod
    def has_object_owner(object_name: MeasurementObjectName) -> bool:
        return (
            object_name is not _MISSING_MEASUREMENT_OBJECT_NAME
            and object_name is not None
        )

    def record_source_context(
        self,
        source_context: CellProfilerMeasurementSourceContext,
        *,
        object_name: MeasurementObjectName,
    ) -> CellProfilerMeasurementSourceContext:
        return source_context.record_context_with_output_value(
            object_name=object_name,
            output_values=self.output_values,
        )

    def projection_source_context(
        self,
        source_context: CellProfilerMeasurementSourceContext,
        *,
        object_name: MeasurementObjectName,
    ) -> CellProfilerMeasurementSourceContext:
        resolved_context = self.record_source_context(
            source_context,
            object_name=object_name,
        )
        if self.has_object_owner(object_name):
            return self.object_source_lookup.source_context_for(
                resolved_context,
                object_name,
            )
        return resolved_context

    def object_label_source_metadata(
        self,
        *,
        object_name: MeasurementObjectName,
    ) -> ImagePayloadMetadata:
        if not self.has_object_owner(object_name):
            return ImagePayloadMetadata()
        return self.object_source_lookup.source_metadata_for(object_name)


@dataclass(slots=True, kw_only=True)
class CellProfilerMeasurementProjectionRequest(CellProfilerMeasurementRecord):
    """Request for projecting measurement rows onto CellProfiler image numbers."""

    adapter: CellProfilerRuntimeAdapter
    source_resolver: CellProfilerMeasurementSourceResolver = field(
        default_factory=CellProfilerMeasurementSourceResolver
    )
    need_row_mappings: bool = False

    def __post_init__(self) -> None:
        """Projection requests must preserve already-resolved row ownership."""

    def project_rows(self) -> ProjectedMeasurementRowsResult:
        """Return rows projected into CellProfiler ImageNumber space."""
        phase_started_at = time.perf_counter()
        start, source_paths = self.axis_image_number_start()
        RuntimeProfileLogger.log(
            logger,
            "measurement_project_axis_start",
            time.perf_counter() - phase_started_at,
            source_paths=len(source_paths),
        )
        phase_started_at = time.perf_counter()
        image_numbers = MeasurementSliceIndexImageNumberProjection(
            start=start,
            image_numbers_by_slice=self.image_numbers_by_slice(
                source_paths,
                start=start,
            ),
        )
        RuntimeProfileLogger.log(
            logger,
            "measurement_project_image_numbers",
            time.perf_counter() - phase_started_at,
            source_paths=len(source_paths),
        )
        phase_started_at = time.perf_counter()
        projection = MeasurementRowsAxisProjection.from_rows(self.rows)
        RuntimeProfileLogger.log(
            logger,
            "measurement_project_projection",
            time.perf_counter() - phase_started_at,
            rows=projection.row_count,
        )
        source_image_numbers = self.source_image_number_projection(
            projection,
            source_paths,
            start=start,
        )
        phase_started_at = time.perf_counter()
        projected_rows = projection.apply(
            image_numbers,
            source_image_numbers=source_image_numbers,
        )
        RuntimeProfileLogger.log(
            logger,
            "measurement_project_apply",
            time.perf_counter() - phase_started_at,
            rows=projection.row_count,
        )
        rows = self.rows if projected_rows is None else projected_rows
        row_mappings: ProjectedMeasurementRows = ()
        if self.need_row_mappings:
            row_mappings = rows
        return rows, row_mappings

    def source_image_number_projection(
        self,
        projection: MeasurementRowsAxisProjection,
        source_paths: tuple[str, ...],
        *,
        start: int,
    ) -> MeasurementSourceImageNumberProjection | None:
        """Return per-source ImageNumber projection for source-qualified image rows."""
        if not projection.has_source_qualified_image_rows:
            return None
        if self.object_name not in (_MISSING_MEASUREMENT_OBJECT_NAME, None):
            return None

        source_context = self.source_resolver.projection_source_context(
            self.source_context,
            object_name=self.object_name,
        )
        source_metadata = source_context.payload_metadata()
        source_image_names = source_metadata.source_image_names
        if not source_image_names:
            if len(source_paths) <= 1:
                source_image_name_field = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
                row_source_names = tuple(
                    dict.fromkeys(
                        str(row_mapping[source_image_name_field])
                        for row in self.rows
                        for row_mapping in (measurement_row_mapping(row),)
                        if source_image_name_field in row_mapping
                        and row_mapping[source_image_name_field] not in (None, "", "None")
                    )
                )
                return MeasurementSourceImageNumberProjection(
                    MappingProxyType(
                        {source_image_name: int(start) for source_image_name in row_source_names}
                    )
                )
            raise ValueError(
                "Cannot project source-qualified measurement rows because source "
                "payload metadata does not declare source_image_names."
            )

        image_numbers_by_slice = self.image_numbers_by_slice(source_paths, start=start)
        image_numbers_by_source_name: dict[str, int] = {}
        for source_index, source_image_name in enumerate(source_image_names):
            if source_index in image_numbers_by_slice:
                image_number = image_numbers_by_slice[source_index]
            elif len(source_paths) <= 1:
                image_number = start
            else:
                source_name_paths = self.adapter.cellprofiler_source_paths_for_image_name(
                    str(source_image_name)
                )
                image_number = self.source_image_number_for_paths(source_name_paths)
                if image_number is None:
                    raise ValueError(
                        "Cannot project source-qualified measurement rows because "
                        f"source plane {source_index} for {source_image_name!r} has no "
                        "CellProfiler ImageNumber. "
                        f"payload_source_paths={source_paths!r}; "
                        f"source_name_paths={source_name_paths!r}; "
                        f"source_image_names={source_image_names!r}."
                    )
            image_numbers_by_source_name[str(source_image_name)] = int(image_number)
        return MeasurementSourceImageNumberProjection(
            MappingProxyType(image_numbers_by_source_name)
        )

    def source_image_number_for_name(self, source_image_name: str) -> int | None:
        """Return the CP ImageNumber for a named source image, when resolvable."""
        source_paths = self.adapter.cellprofiler_source_paths_for_image_name(
            source_image_name
        )
        return self.source_image_number_for_paths(source_paths)

    def source_image_number_for_paths(self, source_paths: tuple[str, ...]) -> int | None:
        """Return the CP ImageNumber for source paths, when resolvable."""
        if not source_paths:
            return None
        return self.adapter.cellprofiler_image_number_for_source_paths(source_paths)

    @property
    def start(self) -> int:
        start, _source_paths = self.axis_image_number_start()
        return start

    def axis_image_number_start(self) -> tuple[int, tuple[str, ...]]:
        """Return CellProfiler ImageNumber start and source-path provenance."""
        phase_started_at = time.perf_counter()
        source_context = self.source_resolver.projection_source_context(
            self.source_context,
            object_name=self.object_name,
        )
        RuntimeProfileLogger.log(
            logger,
            "measurement_axis_source_context",
            time.perf_counter() - phase_started_at,
        )
        phase_started_at = time.perf_counter()
        source_paths = source_context.payload_metadata().source_image_paths
        RuntimeProfileLogger.log(
            logger,
            "measurement_axis_payload_metadata",
            time.perf_counter() - phase_started_at,
            source_paths=len(source_paths),
        )
        if not source_paths:
            phase_started_at = time.perf_counter()
            source_paths = self.adapter.cellprofiler_source_paths_for_image_name(
                source_context.source_image_name
            )
            RuntimeProfileLogger.log(
                logger,
                "measurement_axis_source_paths_for_image",
                time.perf_counter() - phase_started_at,
                source_paths=len(source_paths),
            )
        phase_started_at = time.perf_counter()
        start = self.adapter.cellprofiler_image_number_start_for_source_paths(
            source_paths
        )
        RuntimeProfileLogger.log(
            logger,
            "measurement_axis_image_number_start",
            time.perf_counter() - phase_started_at,
            source_paths=len(source_paths),
            start=start,
        )
        log_object_name = self.object_name
        if log_object_name is _MISSING_MEASUREMENT_OBJECT_NAME:
            log_object_name = None
        RuntimeProfileLogger.log(logger,
            "cp_measurement_axis_start",
            0.0,
            object_name=log_object_name,
            source_image_name=source_context.source_image_name,
            source_paths=source_paths,
            start=start,
        )
        return start, source_paths

    @property
    def slice_index_image_numbers(self) -> MeasurementSliceIndexImageNumberProjection:
        start, source_paths = self.axis_image_number_start()
        return MeasurementSliceIndexImageNumberProjection(
            start=start,
            image_numbers_by_slice=self.image_numbers_by_slice(
                source_paths,
                start=start,
            ),
        )

    def image_numbers_by_slice(
        self,
        source_paths: tuple[str, ...],
        *,
        start: int,
    ) -> Mapping[int, int]:
        if len(source_paths) == 1:
            return MappingProxyType({0: start})
        if not self.adapter.can_resolve_source_candidates:
            return MappingProxyType(
                {
                    slice_index: image_number
                    for slice_index, source_path in enumerate(source_paths)
                    for image_number in (
                        self.adapter.cellprofiler_image_number_for_source_paths(
                            (source_path,)
                        ),
                    )
                    if image_number is not None
                }
            )
        return CellProfilerImageNumberResolver.for_adapter(
            self.adapter
        ).image_numbers_by_source_path_index(
            source_paths
        )

    @property
    def has_measurement_object_name(self) -> bool:
        """Return whether this request has an explicit object owner."""
        return self.source_resolver.has_object_owner(self.object_name)

    def source_payload_metadata(
        self,
        rows: MeasurementRowsInput | None = None,
    ) -> ImagePayloadMetadata:
        """Return source metadata projected to the represented measurement rows."""
        represented_rows = self.rows if rows is None else rows
        projection = MeasurementRowsAxisProjection.from_rows(represented_rows)
        source_context = self.source_resolver.record_source_context(
            self.source_context,
            object_name=self.object_name,
        )
        metadata = source_context.payload_metadata()
        if (
            not metadata.source_image_paths
            and self.has_measurement_object_name
            and self.rows_need_object_source_context(projection)
        ):
            object_metadata = self.source_resolver.object_label_source_metadata(
                object_name=self.object_name,
            )
            if object_metadata.source_image_paths:
                metadata = object_metadata
        projected_metadata = self.source_payload_metadata_for_rows(
            metadata,
            projection,
        )
        if projected_metadata.source_image_paths:
            return projected_metadata
        return self.source_payload_metadata_from_image_number_rows(
            projected_metadata,
            projection,
        )

    @staticmethod
    def rows_need_object_source_context(
        projection: MeasurementRowsAxisProjection,
    ) -> bool:
        """Return whether row axes need object-label source context."""
        if projection.has_image_number and not projection.has_slice_index:
            return False
        return True

    def source_payload_metadata_from_image_number_rows(
        self,
        metadata: ImagePayloadMetadata,
        projection: MeasurementRowsAxisProjection,
    ) -> ImagePayloadMetadata:
        """Derive source metadata from a single CellProfiler ImageNumber row axis."""
        image_number = self.single_present_axis_value(
            projection,
            MeasurementRowAxisField.IMAGE_NUMBER,
        )
        if image_number is None:
            return metadata
        source_path = self.adapter.cellprofiler_source_path_for_image_number(image_number)
        if source_path is None:
            return metadata
        return ImagePayloadMetadata(source_path=source_path)

    def source_payload_metadata_for_rows(
        self,
        metadata: ImagePayloadMetadata,
        projection: MeasurementRowsAxisProjection,
    ) -> ImagePayloadMetadata:
        """Project source provenance when rows represent one source plane."""
        plane_index = self.source_plane_index_for_rows(metadata, projection)
        if plane_index is None:
            return metadata
        return metadata.for_source_plane(plane_index)

    def source_plane_index_for_rows(
        self,
        metadata: ImagePayloadMetadata,
        projection: MeasurementRowsAxisProjection,
    ) -> int | None:
        """Return the represented source-provenance plane for a single-plane row set."""
        provenance = metadata.source_provenance
        if provenance.source_plane_count <= 1:
            return None
        image_number = self.single_present_axis_value(
            projection,
            MeasurementRowAxisField.IMAGE_NUMBER,
        )
        if image_number is not None:
            return self.source_plane_index_for_image_number(metadata, image_number)
        slice_index = self.single_present_axis_value(
            projection,
            MeasurementRowAxisField.SLICE_INDEX,
        )
        if slice_index is not None and 0 <= slice_index < provenance.source_plane_count:
            return slice_index
        return None

    @staticmethod
    def single_present_axis_value(
        projection: MeasurementRowsAxisProjection,
        axis: MeasurementRowAxisField,
    ) -> int | None:
        """Return an axis value only when all present row values agree."""
        values = tuple(dict.fromkeys(projection.present_axis_values(axis.value)))
        if len(values) == 1:
            return values[0]
        return None

    def source_plane_index_for_image_number(
        self,
        metadata: ImagePayloadMetadata,
        image_number: int,
    ) -> int | None:
        """Resolve one CellProfiler ImageNumber back to a source-provenance plane."""
        source_paths = metadata.source_image_paths
        if not source_paths:
            return None
        start = self.adapter.cellprofiler_image_number_start_for_source_paths(source_paths)
        image_numbers_by_slice = self.image_numbers_by_slice(
            source_paths,
            start=start,
        )
        for slice_index, slice_image_number in image_numbers_by_slice.items():
            if int(slice_image_number) == int(image_number):
                return slice_index
        inferred = int(image_number) - start
        if 0 <= inferred < metadata.source_provenance.source_plane_count:
            return inferred
        return None

class CellProfilerMeasurementAxisStateStrategy(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered materialization policy for measurement row-axis state."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    registry_key: ClassVar[MeasurementRowAxisState | None] = None

    @classmethod
    def for_state(
        cls,
        state: MeasurementRowAxisState,
    ) -> "CellProfilerMeasurementAxisStateStrategy":
        strategy_type = cls.__registry__.get(state)
        if strategy_type is None:
            raise TypeError(f"Unsupported CellProfiler measurement axis state {state!r}.")
        return strategy_type()

    @abstractmethod
    def rows_for_materialization(
        self,
        request: CellProfilerMeasurementProjectionRequest,
    ) -> ProjectedMeasurementRowsResult:
        """Return rows with CellProfiler-compatible ImageNumber semantics."""

class CellProfilerMeasurementOutputAxisState:
    """Select projection policy from the measurement rows' declared axis."""

    @classmethod
    def for_rows(
        cls,
        rows: MeasurementRowsInput,
    ) -> MeasurementRowAxisState:
        projection = MeasurementRowsAxisProjection.from_rows(
            rows,
        )
        if projection.has_slice_index:
            return MeasurementRowAxisState.RUNTIME_AXES
        return MeasurementRowAxisState.for_image_number_presence(
            has_image_number=projection.has_image_number,
        )

class RuntimeAxisMeasurementRowsStrategy(CellProfilerMeasurementAxisStateStrategy):
    """Project runtime slice axes into CellProfiler ImageNumber space."""

    registry_key = MeasurementRowAxisState.RUNTIME_AXES

    def rows_for_materialization(
        self,
        request: CellProfilerMeasurementProjectionRequest,
    ) -> ProjectedMeasurementRowsResult:
        return request.project_rows()

class CellProfilerImageNumberMeasurementRowsStrategy(
    CellProfilerMeasurementAxisStateStrategy
):
    """Preserve rows already projected into CellProfiler ImageNumber space."""

    registry_key = MeasurementRowAxisState.IMAGE_NUMBER

    def rows_for_materialization(
        self,
        request: CellProfilerMeasurementProjectionRequest,
    ) -> ProjectedMeasurementRowsResult:
        if isinstance(request.rows, ColumnarRows):
            row_mappings: ColumnarRows | CellProfilerRuntimeValues = ()
            if request.need_row_mappings:
                row_mappings = request.rows
            return request.rows, row_mappings
        row_mappings: Sequence[CellProfilerKwargs] = ()
        if request.need_row_mappings:
            row_mappings = tuple(measurement_row_mapping(row) for row in request.rows)
        return request.rows, row_mappings

@dataclass(slots=True, kw_only=True)
class CellProfilerMeasurementMaterializationRequest(CellProfilerMeasurementProjectionRequest):
    """Request for materializing projected measurement rows into the adapter."""

    name: str
    axis_state: MeasurementRowAxisState = MeasurementRowAxisState.RUNTIME_AXES
    fields: tuple[FieldSpec, ...] = ()

    @classmethod
    def for_rows(
        cls,
        *,
        adapter: CellProfilerRuntimeAdapter,
        name: str,
        rows: MeasurementRowsInput,
        fields: tuple[FieldSpec, ...] = (),
        object_name: MeasurementObjectName = _MISSING_MEASUREMENT_OBJECT_NAME,
        source_context: CellProfilerMeasurementSourceContext = (
            CellProfilerMeasurementSourceContext()
        ),
        source_resolver: CellProfilerMeasurementSourceResolver = (
            CellProfilerMeasurementSourceResolver()
        ),
        axis_state: MeasurementRowAxisState = MeasurementRowAxisState.RUNTIME_AXES,
    ) -> "CellProfilerMeasurementMaterializationRequest":
        return cls(
            name=name,
            adapter=adapter,
            rows=rows,
            source_context=source_context,
            source_resolver=source_resolver,
            object_name=object_name,
            need_row_mappings=bool(fields),
            axis_state=axis_state,
            fields=fields,
        )

class CellProfilerMeasurementMaterializer:
    """Materialize projected CellProfiler measurement rows into the adapter."""

    @classmethod
    def record_per_object(
        cls,
        *,
        adapter: CellProfilerRuntimeAdapter,
        spec: ArtifactSpec,
        func: CellProfilerFunction,
        measurement_row_policy: "CellProfilerObjectMeasurementRowPolicy",
        object_inputs: tuple[ArtifactSpec, ...],
        image_measurement_rows: CellProfilerRuntimeValueSequence,
        combined_rows: list[CellProfilerRuntimeValue],
        columnar_rows: list[ColumnarRows],
        source_context: CellProfilerMeasurementSourceContext,
    ) -> None:
        if combined_rows:
            cls.record_table(
                adapter=adapter,
                spec=spec,
                func=func,
                rows=combined_rows,
                object_name=measurement_row_policy.table_object_owner(
                    object_inputs,
                    contains_image_measurement_rows=bool(image_measurement_rows),
                ),
                source_context=source_context,
                measurement_row_policy=measurement_row_policy,
            )
        if not combined_rows and not columnar_rows:
            cls.record_table(
                adapter=adapter,
                spec=spec,
                func=func,
                rows=(),
                object_name=measurement_row_policy.table_object_owner(object_inputs),
                source_context=source_context,
                measurement_row_policy=measurement_row_policy,
            )
        if columnar_rows:
            columnar_batches = tuple(columnar_rows)
            if len(columnar_batches) == 1:
                columnar_table: MeasurementRowsInput = columnar_batches[0]
            elif cls.columnar_batches_have_matching_columns(columnar_batches):
                columnar_table = ConcatenatedColumnarRows(columnar_batches)
            else:
                columnar_table = MeasurementSparseColumnarRows.from_columnar_batches(
                    columnar_batches,
                    declared_object_measurement_domain_covered=all(
                        rows.covers_declared_object_measurement_domain
                        for rows in columnar_batches
                    ),
                )
            cls.record_table(
                adapter=adapter,
                spec=spec,
                func=func,
                rows=columnar_table,
                object_name=measurement_row_policy.table_object_owner(object_inputs),
                source_context=source_context,
                measurement_row_policy=measurement_row_policy,
            )

    @staticmethod
    def columnar_batches_have_matching_columns(
        batches: Sequence[ColumnarRows],
    ) -> bool:
        """Return whether batches can be concatenated without sparse coalescing."""
        if not batches:
            return False
        first_columns = tuple(str(column) for column in batches[0].columns)
        return all(
            tuple(str(column) for column in batch.columns) == first_columns
            for batch in batches[1:]
        )

    @classmethod
    def record_table(
        cls,
        *,
        adapter: CellProfilerRuntimeAdapter,
        spec: ArtifactSpec,
        func: CellProfilerFunction,
        rows: MeasurementRowsInput,
        object_name: MeasurementObjectName,
        source_context: CellProfilerMeasurementSourceContext,
        axis_state: MeasurementRowAxisState | None = None,
        measurement_row_policy: "CellProfilerObjectMeasurementRowPolicy | None" = None,
    ) -> None:
        record = CellProfilerMeasurementRecord(
            rows=rows,
            fields=CellProfilerMeasurementFieldSchema.for_record(
                spec,
                rows,
                func,
            ),
            object_name=object_name,
            source_context=source_context,
            clear_source_when_rows_declare_object_name=False,
        )
        if (
            measurement_row_policy is not None
            and isinstance(record.rows, ColumnarRows)
            and measurement_row_policy.record_rows_declare_ownership(record)
        ):
            record = record.with_ownership(
                rows=record.rows,
                object_name=None,
                source_image_name=record.source_context.source_image_name,
                source_image_payload=record.source_context.source_image_payload,
            )
        records = (record,)
        if (
            measurement_row_policy is not None
            and object_name is None
            and not isinstance(record.rows, ColumnarRows)
            and measurement_row_policy.record_rows_declare_ownership(record)
        ):
            records = measurement_row_policy.record_partitions(record)
        for partition in records:
            cls.record(
                partition.materialization_request(
                    adapter=adapter,
                    name=spec.name,
                    axis_state=axis_state,
                )
            )

    @staticmethod
    def record(request: CellProfilerMeasurementMaterializationRequest) -> None:
        record_started_at = time.perf_counter()
        kwargs: CellProfilerKwargDict = {}
        if request.object_name is not _MISSING_MEASUREMENT_OBJECT_NAME:
            kwargs["object_name"] = request.object_name
        projection_started_at = time.perf_counter()
        projected_rows: MeasurementRowsInput
        projected_row_mappings: ProjectedMeasurementRows
        projected_rows, projected_row_mappings = (
            CellProfilerMeasurementAxisStateStrategy.for_state(
                request.axis_state
            ).rows_for_materialization(request)
        )
        RuntimeProfileLogger.log(logger, 
            "record_measurements_project_rows",
            time.perf_counter() - projection_started_at,
            rows=len(projected_rows),
            fields=bool(request.fields),
        )
        fields_started_at = time.perf_counter()
        fields = _measurement_fields_covering_mappings(
            request.fields,
            projected_row_mappings,
        )
        fields, projected_rows = CellProfilerMeasurementOutputProjection(
            fields=fields,
            rows=projected_rows,
        ).apply()
        RuntimeProfileLogger.log(logger, 
            "record_measurements_fields",
            time.perf_counter() - fields_started_at,
            rows=len(projected_row_mappings),
            fields=bool(fields),
        )
        if fields:
            kwargs["fields"] = fields
        kwargs["source_image_name"] = request.source_context.source_image_name
        source_metadata_started_at = time.perf_counter()
        source_metadata = request.source_payload_metadata(projected_rows)
        RuntimeProfileLogger.log(
            logger,
            "record_measurements_source_metadata",
            time.perf_counter() - source_metadata_started_at,
            rows=len(projected_rows),
        )
        if source_metadata.source_path is not None:
            kwargs["source_path"] = source_metadata.source_path
        if source_metadata.source_component_metadata is not None:
            kwargs["source_component_metadata"] = (
                source_metadata.source_component_metadata
            )
        if source_metadata.source_image_provenance_planes.has_values:
            kwargs["source_image_provenance_planes"] = (
                source_metadata.source_image_provenance_planes
            )
        add_started_at = time.perf_counter()
        request.adapter.add_measurements(
            request.name,
            projected_rows,
            **kwargs,
        )
        RuntimeProfileLogger.log(logger, 
            "record_measurements_add",
            time.perf_counter() - add_started_at,
            rows=len(projected_rows),
        )
        RuntimeProfileLogger.log(
            logger,
            "record_measurements_total",
            time.perf_counter() - record_started_at,
            artifact=request.name,
            rows=len(projected_rows),
            fields=bool(fields),
            object_name=(
                None
                if request.object_name is _MISSING_MEASUREMENT_OBJECT_NAME
                else request.object_name
            ),
        )

def _measurement_fields_covering_mappings(
    fields: tuple[FieldSpec, ...],
    rows: ProjectedMeasurementRows,
) -> tuple[FieldSpec, ...]:
    """Preserve declared table order while retaining projected semantic fields."""
    if not fields:
        return fields
    if isinstance(rows, ColumnarRows):
        declared_names = {field.name for field in fields}
        extra_names = tuple(
            str(field_name)
            for field_name in rows.columns
            if str(field_name) not in declared_names
        )
        return (*fields, *(FieldSpec(field_name) for field_name in extra_names))
    declared_names = {field.name for field in fields}
    if rows:
        first_extra_names = tuple(
            field_name
            for field_name in rows[0]
            if field_name not in declared_names
        )
        if not first_extra_names and all(
            len(row) <= len(declared_names)
            for row in rows[1:]
        ):
            return fields
        extra_names = tuple(
            dict.fromkeys(
                (
                    *first_extra_names,
                    *(
                        field_name
                        for row in rows[1:]
                        for field_name in row
                        if field_name not in declared_names
                    ),
                )
            )
        )
        return (*fields, *(FieldSpec(field_name) for field_name in extra_names))
    extra_names = tuple(
        dict.fromkeys(
            field_name
            for row in rows
            for field_name in row
            if field_name not in declared_names
        )
    )
    if not extra_names:
        return fields
    return (*fields, *(FieldSpec(field_name) for field_name in extra_names))
