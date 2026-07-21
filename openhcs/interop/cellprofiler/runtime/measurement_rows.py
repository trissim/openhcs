"""CellProfiler measurement row projection authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
import json
import re
from string import Formatter
from types import MappingProxyType
from typing import Annotated, TypeVar, get_args, get_origin, get_type_hints

from openhcs.core.alias_property import AliasProperty
import numpy as np

from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.registry_strategies import (
    RegisteredEnumMeta,
    str_enum_member_with_payload,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    ObjectCoreMeasurementFeature,
    object_location_coordinate_arrays,
)
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxisValueProjection
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_axis_centers,
    object_label_project_plane,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    CellProfilerModuleAuthority,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerObjectCoreMeasurementFeature,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument

NestedDeclarationT = TypeVar("NestedDeclarationT")
FieldAnnotationT = TypeVar("FieldAnnotationT")


def measurement_source_image_name_for_slice(
    source_metadata: ImagePayloadMetadata,
    plane_projection: RuntimePlaneAxisValueProjection | None,
    slice_index: int,
) -> str:
    """Project one measurement row's exact source name from runtime metadata."""

    if not isinstance(source_metadata, ImagePayloadMetadata):
        raise TypeError(
            "Measurement source projection requires ImagePayloadMetadata, got "
            f"{type(source_metadata).__name__}."
        )
    if isinstance(slice_index, bool) or not isinstance(slice_index, int):
        raise TypeError(
            "Measurement source projection slice_index must be int, got "
            f"{type(slice_index).__name__}."
        )
    if slice_index < 0:
        raise ValueError(
            "Measurement source projection slice_index cannot be negative."
        )

    if plane_projection is None:
        if source_metadata.plane_axis is not None:
            raise ValueError(
                "Measurement source metadata declares plane axis "
                f"{source_metadata.plane_axis.value!r} without a runtime projection."
            )
        if slice_index != 0:
            raise ValueError(
                "Scalar measurement source metadata cannot project nonzero slice "
                f"index {slice_index}."
            )
        projected_metadata = source_metadata
    else:
        if not isinstance(plane_projection, RuntimePlaneAxisValueProjection):
            raise TypeError(
                "Measurement source projection requires "
                "RuntimePlaneAxisValueProjection or None, got "
                f"{type(plane_projection).__name__}."
            )
        if plane_projection.plane_index is not None:
            raise ValueError(
                "Measurement row source projection requires a preserved runtime "
                "plane axis, not a preselected plane."
            )
        if source_metadata.plane_axis is not plane_projection.axis:
            raise ValueError(
                "Measurement source metadata plane axis conflicts with the runtime "
                f"projection: {source_metadata.plane_axis!r} != "
                f"{plane_projection.axis!r}."
            )
        source_plane_count = source_metadata.source_provenance.source_plane_count
        if source_plane_count != plane_projection.axis_size:
            raise ValueError(
                "Measurement source metadata plane count conflicts with the runtime "
                f"projection: {source_plane_count} != "
                f"{plane_projection.axis_size}."
            )
        if slice_index >= plane_projection.axis_size:
            raise ValueError(
                "Measurement source projection slice_index exceeds the runtime "
                f"axis: {slice_index} >= {plane_projection.axis_size}."
            )
        projected_metadata = source_metadata.for_source_plane(slice_index)

    source_names = projected_metadata.source_provenance.represented_source_image_names
    if len(source_names) != 1:
        raise ValueError(
            "Measurement source projection requires exactly one source image name "
            f"for slice {slice_index}, got {source_names!r}."
        )
    return source_names[0]


class FormattingMeasurementFeatureTemplate(
    str,
    CellProfilerModuleAuthority,
    Enum,
    metaclass=RegisteredEnumMeta,
):
    """Shared feature-name formatting contract for templated measurement names."""

    __registry_key__ = "__name__"

    def __new__(
        cls,
        value: str,
        measurement_dtype: type[object] | None = None,
    ):
        return str_enum_member_with_payload(
            cls,
            value,
            payload_attribute="_measurement_dtype",
            payload=measurement_dtype,
        )

    measurement_dtype = AliasProperty[type[object] | None]("_measurement_dtype")

    @classmethod
    def database_measurement_dtype(cls) -> type[object] | None:
        """Return an external CellProfiler database type override, when declared."""

        return None

    def matches_feature_name(self, feature_name: str) -> bool:
        """Match a concrete feature name against this exact declared template."""

        template_parts: list[str] = []
        field_tokens: list[str] = []
        for index, (literal, field_name, _format_spec, _conversion) in enumerate(
            Formatter().parse(self.value)
        ):
            template_parts.append(literal)
            if field_name is not None:
                token = f"OpenHCSFormatField{index}"
                field_tokens.append(normalize_runtime_identifier(token))
                template_parts.append(token)
        pattern = re.escape(
            normalize_runtime_identifier("".join(template_parts))
        )
        for token in field_tokens:
            pattern = pattern.replace(re.escape(token), ".+")
        return re.fullmatch(
            pattern,
            normalize_runtime_identifier(feature_name),
        ) is not None

    def database_field_spec(
        self,
        name: str,
        *,
        required: bool = False,
    ) -> FieldSpec:
        """Build the CellProfiler database declaration for one emitted field."""

        database_dtype = type(self).database_measurement_dtype()
        return FieldSpec(
            name,
            self.measurement_dtype if database_dtype is None else database_dtype,
            required=required,
        )

    def feature_name(self, **values: RuntimeCallableArgument) -> str:
        return self.value.format(**values)

    def field_spec(self, name: str, *, required: bool = False) -> FieldSpec:
        """Build one emitted field from this producer-owned feature declaration."""

        if self.measurement_dtype is None:
            raise TypeError(
                f"{type(self).__name__}.{self.name} does not declare an emitted dtype."
            )
        return FieldSpec(name, self.measurement_dtype, required=required)


@dataclass(frozen=True, slots=True)
class CellProfilerResultMeasurementRows(ABC):
    """Project one schema-bearing absorbed function result into measurement rows."""

    results: ColumnarRows

    def __post_init__(self) -> None:
        if not isinstance(self.results, ColumnarRows):
            raise TypeError(
                f"{type(self).__name__} requires ColumnarRows results, got "
                f"{type(self.results).__name__}."
            )

    @classmethod
    @abstractmethod
    def for_request(
        cls,
        module_type: type[object],
        request: object,
    ) -> "CellProfilerResultMeasurementRows":
        """Bind one module-owned projector to its exact runtime request."""

    @abstractmethod
    def rows(self) -> ColumnarRows:
        """Return the exact projected measurement table."""

    def source_rows(self) -> ColumnarRows:
        return self.results

    def source_field(self, field: Enum | str) -> FieldSpec:
        """Return one exact field declared by the producer-owned row schema."""

        field_name = str(field.value if isinstance(field, Enum) else field)
        matching = tuple(
            field_spec
            for field_spec in self.source_rows().fields
            if field_spec.name == field_name
        )
        if len(matching) != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one source field "
                f"{field_name!r}, got {matching!r}."
            )
        return matching[0]

    def source_fields_annotated_with(
        self,
        row_type: type[object],
        annotation_type: type[FieldAnnotationT],
    ) -> tuple[tuple[FieldSpec, FieldAnnotationT], ...]:
        """Return producer fields carrying exactly one annotation of a given type."""

        annotations = self.producer_field_annotations(row_type)
        selected: list[tuple[FieldSpec, FieldAnnotationT]] = []
        for field_spec, metadata in annotations:
            matching = tuple(
                annotation
                for annotation in metadata
                if isinstance(annotation, annotation_type)
            )
            if len(matching) > 1:
                raise TypeError(
                    f"{row_type.__name__}.{field_spec.name} declares multiple "
                    f"{annotation_type.__name__} annotations: {matching!r}."
                )
            if matching:
                selected.append((field_spec, matching[0]))
        return tuple(selected)

    def source_field_annotated_by(
        self,
        row_type: type[object],
        annotation: object,
    ) -> FieldSpec:
        """Return the single producer field carrying one exact nominal annotation."""

        matching = tuple(
            field_spec
            for field_spec, metadata in self.producer_field_annotations(row_type)
            if any(declared is annotation for declared in metadata)
        )
        if len(matching) != 1:
            raise TypeError(
                f"{row_type.__name__} must declare exactly one field annotated by "
                f"{annotation!r}, got {matching!r}."
            )
        return matching[0]

    def producer_field_annotations(
        self,
        row_type: type[object],
    ) -> tuple[tuple[FieldSpec, tuple[object, ...]], ...]:
        """Resolve producer dataclass fields and their nominal annotation metadata."""

        if not isinstance(row_type, type) or not is_dataclass(row_type):
            raise TypeError(
                "Producer measurement row schema must be a dataclass type, got "
                f"{row_type!r}."
            )
        annotations = get_type_hints(row_type, include_extras=True)
        return tuple(
            (
                self.source_field(row_field.name),
                (
                    tuple(get_args(field_annotation)[1:])
                    if get_origin(field_annotation) is Annotated
                    else ()
                ),
            )
            for row_field in dataclass_fields(row_type)
            for field_annotation in (annotations[row_field.name],)
        )

    @staticmethod
    def json_object_mapping(
        value: RuntimeCallableArgument,
    ) -> Mapping[str, RuntimeCallableArgument]:
        if isinstance(value, Mapping):
            return value
        if value in (None, ""):
            return {}
        parsed = json.loads(str(value))
        if not isinstance(parsed, Mapping):
            raise TypeError(
                f"Expected JSON object mapping, got {type(parsed).__name__}."
            )
        return parsed


@dataclass(frozen=True, slots=True)
class ModuleOwnedResultMeasurementRows(CellProfilerResultMeasurementRows):
    """Result rows whose semantic declarations live on the owning module MRO."""

    module_type: type[object]

    @classmethod
    def nested_declarations(
        cls,
        module_type: type[object],
        base_type: type[NestedDeclarationT],
    ) -> tuple[type[NestedDeclarationT], ...]:
        if not issubclass(module_type, CellProfilerModule):
            raise TypeError(
                f"{cls.__name__} requires a CellProfilerModule owner, "
                f"got {module_type.__name__}."
            )
        return module_type.declared_authority_types(base_type)

    @classmethod
    def single_nested_declaration(
        cls,
        module_type: type[object],
        base_type: type[NestedDeclarationT],
    ) -> type[NestedDeclarationT]:
        declarations = cls.nested_declarations(module_type, base_type)
        if len(declarations) != 1:
            raise ValueError(
                f"{module_type.__name__} must declare exactly one "
                f"{base_type.__name__} nested type for {cls.__name__}, "
                f"got {[declaration.__name__ for declaration in declarations]!r}."
            )
        return declarations[0]

    @property
    def feature_template_type(self) -> type[FormattingMeasurementFeatureTemplate]:
        return self.single_nested_declaration(
            self.module_type,
            FormattingMeasurementFeatureTemplate,
        )


@dataclass(frozen=True, slots=True)
class ObjectLocationCenterValues:
    """Center coordinate values for one object-label domain."""

    object_ids: tuple[int, ...]
    coordinates: tuple[tuple[CellProfilerObjectCoreMeasurementFeature, np.ndarray], ...]

    def feature_values(
        self,
        object_index: int,
    ) -> tuple[tuple[CellProfilerObjectCoreMeasurementFeature, float], ...]:
        return tuple(
            (feature, float(values[object_index]))
            for feature, values in self.coordinates
        )


@dataclass(frozen=True, slots=True)
class ObjectLocationMeasurementRows:
    """Emit CP object location rows from a declared object-label domain."""

    label_payload: ObjectLabelValue
    object_name: str
    domain_scope: ObjectLabelDomainScope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label_payload, ObjectLabelValue):
            raise TypeError(
                "ObjectLocationMeasurementRows requires an ObjectLabelValue, "
                f"got {type(self.label_payload).__name__}."
            )
        declared_scope = self.label_payload.object_label_domain().scope
        if self.domain_scope is not None and self.domain_scope is not declared_scope:
            raise ValueError(
                "ObjectLocationMeasurementRows domain_scope must match the "
                f"object-label declaration; got {self.domain_scope.value!r}, "
                f"expected {declared_scope.value!r}."
            )

    def rows(self) -> MeasurementProjectedColumnarRows:
        object_names: list[str] = []
        object_labels: list[int] = []
        slice_indices: list[int] = []
        feature_names: list[str] = []
        values: list[float] = []
        label_plane_domains = self.label_plane_domains()
        for slice_index, (label_plane, domain) in enumerate(label_plane_domains):
            centers = self.centers_for_plane(
                label_plane,
                domain=domain,
            )
            for object_index, object_label in enumerate(centers.object_ids):
                for feature, value in centers.feature_values(object_index):
                    object_names.append(self.object_name)
                    object_labels.append(object_label)
                    slice_indices.append(slice_index)
                    feature_names.append(feature.value)
                    values.append(value)
        columns = MappingProxyType(
            {
                MeasurementRowAxisField.OBJECT_NAME.value: tuple(object_names),
                MeasurementRowAxisField.OBJECT_LABEL.value: tuple(object_labels),
                MeasurementRowAxisField.SLICE_INDEX.value: tuple(slice_indices),
                MeasurementRowAxisField.FEATURE_NAME.value: tuple(feature_names),
                MeasurementRowValueField.RESULT_VALUE.value: tuple(values),
            }
        )
        return MeasurementProjectedColumnarRows(
            columns,
            fields=(
                FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
                FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
                FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
                FieldSpec(MeasurementRowAxisField.FEATURE_NAME.value, str),
                FieldSpec(MeasurementRowValueField.RESULT_VALUE.value, float),
            ),
        )

    def label_plane_domains(self) -> tuple[tuple[object, tuple[int, ...]], ...]:
        domain = self.label_payload.object_label_domain()
        if domain.scope is ObjectLabelDomainScope.PAYLOAD:
            object_ids = domain.explicit_id_domain()
            if object_ids is None:
                raise ValueError(
                    "ObjectLocationMeasurementRows requires an explicit payload "
                    "object-ID domain."
                )
            return ((self.label_payload, object_ids),)

        object_id_domains = domain.declared_object_id_domains
        if not object_id_domains:
            raise ValueError(
                "ObjectLocationMeasurementRows requires one declared object-ID "
                "domain per label plane."
            )
        plane_count = self.label_payload.declared_plane_count()
        if plane_count != len(object_id_domains):
            raise ValueError(
                "ObjectLocationMeasurementRows label-plane cardinality must match "
                f"its declared domains; got {plane_count!r} planes and "
                f"{len(object_id_domains)} domains."
            )
        return tuple(
            (
                object_label_project_plane(
                    self.label_payload.labels,
                    plane_index,
                    plane_count=plane_count,
                ),
                object_ids,
            )
            for plane_index, object_ids in enumerate(object_id_domains)
        )

    def centers_for_plane(
        self,
        label_plane: RuntimeCallableArgument,
        *,
        domain: tuple[int, ...],
    ) -> ObjectLocationCenterValues:
        axis_centers, counts = object_label_axis_centers(
            label_plane,
            domain=domain,
        )
        return ObjectLocationCenterValues(
            object_ids=domain,
            coordinates=tuple(
                (
                    CellProfilerObjectCoreMeasurementFeature[
                        ObjectCoreMeasurementFeature(feature_name).name
                    ],
                    np.asarray(coordinate.values, dtype=np.float64)[
                        np.asarray(domain, dtype=np.int64)
                    ],
                )
                for feature_name, coordinate in object_location_coordinate_arrays(
                    axis_centers,
                    counts,
                )
            ),
        )


def measurement_table_rows(rows: RuntimeCallableArgument) -> ColumnarRows:
    match rows:
        case ColumnarRows():
            return rows
        case RuntimeSliceAlignedValues(slices=slices) if all(
            isinstance(row, ColumnarRows) for row in slices
        ):
            return ConcatenatedColumnarRows(slices)
        case _:
            raise TypeError(
                "CellProfiler measurement outputs must be schema-bearing "
                f"ColumnarRows, got {type(rows).__name__}."
            )
