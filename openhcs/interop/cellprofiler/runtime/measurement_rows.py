"""CellProfiler measurement row projection authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import ClassVar, TYPE_CHECKING, TypeVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from openhcs.core.alias_property import AliasProperty
import numpy as np

from openhcs.core.artifacts import ArtifactSpec, ArtifactSpecCollection, ObjectLabelsArtifactType
from openhcs.core.registry_strategies import RegisteredEnumMeta
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
)
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    ObjectLabelDomainScope,
    ObjectLocationMeasurementFeature,
    ObjectLabelRepresentation,
    measurement_row_mapping,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelPlaneDomainStackRequest,
    ObjectLabelDomainMetadata,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    SingletonObjectLabelStackCollapseStrategy,
    SparseIJVLabelRows,
)
from openhcs.processing.backends.lib_registry.unified_registry import runtime_output_tuple
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    MeasurementRowsInput,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter


NestedDeclarationT = TypeVar("NestedDeclarationT")


ObjectLocationFeatureValues = tuple[
    tuple[ObjectLocationMeasurementFeature, float],
    ...,
]


from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
    CellProfilerModuleAuthority,
)


class CellProfilerMeasurementStatField(str, CellProfilerModuleAuthority, Enum):
    """Base for generated absorbed-result stat-field enums."""

    field_name = AliasProperty[str]("value")


class FormattingMeasurementFeatureTemplate(
    str,
    CellProfilerModuleAuthority,
    Enum,
    metaclass=RegisteredEnumMeta,
):
    """Shared feature-name formatting contract for templated measurement names."""

    __registry_key__ = "__name__"

    def feature_name(self, **values: CellProfilerRuntimeValue) -> str:
        return self.value.format(**values)


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRowProjection(ABC):
    """Base contract for emitted CellProfiler measurement fact rows."""

    @abstractmethod
    def rows(self) -> list[CellProfilerKwargDict]:
        """Return long/tall measurement rows."""

    @staticmethod
    def measurement_row(
        *,
        feature_name: str,
        value: CellProfilerRuntimeValue,
        axis_values: Mapping[str, CellProfilerRuntimeValue] | None = None,
        value_field: MeasurementRowValueField = MeasurementRowValueField.RESULT_VALUE,
    ) -> CellProfilerKwargDict:
        resolved_axis_values = {} if axis_values is None else dict(axis_values)
        return {
            **resolved_axis_values,
            MeasurementRowAxisField.FEATURE_NAME.value: feature_name,
            value_field.value: value,
        }

    @classmethod
    def object_measurement_row(
        cls,
        *,
        object_name: str,
        object_label: int,
        feature_name: str,
        value: CellProfilerRuntimeValue,
        axis_values: Mapping[str, CellProfilerRuntimeValue] | None = None,
        value_field: MeasurementRowValueField = MeasurementRowValueField.RESULT_VALUE,
    ) -> CellProfilerKwargDict:
        resolved_axis_values = {} if axis_values is None else dict(axis_values)
        return cls.measurement_row(
            axis_values={
                MeasurementRowAxisField.OBJECT_NAME.value: object_name,
                MeasurementRowAxisField.OBJECT_LABEL.value: object_label,
                **resolved_axis_values,
            },
            feature_name=feature_name,
            value=value,
            value_field=value_field,
        )

    @classmethod
    def source_image_measurement_row(
        cls,
        *,
        source_image_name: str,
        feature_name: str,
        value: CellProfilerRuntimeValue,
        axis_values: Mapping[str, CellProfilerRuntimeValue] | None = None,
        value_field: MeasurementRowValueField = MeasurementRowValueField.RESULT_VALUE,
    ) -> CellProfilerKwargDict:
        resolved_axis_values = {} if axis_values is None else dict(axis_values)
        return cls.measurement_row(
            axis_values={
                MeasurementRowAxisField.SOURCE_IMAGE_NAME.value: source_image_name,
                **resolved_axis_values,
            },
            feature_name=feature_name,
            value=value,
            value_field=value_field,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRows(
    CellProfilerMeasurementRowProjection,
    metaclass=AutoRegisterMeta,
):
    """Registered module-result measurement row projectors."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    stable_key_axis: ClassVar[str] = RegistryKeyAttribute.REGISTRY_KEY.value
    registry_key: ClassVar[str | None] = None

    @classmethod
    def for_request(
        cls,
        module_type: type[object],
        request: object,
    ) -> "CellProfilerMeasurementRows":
        del module_type, request
        raise NotImplementedError(f"{cls.__name__} must implement for_request().")


@dataclass(frozen=True, slots=True)
class CellProfilerResultMeasurementRows(CellProfilerMeasurementRows):
    """Measurement rows projected from absorbed function result records."""

    results: CellProfilerRuntimeValue

    def source_rows(self) -> list[CellProfilerRuntimeValue]:
        return measurement_table_rows(self.results)

    @staticmethod
    def row_value(
        row: CellProfilerRuntimeValue,
        field: Enum | str,
        default: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        field_name = field.value if isinstance(field, Enum) else field
        return MappingValueLookup(
            measurement_row_mapping(row),
            str(field_name),
        ).value_or(default)

    @staticmethod
    def json_object_mapping(value: CellProfilerRuntimeValue) -> CellProfilerKwargs:
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
    def stat_field_type(self) -> type[CellProfilerMeasurementStatField]:
        return self.single_nested_declaration(
            self.module_type,
            CellProfilerMeasurementStatField,
        )

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
    coordinates: tuple[tuple[ObjectLocationMeasurementFeature, np.ndarray], ...]

    def feature_values(
        self,
        object_index: int,
    ) -> ObjectLocationFeatureValues:
        return (
            (feature, float(values[object_index]))
            for feature, values in self.coordinates
        )


@dataclass(frozen=True, slots=True)
class ObjectLocationMeasurementRows(CellProfilerMeasurementRows):
    """Emit CP object location rows from a declared object-label domain."""

    registry_key = "object_location"

    label_payload: CellProfilerRuntimeValue
    object_name: str
    include_declared_empty: bool = True
    domain_scope: ObjectLabelDomainScope | None = None

    def rows(self) -> list[CellProfilerKwargDict]:
        rows: list[CellProfilerKwargDict] = []
        label_plane_domains = self.label_plane_domains()
        for slice_index, (label_plane, domain) in enumerate(label_plane_domains):
            centers = self.centers_for_plane(
                label_plane,
                domain=domain,
            )
            rows.extend(
                self.rows_for_object(
                    object_label=object_label,
                    slice_index=slice_index,
                    feature_values=centers.feature_values(object_index),
                )
                for object_index, object_label in enumerate(centers.object_ids)
            )
        return [row for object_rows in rows for row in object_rows]

    def rows_for_object(
        self,
        *,
        object_label: int,
        slice_index: int,
        feature_values: ObjectLocationFeatureValues,
    ) -> tuple[CellProfilerKwargDict, ...]:
        return tuple(
            self.object_measurement_row(
                object_name=self.object_name,
                object_label=object_label,
                axis_values={MeasurementRowAxisField.SLICE_INDEX.value: slice_index},
                feature_name=feature.value,
                value=value,
            )
            for feature, value in feature_values
        )

    def label_planes(self) -> tuple[np.ndarray, ...]:
        label_array = np.asarray(LABEL_PAYLOAD_FINAL.value(self.label_payload))
        if self.domain_scope is ObjectLabelDomainScope.PAYLOAD:
            return (label_array,)
        if label_array.ndim <= 2:
            return (label_array,)
        return tuple(label_array[index] for index in range(label_array.shape[0]))

    def label_plane_domains(self) -> tuple[tuple[np.ndarray, tuple[int, ...]], ...]:
        if self.domain_scope is ObjectLabelDomainScope.PAYLOAD:
            label_array = np.asarray(LABEL_PAYLOAD_FINAL.value(self.label_payload))
            return ((label_array, self.object_domain_for_payload(label_array)),)
        domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            self.label_payload,
            None,
            True,
            True,
        ).stack()
        if domain_stack is not None:
            return tuple(
                (domain_stack.labels[index], object_ids)
                for index, object_ids in enumerate(domain_stack.object_id_domains)
            )
        label_planes = self.label_planes()
        slice_count = len(label_planes)
        return tuple(
            (
                label_plane,
                self.object_domain_for_plane(
                    label_plane,
                    slice_index=slice_index,
                    slice_count=slice_count,
                ),
            )
            for slice_index, label_plane in enumerate(label_planes)
        )

    def centers_for_plane(
        self,
        label_plane: CellProfilerRuntimeValue,
        *,
        domain: tuple[int, ...],
    ) -> ObjectLocationCenterValues:
        coordinates = self.dense_label_centers_for_domain(label_plane, domain)
        return ObjectLocationCenterValues(
            object_ids=domain,
            coordinates=coordinates,
        )

    def object_domain_for_payload(
        self,
        label_payload: CellProfilerRuntimeValue,
    ) -> tuple[int, ...]:
        if self.include_declared_empty and isinstance(
            self.label_payload,
            ObjectLabelDomainMetadata,
        ):
            declared_domain = (
                self.label_payload.object_label_domain().explicit_id_domain()
            )
            if declared_domain is not None:
                return declared_domain
        return self.present_domain_for_plane(
            label_payload,
            dense_extent=not self.include_declared_empty,
        )

    def object_domain_for_plane(
        self,
        label_plane: CellProfilerRuntimeValue,
        *,
        slice_index: int,
        slice_count: int,
    ) -> tuple[int, ...]:
        if self.include_declared_empty:
            declared_domain = self.declared_domain_for_plane(
                slice_index=slice_index,
                slice_count=slice_count,
            )
            if declared_domain is not None:
                return declared_domain
        return self.present_domain_for_plane(
            label_plane,
            dense_extent=not self.include_declared_empty,
        )

    def declared_domain_for_plane(
        self,
        *,
        slice_index: int,
        slice_count: int,
    ) -> tuple[int, ...] | None:
        if not isinstance(self.label_payload, ObjectLabelDomainMetadata):
            return None
        return (
            self.label_payload.object_label_domain()
            .project_slice(slice_index, slice_count)
            .explicit_id_domain()
        )

    @staticmethod
    def present_domain_for_plane(
        label_plane: CellProfilerRuntimeValue,
        *,
        dense_extent: bool,
    ) -> tuple[int, ...]:
        labels = np.asarray(label_plane)
        if labels.size == 0:
            return ()
        positive_labels = labels[labels > 0]
        if positive_labels.size == 0:
            return ()
        if dense_extent:
            return tuple(range(1, int(np.max(positive_labels)) + 1))
        return tuple(int(label_id) for label_id in np.unique(positive_labels))

    @staticmethod
    def dense_label_centers_for_domain(
        label_plane: CellProfilerRuntimeValue,
        domain: Sequence[int],
    ) -> tuple[tuple[ObjectLocationMeasurementFeature, np.ndarray], ...]:
        labels = np.asarray(label_plane, dtype=np.int64)
        center_x = np.full(len(domain), np.nan, dtype=np.float64)
        center_y = np.full(len(domain), np.nan, dtype=np.float64)
        center_z = np.full(len(domain), np.nan, dtype=np.float64)
        if not domain or labels.size == 0:
            return (
                (ObjectLocationMeasurementFeature.CENTER_X, center_x),
                (ObjectLocationMeasurementFeature.CENTER_Y, center_y),
                *(
                    ((ObjectLocationMeasurementFeature.CENTER_Z, center_z),)
                    if labels.ndim >= 3
                    else ()
                ),
            )

        positive_coordinates = np.nonzero(labels > 0)
        if not positive_coordinates or positive_coordinates[-1].size == 0:
            return (
                (ObjectLocationMeasurementFeature.CENTER_X, center_x),
                (ObjectLocationMeasurementFeature.CENTER_Y, center_y),
                *(
                    ((ObjectLocationMeasurementFeature.CENTER_Z, center_z),)
                    if labels.ndim >= 3
                    else ()
                ),
            )

        object_ids = labels[positive_coordinates]
        max_domain_label = 0
        if domain:
            max_domain_label = max(domain)
        max_label = max(int(object_ids.max()), max_domain_label)
        counts = np.bincount(object_ids, minlength=max_label + 1)
        axis_centers: list[np.ndarray] = []
        for coordinates in positive_coordinates:
            sums = np.bincount(
                object_ids,
                weights=coordinates,
                minlength=max_label + 1,
            )
            centers = np.full(max_label + 1, np.nan, dtype=np.float64)
            np.divide(sums, counts, out=centers, where=counts > 0)
            axis_centers.append(centers)
        for index, object_label in enumerate(domain):
            if object_label <= 0 or object_label >= counts.shape[0]:
                continue
            count = counts[object_label]
            if count <= 0:
                continue
            center_x[index] = axis_centers[-1][object_label]
            center_y[index] = axis_centers[-2][object_label]
            if labels.ndim >= 3:
                center_z[index] = axis_centers[-3][object_label]
        return (
            (ObjectLocationMeasurementFeature.CENTER_X, center_x),
            (ObjectLocationMeasurementFeature.CENTER_Y, center_y),
            *(
                ((ObjectLocationMeasurementFeature.CENTER_Z, center_z),)
                if labels.ndim >= 3
                else ()
            ),
        )


def _split_cellprofiler_output(raw_output: CellProfilerRuntimeValue) -> tuple[CellProfilerRuntimeValue, CellProfilerRuntimeValues]:
    raw_output = runtime_output_tuple(raw_output)
    if isinstance(raw_output, tuple):
        return raw_output[0], tuple(raw_output[1:])
    return raw_output, ()


def _measurement_rows_from_output(
    artifact_values: CellProfilerRuntimeValues,
) -> MeasurementRowsInput:
    if not artifact_values:
        return []
    rows = artifact_values[0]
    return measurement_table_rows(rows)


def measurement_table_rows(rows: CellProfilerRuntimeValue) -> MeasurementRowsInput:
    match rows:
        case ColumnarRows():
            return rows
        case RuntimeSliceAlignedValues(slices=slices) if all(
            isinstance(row, ColumnarRows) for row in slices
        ):
            return ConcatenatedMeasurementColumnarRows(slices)
        case list() as row_list if row_list and all(
            isinstance(row, ColumnarRows) for row in row_list
        ):
            return ConcatenatedMeasurementColumnarRows(tuple(row_list))
        case list() as row_list:
            return row_list
        case tuple() as row_tuple if row_tuple and all(
            isinstance(row, ColumnarRows) for row in row_tuple
        ):
            return ConcatenatedMeasurementColumnarRows(row_tuple)
        case tuple() as row_tuple:
            return list(row_tuple)
        case _:
            return [rows]


@dataclass(slots=True)
class ConcatenatedMeasurementColumnarRows(ConcatenatedColumnarRows):
    """Columnar table view over per-slice measurement column batches."""


class LabelPayloadFinalProjection:
    """Project runtime object-label payloads to their final label values."""

    def value(self, payload: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        """Return the final label plane from a runtime label payload."""
        final_labels = ObjectLabelFinalLabels(payload).value()
        return SingletonObjectLabelStackCollapseStrategy.for_labels(final_labels).collapse(
            final_labels
        )


LABEL_PAYLOAD_FINAL = LabelPayloadFinalProjection()


@dataclass(frozen=True, slots=True)
class ObjectLabelFinalLabels:
    """Resolve the final-label value from native or serialized label payloads."""

    payload: CellProfilerRuntimeValue

    def value(self) -> CellProfilerRuntimeValue:
        match self.payload:
            case ObjectLabelSet(
                representation=ObjectLabelRepresentation.SPARSE_IJV
            ) as label_set:
                return label_set
            case ObjectLabelSet() as label_set:
                payload = label_set.runtime_payload()
                if isinstance(payload, ObjectLabelPayload):
                    return payload.labels
                return payload
            case ObjectLabelPayload() as payload:
                return payload.labels
            case _:
                return self.payload


@dataclass(frozen=True, slots=True)
class ObjectLabelSmallRemovedLabels:
    """Resolve the small-removed variant from native or serialized label payloads."""

    payload: CellProfilerRuntimeValue

    def value_or_none(self) -> CellProfilerRuntimeValue | None:
        match self.payload:
            case ObjectLabelSet() as label_set:
                return label_set.small_removed_labels
            case ObjectLabelPayload() as payload:
                return payload.small_removed_labels
            case _:
                return None


def _label_payload_small_removed(payload: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue | None:
    """Return the small-removed label variant when the runtime provides it."""
    labels = ObjectLabelSmallRemovedLabels(payload).value_or_none()
    if labels is None:
        return None
    return SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(labels)

def _sparse_ijv_array(value: CellProfilerRuntimeValue) -> np.ndarray:
    if not isinstance(value, SparseIJVLabelRows):
        return np.asarray(value, dtype=np.int32)
    return np.asarray(value.as_array(), dtype=np.int32)


class ObjectLabelCountAuthority:
    """Authoritative CellProfiler object-count projection for label payloads."""

    @classmethod
    def count_from_adapter(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        object_name: str,
        *,
        slice_index: int | None = None,
    ) -> int:
        return cls.count_from_value(
            adapter.get_objects(object_name),
            slice_index=slice_index,
        )

    @staticmethod
    def count_from_value(
        value: CellProfilerRuntimeValue,
        *,
        slice_index: int | None = None,
    ) -> int:
        match value:
            case ObjectLabelPayload(labels=labels):
                pass
            case _:
                labels = value
        match labels:
            case SparseIJVLabelRows() as sparse_labels:
                label_array = _sparse_ijv_array(sparse_labels)
            case _:
                label_array = np.asarray(labels)
        match value:
            case ObjectLabelSet(
                representation=ObjectLabelRepresentation.SPARSE_IJV
            ):
                sparse_rows = SparseLabelRowsCoercion(labels).rows()
            case _:
                sparse_rows = None
        if sparse_rows is not None:
            if label_array.size == 0:
                return 0
            if slice_index is not None:
                sparse_rows = sparse_rows.slice(slice_index)
                label_array = sparse_rows.as_array()
                if label_array.size == 0:
                    return 0
            return int(np.max(label_array[:, sparse_rows.label_column]))
        match value:
            case ObjectLabelValue(declared_object_count=declared_count) if (
                declared_count is not None
                and (
                    slice_index is None
                    or label_array.ndim < 3
                    or label_array.shape[0] == 1
                )
            ):
                return int(declared_count)
        if slice_index is not None and label_array.ndim >= 3:
            if slice_index < label_array.shape[0]:
                label_array = label_array[slice_index]
            elif label_array.shape[0] == 1:
                label_array = label_array[0]
            else:
                raise ValueError(
                    "Object label stack does not contain requested slice "
                    f"{slice_index}; shape={label_array.shape!r}."
                )
        if label_array.size == 0:
            return 0
        return int(label_array.max())


@dataclass(frozen=True, slots=True)
class SparseLabelRowsCoercion:
    """Coerce sparse label data into SparseIJV rows."""

    labels: CellProfilerRuntimeValue

    def rows(self) -> SparseIJVLabelRows:
        match self.labels:
            case SparseIJVLabelRows() as sparse_rows:
                return sparse_rows
            case _:
                return SparseIJVLabelRows(self.labels)

def _measurement_object_name(
    inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    object_inputs = ArtifactSpecCollection(inputs).of_artifact_type(ObjectLabelsArtifactType)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None
