"""CellProfiler measurement row projection authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.registry_strategies import (
    GeneratedEnumClassSpec,
    RegisteredEnumMeta,
)
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    ConcatenatedColumnarRows,
)
from openhcs.core.measurement_feature_queries import (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_RESULT_VALUE_FIELD,
)
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
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


ObjectLocationFeatureValues = tuple[
    tuple[ObjectLocationMeasurementFeature, float],
    ...,
]


class CellProfilerMeasurementStatField(str, Enum):
    """Base for generated absorbed-result stat-field enums."""

    field_name = AliasProperty[str]("value")


for _measurement_stat_field_spec in (
    GeneratedEnumClassSpec(
        class_name="ClassifyObjectsMeasurementStatField",
        base_type=CellProfilerMeasurementStatField,
        members={
            "BIN_COUNTS": "bin_counts",
            "BIN_PERCENTAGES": "bin_percentages",
            "OBJECT_CLASSES": "object_classes",
            "TOTAL_OBJECTS": "total_objects",
            "SLICE_INDEX": MeasurementRowAxisField.SLICE_INDEX.value,
        },
    ),
):
    _measurement_stat_field_spec.declare_in(globals())


class FormattingMeasurementFeatureTemplate(str, Enum, metaclass=RegisteredEnumMeta):
    """Shared feature-name formatting contract for templated measurement names."""

    __registry_key__ = "__name__"

    def feature_name(self, **values: CellProfilerRuntimeValue) -> str:
        return self.value.format(**values)


for _measurement_feature_template_spec in (
    GeneratedEnumClassSpec(
        class_name="ClassifyObjectsMeasurementFeatureTemplate",
        base_type=FormattingMeasurementFeatureTemplate,
        members={
            "OBJECTS_PER_BIN": "Classify_{bin_name}_NumObjectsPerBin",
            "PERCENT_PER_BIN": "Classify_{bin_name}_PctObjectsPerBin",
            "OBJECT_CLASS": "Classify_{bin_name}",
        },
    ),
):
    _measurement_feature_template_spec.declare_in(globals())


for _measurement_stat_field_spec in (
    GeneratedEnumClassSpec(
        class_name="AlignMeasurementStatField",
        base_type=CellProfilerMeasurementStatField,
        members={
            "OUTPUT_INDEX": "output_index",
            "SLICE_INDEX": MeasurementRowAxisField.SLICE_INDEX.value,
            "X_SHIFT": "x_shift",
            "Y_SHIFT": "y_shift",
        },
    ),
):
    _measurement_stat_field_spec.declare_in(globals())


class AlignMeasurementFeature(str, Enum):
    """CellProfiler Align feature names."""

    X_SHIFT = "Align_Xshift"
    Y_SHIFT = "Align_Yshift"

    @property
    def source_field(self) -> AlignMeasurementStatField:
        return AlignMeasurementStatField[self.name]


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRowProjection(ABC):
    """Base contract for emitted CellProfiler measurement fact rows."""

    @abstractmethod
    def rows(self) -> list[CellProfilerKwargDict]:
        """Return long/tall measurement rows."""


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRows(
    CellProfilerMeasurementRowProjection,
    metaclass=AutoRegisterMeta,
):
    """Registered module-result measurement row projectors."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    stable_key_axis: ClassVar[str] = RegistryKeyAttribute.REGISTRY_KEY.value
    registry_key: ClassVar[str | None] = None


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
class ClassifyObjectsMeasurementRows(CellProfilerResultMeasurementRows):
    """Project absorbed ClassifyObjects results into CP measurement rows."""

    registry_key = "classify_objects"

    object_name: str | None

    def rows(self) -> list[CellProfilerKwargDict]:
        rows: list[CellProfilerKwargDict] = []
        for result in self.source_rows():
            bin_counts = self.json_object_mapping(
                self.row_value(
                    result, ClassifyObjectsMeasurementStatField.BIN_COUNTS, {}
                )
            )
            bin_percentages = self.json_object_mapping(
                self.row_value(
                    result,
                    ClassifyObjectsMeasurementStatField.BIN_PERCENTAGES,
                    {},
                )
            )
            object_classes = self.json_object_mapping(
                self.row_value(
                    result,
                    ClassifyObjectsMeasurementStatField.OBJECT_CLASSES,
                    {},
                )
            )
            slice_index = int(
                self.row_value(
                    result, ClassifyObjectsMeasurementStatField.SLICE_INDEX, 0
                )
            )
            bin_names = tuple(str(name) for name in bin_counts)
            for bin_name, count in bin_counts.items():
                rows.append(
                    {
                        MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                        MEASUREMENT_FEATURE_NAME_FIELD: (
                            ClassifyObjectsMeasurementFeatureTemplate.OBJECTS_PER_BIN.feature_name(
                                bin_name=str(bin_name)
                            )
                        ),
                        MEASUREMENT_RESULT_VALUE_FIELD: count,
                    }
                )
                rows.append(
                    {
                        MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                        MEASUREMENT_FEATURE_NAME_FIELD: (
                            ClassifyObjectsMeasurementFeatureTemplate.PERCENT_PER_BIN.feature_name(
                                bin_name=str(bin_name)
                            )
                        ),
                        MEASUREMENT_RESULT_VALUE_FIELD: MappingValueLookup(
                            bin_percentages,
                            bin_name,
                        ).value_or(0.0),
                    }
                )
            rows.extend(
                self.object_class_rows(
                    object_classes=object_classes,
                    bin_names=bin_names,
                    result=result,
                    slice_index=slice_index,
                )
            )
        return rows

    def object_class_rows(
        self,
        *,
        object_classes: CellProfilerKwargs,
        bin_names: tuple[str, ...],
        result: CellProfilerRuntimeValue,
        slice_index: int,
    ) -> list[CellProfilerKwargDict]:
        if self.object_name is None:
            return []
        total_objects = int(
            self.row_value(
                result,
                ClassifyObjectsMeasurementStatField.TOTAL_OBJECTS,
                0,
            )
        )
        class_labels = tuple(sorted(int(label) for label in object_classes))
        dense_labels = tuple(range(1, total_objects + 1))
        object_labels = tuple(dict.fromkeys((*dense_labels, *class_labels)))
        return [
            {
                MEASUREMENT_OBJECT_NAME_FIELD: self.object_name,
                MEASUREMENT_OBJECT_LABEL_FIELD: object_label,
                MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                MEASUREMENT_FEATURE_NAME_FIELD: (
                    ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                        bin_name=bin_name
                    )
                ),
                MEASUREMENT_RESULT_VALUE_FIELD: int(
                    object_classes.get(str(object_label)) == bin_name
                ),
            }
            for object_label in object_labels
            for bin_name in bin_names
        ]


@dataclass(frozen=True, slots=True)
class AlignMeasurementRows(CellProfilerResultMeasurementRows):
    """Project absorbed Align results into CP image measurement rows."""

    registry_key = "align"

    output_names: tuple[str, ...]
    features: ClassVar[tuple[AlignMeasurementFeature, ...]] = (
        AlignMeasurementFeature.X_SHIFT,
        AlignMeasurementFeature.Y_SHIFT,
    )

    def rows(self) -> list[CellProfilerKwargDict]:
        rows: list[CellProfilerKwargDict] = []
        for result in self.source_rows():
            output_index = int(
                self.row_value(result, AlignMeasurementStatField.OUTPUT_INDEX, 0)
            )
            if output_index < 0 or output_index >= len(self.output_names):
                raise ValueError(
                    f"Align measurement output_index {output_index} does not match "
                    f"declared image outputs {self.output_names!r}."
                )
            slice_index = int(
                self.row_value(result, AlignMeasurementStatField.SLICE_INDEX, 0)
            )
            source_image_name = self.output_names[output_index]
            rows.extend(
                {
                    MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD: source_image_name,
                    MEASUREMENT_FEATURE_NAME_FIELD: feature.value,
                    MEASUREMENT_RESULT_VALUE_FIELD: float(
                        self.row_value(result, feature.source_field, 0.0)
                    ),
                }
                for feature in type(self).features
            )
        return rows


for _measurement_stat_field_spec in (
    GeneratedEnumClassSpec(
        class_name="ThresholdMeasurementStatField",
        base_type=CellProfilerMeasurementStatField,
        members={
            "SLICE_INDEX": "slice_index",
            "THRESHOLD_USED": "threshold_used",
            "THRESHOLD_VALUE": "threshold_value",
            "FINAL_THRESHOLD": "final_threshold",
            "ORIGINAL_THRESHOLD": "original_threshold",
            "WEIGHTED_VARIANCE": "weighted_variance",
            "SUM_OF_ENTROPIES": "sum_of_entropies",
        },
    ),
):
    _measurement_stat_field_spec.declare_in(globals())


for _measurement_feature_template_spec in (
    GeneratedEnumClassSpec(
        class_name="ThresholdMeasurementFeatureTemplate",
        base_type=FormattingMeasurementFeatureTemplate,
        members={
            "FINAL_THRESHOLD": "FinalThreshold_{object_name}",
            "ORIGINAL_THRESHOLD": "OrigThreshold_{object_name}",
            "WEIGHTED_VARIANCE": "WeightedVariance_{object_name}",
            "SUM_OF_ENTROPIES": "SumOfEntropies_{object_name}",
        },
    ),
):
    _measurement_feature_template_spec.declare_in(globals())


@dataclass(frozen=True, slots=True)
class ThresholdMeasurementStatSchema:
    """Nominal mapping from supported threshold stat rows to CP features."""

    final_threshold_fields: tuple[ThresholdMeasurementStatField, ...] = (
        ThresholdMeasurementStatField.THRESHOLD_USED,
        ThresholdMeasurementStatField.THRESHOLD_VALUE,
        ThresholdMeasurementStatField.FINAL_THRESHOLD,
    )

    def final_threshold(self, row: CellProfilerKwargs) -> CellProfilerRuntimeValue:
        for field in self.final_threshold_fields:
            if field.value in row:
                return row[field.value]
        raise KeyError(
            "Threshold measurement row does not expose any known final-threshold "
            f"field: {tuple(field.value for field in self.final_threshold_fields)!r}."
        )

    def value_or_default(
        self,
        row: CellProfilerKwargs,
        field: ThresholdMeasurementStatField,
        default: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        if field.value in row:
            return row[field.value]
        return default


@dataclass(frozen=True, slots=True)
class ThresholdMeasurementRows(CellProfilerResultMeasurementRows):
    """Project absorbed threshold stats into CP image measurement rows."""

    registry_key = "threshold"

    object_name: str
    schema: ThresholdMeasurementStatSchema = ThresholdMeasurementStatSchema()

    def rows(self) -> list[CellProfilerKwargDict]:
        rows: list[CellProfilerKwargDict] = []
        for slice_stats in self.source_rows():
            stat_row = measurement_row_mapping(slice_stats)
            slice_index = self.schema.value_or_default(
                stat_row,
                ThresholdMeasurementStatField.SLICE_INDEX,
                0,
            )
            final_threshold = self.schema.final_threshold(stat_row)
            values = {
                ThresholdMeasurementFeatureTemplate.FINAL_THRESHOLD.feature_name(
                    object_name=self.object_name
                ): final_threshold,
                ThresholdMeasurementFeatureTemplate.ORIGINAL_THRESHOLD.feature_name(
                    object_name=self.object_name
                ): self.schema.value_or_default(
                    stat_row,
                    ThresholdMeasurementStatField.ORIGINAL_THRESHOLD,
                    final_threshold,
                ),
                ThresholdMeasurementFeatureTemplate.WEIGHTED_VARIANCE.feature_name(
                    object_name=self.object_name
                ): self.schema.value_or_default(
                    stat_row,
                    ThresholdMeasurementStatField.WEIGHTED_VARIANCE,
                    0.0,
                ),
                ThresholdMeasurementFeatureTemplate.SUM_OF_ENTROPIES.feature_name(
                    object_name=self.object_name
                ): self.schema.value_or_default(
                    stat_row,
                    ThresholdMeasurementStatField.SUM_OF_ENTROPIES,
                    0.0,
                ),
            }
            rows.extend(
                {
                    MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                    MEASUREMENT_FEATURE_NAME_FIELD: feature_name,
                    MEASUREMENT_RESULT_VALUE_FIELD: value,
                }
                for feature_name, value in values.items()
            )
        return rows


@dataclass(frozen=True, slots=True)
class ObjectLocationCenterValues:
    """XY center values for one object-label domain."""

    object_ids: tuple[int, ...]
    center_y: np.ndarray
    center_x: np.ndarray

    def feature_values(
        self,
        object_index: int,
    ) -> ObjectLocationFeatureValues:
        return (
            (
                ObjectLocationMeasurementFeature.CENTER_X,
                float(self.center_x[object_index]),
            ),
            (
                ObjectLocationMeasurementFeature.CENTER_Y,
                float(self.center_y[object_index]),
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectLocationMeasurementRows(CellProfilerMeasurementRows):
    """Emit CP object location rows from a declared object-label domain."""

    registry_key = "object_location"

    label_payload: CellProfilerRuntimeValue
    object_name: str
    include_declared_empty: bool = True

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
            {
                MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                MEASUREMENT_OBJECT_NAME_FIELD: self.object_name,
                MEASUREMENT_OBJECT_LABEL_FIELD: object_label,
                MEASUREMENT_FEATURE_NAME_FIELD: feature.value,
                MEASUREMENT_RESULT_VALUE_FIELD: value,
            }
            for feature, value in feature_values
        )

    def label_planes(self) -> tuple[np.ndarray, ...]:
        label_array = np.asarray(LABEL_PAYLOAD_FINAL.value(self.label_payload))
        if label_array.ndim <= 2:
            return (label_array,)
        return tuple(label_array[index] for index in range(label_array.shape[0]))

    def label_plane_domains(self) -> tuple[tuple[np.ndarray, tuple[int, ...]], ...]:
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
        center_y, center_x = self.dense_label_centers_for_domain(label_plane, domain)
        return ObjectLocationCenterValues(
            object_ids=domain,
            center_y=center_y,
            center_x=center_x,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(label_plane, dtype=np.int64)
        center_y = np.full(len(domain), np.nan, dtype=np.float64)
        center_x = np.full(len(domain), np.nan, dtype=np.float64)
        if not domain or labels.size == 0:
            return center_y, center_x

        y_indices, x_indices = np.nonzero(labels > 0)
        if y_indices.size == 0:
            return center_y, center_x

        object_ids = labels[y_indices, x_indices]
        max_domain_label = 0
        if domain:
            max_domain_label = max(domain)
        max_label = max(int(object_ids.max()), max_domain_label)
        counts = np.bincount(object_ids, minlength=max_label + 1)
        y_sums = np.bincount(object_ids, weights=y_indices, minlength=max_label + 1)
        x_sums = np.bincount(object_ids, weights=x_indices, minlength=max_label + 1)
        for index, object_label in enumerate(domain):
            if object_label <= 0 or object_label >= counts.shape[0]:
                continue
            count = counts[object_label]
            if count <= 0:
                continue
            center_y[index] = y_sums[object_label] / count
            center_x[index] = x_sums[object_label] / count
        return center_y, center_x


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
    object_inputs = ArtifactSpecCollection(inputs).of_kind(ArtifactKind.OBJECT_LABELS)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None
