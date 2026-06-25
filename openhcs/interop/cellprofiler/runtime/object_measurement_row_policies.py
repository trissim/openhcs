"""CellProfiler object-measurement row ownership and completion policies."""

from __future__ import annotations

from abc import ABC
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from inspect import unwrap
from typing import ClassVar

from metaclass_registry import RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactSpecCollection
from openhcs.core.registry_strategies import GeneratedLeafClassSpec
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MeasurementRowOwnership,
    measurement_object_label,
    measurement_row_has_object_identity,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementScalarLiteral,
    ObjectLabelDomainScope,
    ObjectShapeMeasurementFeature,
    RuntimePlaneAxis,
    dense_object_label_id_domain,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    ImagePayloadMetadataCompositionMode,
    MeasurementTable,
    ObjectLabelValue,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerMeasurementImage,
    CellProfilerSourceImagePair,
    CellProfilerSourcePairFeature,
)
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementRecord,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import LABEL_PAYLOAD_FINAL
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_row_source_names_required,
)
from openhcs.interop.cellprofiler.runtime.module_names import (
    CELLPROFILER_MEASURE_COLOCALIZATION_MODULE,
    CELLPROFILER_MEASURE_GRANULARITY_MODULE,
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_MODULE,
    CELLPROFILER_MEASURE_OBJECT_NEIGHBORS_MODULE,
    CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE,
    CELLPROFILER_MEASURE_TEXTURE_MODULE,
    CELLPROFILER_TRACK_OBJECTS_MODULE,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    MISSING_MEASUREMENT_ROW_VALUE,
    MeasurementObjectRowIdentityProjectionStrategy,
    MissingObjectMeasurementValuePolicy,
    MissingObjectMeasurementValueRequest,
    MissingObjectMeasurementValueStrategy,
    ObjectMeasurementIdsByAxis,
    ObjectMeasurementIdsByAxisView,
    ObjectMeasurementProjectedRowKeys,
    ObjectMeasurementRowCompletionSchema,
    ObjectMeasurementRowIdentityProjectionRequest,
    ObjectMeasurementRowIdentityProjectionResult,
    ObjectMeasurementRowsByName,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
    MeasurementRowMapping,
    MeasurementRowsInput,
    MissingObjectMeasurementCellValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLeafSpec,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyMultiBaseLeafSpec,
    CellProfilerModulePolicyRegistryKey,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    RuntimeShapeInspection,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.processing.backends.cellprofiler.texture import measure_texture_objects

_MEASURE_OBJECT_SIZE_SHAPE_MODULE = CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE
_MEASURE_OBJECT_INTENSITY_MODULE = CELLPROFILER_MEASURE_OBJECT_INTENSITY_MODULE
_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE = (
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE
)
_MEASURE_TEXTURE_MODULE = CELLPROFILER_MEASURE_TEXTURE_MODULE
_MEASURE_COLOCALIZATION_MODULE = CELLPROFILER_MEASURE_COLOCALIZATION_MODULE
_MEASURE_GRANULARITY_MODULE = CELLPROFILER_MEASURE_GRANULARITY_MODULE
_MEASURE_OBJECT_NEIGHBORS_MODULE = CELLPROFILER_MEASURE_OBJECT_NEIGHBORS_MODULE
_TRACK_OBJECTS_MODULE = CELLPROFILER_TRACK_OBJECTS_MODULE

@dataclass(frozen=True, slots=True)
class ObjectMeasurementInvocation:
    """One semantic object-measurement function invocation."""

    kwargs: CellProfilerKwargs
    source_pair: CellProfilerSourceImagePair | None = None

    def lowered_kwargs(self) -> CellProfilerKwargDict:
        """Return kwargs lowered to the CellProfiler function-call ABI."""
        return dict(self.kwargs)


@dataclass(frozen=True, slots=True)
class SourcePairObjectMeasurementInvocation(ObjectMeasurementInvocation):
    """Object measurement invocation over one ordered source-image pair."""

    first_channel_kwarg: str = "channel_1"
    second_channel_kwarg: str = "channel_2"

    def __post_init__(self) -> None:
        if self.source_pair is None:
            raise ValueError(
                "SourcePairObjectMeasurementInvocation requires a source_pair."
            )

    def lowered_kwargs(self) -> CellProfilerKwargDict:
        assert self.source_pair is not None
        return {
            **self.kwargs,
            **self.source_pair.invocation_kwargs(
                first_channel_kwarg=self.first_channel_kwarg,
                second_channel_kwarg=self.second_channel_kwarg,
            ),
        }


@dataclass(frozen=True, slots=True)
class ObjectMeasurementResultPayloadRequest:
    """Row fields needed to decide whether an object measurement row has values."""

    row_mapping: MeasurementRowMapping
    object_id_field: str
    axis_fields: tuple[str, ...]

    @property
    def metadata_fields(self) -> frozenset[str]:
        return frozenset(
            (
                self.object_id_field,
                *MEASUREMENT_OBJECT_ID_FIELDS,
                *self.axis_fields,
                MEASUREMENT_OBJECT_NAME_FIELD,
                MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
            )
        )


@dataclass(frozen=True, slots=True)
class ExplicitMeasurementRowOwnershipGroups:
    """Rows grouped by explicit object/source ownership fields."""

    object_rows: ObjectMeasurementRowsByName
    image_rows: ObjectMeasurementRowsByName
    unowned_rows: tuple[CellProfilerRuntimeValue, ...]

    @classmethod
    def from_record(
        cls,
        record: CellProfilerMeasurementRecord,
    ) -> "ExplicitMeasurementRowOwnershipGroups":
        object_rows: ObjectMeasurementRowsByName = {}
        image_rows: ObjectMeasurementRowsByName = {}
        unowned_rows: list[CellProfilerRuntimeValue] = []
        for row in record.rows:
            row_mapping = measurement_row_mapping(row)
            object_name = measurement_row_object_name(row_mapping)
            source_image_name = measurement_row_source_image_name(row_mapping)
            if object_name is not None:
                if object_name not in object_rows:
                    object_rows[object_name] = []
                object_rows[object_name].append(row)
                continue
            if source_image_name is not None:
                if source_image_name not in image_rows:
                    image_rows[source_image_name] = []
                image_rows[source_image_name].append(row)
                continue
            unowned_rows.append(row)
        return cls(
            object_rows=object_rows,
            image_rows=image_rows,
            unowned_rows=tuple(unowned_rows),
        )

    @property
    def has_mixed_owner_kinds(self) -> bool:
        return bool(self.object_rows and self.image_rows)


class CellProfilerObjectMeasurementRowPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal export-row policy for object-scoped measurement modules."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    row_identity: ClassVar[MeasurementObjectRowIdentity] = (
        MeasurementObjectRowIdentity.LABEL_ID
    )
    missing_value_policy: ClassVar[MissingObjectMeasurementValuePolicy] = (
        MissingObjectMeasurementValuePolicy.NAN
    )
    explicit_row_ownership_required: ClassVar[bool] = False

    def object_identity(self) -> MeasurementObjectRowIdentity:
        """Return the object identity projection for rows emitted by this module."""
        return MeasurementObjectRowIdentity(type(self).row_identity)

    def object_identity_for_label_payload(
        self,
        label_payload: CellProfilerRuntimeValue,
    ) -> MeasurementObjectRowIdentity:
        """Return row identity for a concrete object-measurement label domain."""
        del label_payload
        return self.object_identity()

    def row_identity_axis_fields(
        self,
        axis_fields: Sequence[str],
        *,
        label_payload: CellProfilerRuntimeValue | None = None,
    ) -> tuple[str, ...]:
        """Return row-axis fields that partition object-row identity."""
        del label_payload
        return tuple(axis_fields)

    def invocations(
        self,
        measurement_image: CellProfilerMeasurementImage,
        kwargs: CellProfilerKwargs,
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        """Return semantic function invocations for this measurement image."""
        del measurement_image
        return (ObjectMeasurementInvocation(kwargs=kwargs),)

    def project_rows(
        self,
        rows: MeasurementRowsInput,
        invocation: ObjectMeasurementInvocation,
    ) -> MeasurementRowsInput:
        """Return emitted rows projected into this module's feature namespace."""
        del invocation
        if isinstance(rows, ColumnarRows):
            return rows
        return list(rows)

    def split_scoped_rows(
        self,
        rows: MeasurementRowsInput,
    ) -> tuple[MeasurementRowsInput, CellProfilerRuntimeValueSequence]:
        """Partition object-scoped measurement rows from image-scoped rows."""
        if isinstance(rows, ColumnarRows):
            return rows, ()
        object_rows: list[CellProfilerRuntimeValue] = []
        non_object_rows: list[CellProfilerRuntimeValue] = []
        for row in rows:
            if self.row_is_object_scoped(row):
                object_rows.append(row)
            else:
                non_object_rows.append(row)
        return object_rows, non_object_rows

    def table_source_image_name(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        source_image_name: str | None,
    ) -> str | None:
        """Return table-level source ownership for rows emitted by this policy."""
        if not measurement_images:
            return source_image_name
        return CellProfilerMeasurementImage.shared_source_image_name(measurement_images)

    def row_source_owner(
        self,
        measurement_image: CellProfilerMeasurementImage,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
    ) -> str | None:
        """Return row-level source ownership for one measurement image."""
        if measurement_row_source_names_required(measurement_images):
            return measurement_image.source_image_name
        return None

    def source_metadata_composition_mode(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
    ) -> ImagePayloadMetadataCompositionMode | None:
        """Return source metadata topology for this policy's measurement rows."""
        if any(
            self.row_source_owner(measurement_image, measurement_images) is not None
            for measurement_image in measurement_images
        ):
            return ImagePayloadMetadataCompositionMode.STACK
        return None

    def row_ownership(
        self,
        *,
        measurement_image: CellProfilerMeasurementImage,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        object_name: str | None,
        object_inputs: tuple[ArtifactSpec, ...],
        contains_image_measurement_rows: bool,
    ) -> MeasurementRowOwnership:
        """Return row-level object/source ownership for one measurement table."""
        row_object_name = None
        if object_name is not None:
            row_object_name = self.row_object_owner(
                object_name,
                object_inputs=object_inputs,
                measurement_images=measurement_images,
                contains_image_measurement_rows=contains_image_measurement_rows,
            )
        return MeasurementRowOwnership(
            object_name=row_object_name,
            source_image_name=self.row_source_owner(
                measurement_image,
                measurement_images,
            ),
        )

    def row_object_owner(
        self,
        object_name: str,
        *,
        object_inputs: tuple[ArtifactSpec, ...],
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        contains_image_measurement_rows: bool,
    ) -> str | None:
        """Return row-level object ownership for one object measurement table."""
        if (
            contains_image_measurement_rows
            or len(object_inputs) != 1
            or measurement_row_source_names_required(measurement_images)
            or self.requires_explicit_row_ownership()
        ):
            return object_name
        return None

    def table_object_owner(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
        *,
        contains_image_measurement_rows: bool = False,
    ) -> str | None:
        """Return table-level object ownership for materialized measurements."""
        if (
            contains_image_measurement_rows
            or len(object_inputs) != 1
            or self.requires_explicit_row_ownership()
        ):
            return None
        return object_inputs[0].name

    def requires_explicit_row_ownership(self) -> bool:
        """Return whether emitted rows carry mixed measurement ownership."""
        return type(self).explicit_row_ownership_required

    def record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return single-owner measurement table partitions for a record."""
        if self.requires_explicit_row_ownership():
            return self.explicit_owner_record_partitions(
                record,
                require_declared_ownership=True,
            )
        if self.record_rows_declare_ownership(record):
            return self.explicit_owner_record_partitions(
                record,
                require_declared_ownership=False,
            )
        return (
            record.with_ownership(
                rows=record.rows,
                object_name=record.object_name,
                source_image_name=record.source_context.source_image_name,
                source_image_payload=record.source_context.source_image_payload,
            ),
        )

    @staticmethod
    def record_rows_declare_ownership(
        record: CellProfilerMeasurementRecord,
    ) -> bool:
        """Return whether row-level object/source ownership is present."""
        if isinstance(record.rows, ColumnarRows):
            columns = tuple(str(column) for column in record.rows.columns)
            return (
                MEASUREMENT_OBJECT_NAME_FIELD in columns
                or MEASUREMENT_SOURCE_IMAGE_NAME_FIELD in columns
            )
        return any(
            (
                measurement_row_object_name(row_mapping) is not None
                or measurement_row_source_image_name(row_mapping) is not None
            )
            for row in record.rows
            for row_mapping in (measurement_row_mapping(row),)
        )

    def unowned_record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
        rows: CellProfilerRuntimeValueSequence,
        *,
        require_declared_ownership: bool,
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return neutral partitions for rows without row-level ownership."""
        if not rows:
            return ()
        if require_declared_ownership:
            raise ValueError(
                f"{type(self).__name__} requires every mixed-scope measurement row "
                "to declare object or source-image ownership."
            )
        return (
            record.with_ownership(
                rows=rows,
                object_name=record.object_name,
                source_image_name=None,
                source_image_payload=None,
            ),
        )

    def explicit_owner_record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
        *,
        require_declared_ownership: bool = True,
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return fail-loud table ownership for rows with explicit row owners."""
        groups = ExplicitMeasurementRowOwnershipGroups.from_record(record)
        if groups.has_mixed_owner_kinds:
            return self.mixed_owner_record_partitions(record, groups)
        if groups.image_rows:
            return self.image_owner_record_partitions(record, groups)
        return self.object_and_unowned_record_partitions(
            record=record,
            groups=groups,
            require_declared_ownership=require_declared_ownership,
        )

    def mixed_owner_record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
        groups: ExplicitMeasurementRowOwnershipGroups,
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return a table-level object partition for mixed explicit owners."""
        if len(groups.object_rows) != 1:
            raise ValueError(
                f"{type(self).__name__} requires one table-level object owner "
                f"for mixed measurement rows, got {tuple(groups.object_rows)}."
            )
        object_name = next(iter(groups.object_rows))
        return (
            record.with_ownership(
                rows=record.rows,
                object_name=object_name,
                source_image_name=None,
                source_image_payload=record.source_context.source_image_payload,
            ),
        )

    def image_owner_record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
        groups: ExplicitMeasurementRowOwnershipGroups,
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return a table-level source-image partition for explicit image rows."""
        if len(groups.image_rows) != 1:
            return (
                record.with_ownership(
                    rows=record.rows,
                    object_name=None,
                    source_image_name=None,
                    source_image_payload=None,
                ),
            )
        source_image_name = next(iter(groups.image_rows))
        return (
            record.with_ownership(
                rows=record.rows,
                object_name=None,
                source_image_name=source_image_name,
                source_image_payload=record.source_context.source_image_payload,
            ),
        )

    def object_and_unowned_record_partitions(
        self,
        *,
        record: CellProfilerMeasurementRecord,
        groups: ExplicitMeasurementRowOwnershipGroups,
        require_declared_ownership: bool,
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return object-owner partitions plus permitted unowned rows."""
        return (
            *self.unowned_record_partitions(
                record,
                groups.unowned_rows,
                require_declared_ownership=require_declared_ownership,
            ),
            *(
                record.with_ownership(
                    rows=rows,
                    object_name=object_name,
                    source_image_name=None,
                    source_image_payload=record.source_context.source_image_payload,
                )
                for object_name, rows in groups.object_rows.items()
            ),
            *(
                record.with_ownership(
                    rows=rows,
                    object_name=None,
                    source_image_name=source_image_name,
                    source_image_payload=record.source_context.source_image_payload,
                )
                for source_image_name, rows in groups.image_rows.items()
            ),
        )

    def annotate_record_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        object_name: str | None,
        source_image_name: str | None,
    ) -> list[CellProfilerRuntimeValue]:
        """Return rows with row-level ownership declared when policy requires it."""
        if not self.requires_explicit_row_ownership():
            return list(rows)
        return [
            self.annotate_record_row(
                row,
                object_name=object_name,
                source_image_name=source_image_name,
            )
            for row in rows
        ]

    def annotate_record_row(
        self,
        row: CellProfilerRuntimeValue,
        *,
        object_name: str | None,
        source_image_name: str | None,
    ) -> CellProfilerRuntimeValue:
        """Return one row with the semantic owner required by this policy."""
        if self.row_is_object_scoped(row):
            if object_name is None:
                raise ValueError(
                    f"{type(self).__name__} requires an object name for object rows."
                )
            return MeasurementRowOwnership(object_name=object_name).annotate_row(row)
        resolved_source_image_name = self.image_row_source_image_name(source_image_name)
        if resolved_source_image_name is None:
            raise ValueError(
                f"{type(self).__name__} requires a source image name for image rows."
            )
        return MeasurementRowOwnership(
            source_image_name=resolved_source_image_name,
        ).annotate_row(row)

    def image_row_source_image_name(
        self,
        source_image_name: str | None,
    ) -> str | None:
        """Return the source owner for image-scoped rows emitted by this module."""
        return source_image_name

    def row_is_object_scoped(self, row: CellProfilerRuntimeValue) -> bool:
        """Return whether a raw emitted row belongs to the object domain."""
        del row
        return True

    def row_has_measured_object(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        """Return whether a source row should consume an object row identity."""
        return self.row_has_result_payload(
            ObjectMeasurementResultPayloadRequest(
                row_mapping,
                object_id_field,
                tuple(axis_fields),
            )
        )

    def row_has_result_payload(
        self,
        request: ObjectMeasurementResultPayloadRequest,
    ) -> bool:
        """Return whether a row carries result values, not just identity padding."""
        metadata_fields = request.metadata_fields
        return any(
            field_name not in metadata_fields
            and self.measurement_value_is_present(value)
            for field_name, value in request.row_mapping.items()
        )

    def measurement_value_is_present(self, value: CellProfilerRuntimeValue) -> bool:
        """Return whether a measurement cell is an observed value, not padding."""
        return MeasurementScalarLiteral(value).is_present_measurement_value

    def retains_unmeasured_compact_row(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        """Return whether compact row projection should keep an unmeasured row."""
        del row_mapping, object_id_field, axis_fields
        return True

    def required_object_ids_for_axis(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        projected_rows: CellProfilerRuntimeValueSequence,
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_key: CellProfilerRuntimeValues,
    ) -> tuple[int, ...]:
        """Return object row IDs required by this policy for one measurement axis."""
        del projected_rows, object_id_field
        schema = ObjectMeasurementRowCompletionSchema.for_completion_fields(
            object_id_field=MEASUREMENT_OBJECT_LABEL_FIELD,
            axis_fields=axis_fields,
        )
        return schema.object_ids_for_axis(
            label_payload=label_payload,
            object_identity=object_identity,
            axis_key=axis_key,
        )

    def required_object_ids_by_axis(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_keys: Sequence[CellProfilerRuntimeValues],
    ) -> ObjectMeasurementIdsByAxis:
        """Return required object row IDs for every measurement axis."""
        schema = ObjectMeasurementRowCompletionSchema.for_completion_fields(
            object_id_field=object_id_field,
            axis_fields=axis_fields,
        )
        object_ids_by_label_axis: ObjectMeasurementIdsByAxis = {}
        object_ids_by_axis: ObjectMeasurementIdsByAxis = {}
        for axis_key in axis_keys:
            label_axis_key = schema.label_domain_axis_key(
                label_payload=label_payload,
                axis_key=axis_key,
            )
            object_ids = object_ids_by_label_axis.get(label_axis_key)
            if object_ids is None:
                object_ids = self.required_object_ids_for_axis(
                    label_payload=label_payload,
                    projected_rows=projection.rows,
                    object_identity=object_identity,
                    object_id_field=object_id_field,
                    axis_fields=axis_fields,
                    axis_key=axis_key,
                )
                object_ids_by_label_axis[label_axis_key] = object_ids
            object_ids_by_axis[axis_key] = object_ids
        return object_ids_by_axis

    def complete_rows(
        self,
        rows: MeasurementRowsInput,
        *,
        label_payload: CellProfilerRuntimeValue,
        func: CellProfilerFunction,
    ) -> MeasurementRowsInput:
        """Pad per-object measurement rows across this policy's object domain."""
        if isinstance(rows, ColumnarRows):
            rows = tuple(rows.row_mappings())
        schema = self.completion_schema(
            rows,
            func,
            label_payload=label_payload,
        )
        object_identity = self.object_identity_for_label_payload(label_payload)
        completed_rows = self.already_complete_dense_domain_rows(
            rows,
            schema=schema,
            object_identity=object_identity,
            label_payload=label_payload,
        )
        if completed_rows is not None:
            return completed_rows
        projection_request = self.completion_projection_request(rows, schema)
        projection = self.project_completion_rows(
            projection_request,
            object_identity,
            original_row_count=len(rows),
        )
        projected_rows = projection.rows
        if self.projected_rows_have_no_object_ids(projection):
            return list(projected_rows)
        axis_keys = self.completion_axis_keys(
            projection_request,
            projection,
            label_payload=label_payload,
        )
        object_ids_by_axis = self.required_object_ids_by_axis(
            label_payload=label_payload,
            projection=projection,
            object_identity=object_identity,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            axis_keys=axis_keys,
        )
        object_ids = ObjectMeasurementProjectedRowKeys.object_ids_from_axis_domain(
            object_ids_by_axis
        )
        if not object_ids:
            return list(projected_rows)
        bounded_projection = projection.within_axis_domain(
            axis_keys=axis_keys,
            object_ids_by_axis=object_ids_by_axis,
        )
        missing_row_keys = bounded_projection.row_keys.missing_from_axis_domain(
            axis_keys,
            object_ids_by_axis,
        )
        if not missing_row_keys:
            return bounded_projection.ordered_rows(
                object_ids=object_ids,
                axis_keys=axis_keys,
            )
        missing_rows = schema.missing_rows(
            missing_row_keys=missing_row_keys,
            label_payload=label_payload,
            row_policy=self,
            measured_positive_extent_by_axis=(
                bounded_projection.measured_positive_extent_by_axis()
            ),
        )
        return bounded_projection.ordered_rows(
            rows=(*bounded_projection.rows, *missing_rows),
            row_keys=bounded_projection.row_keys.appended(missing_row_keys),
            object_ids=object_ids,
            axis_keys=axis_keys,
        )

    def already_complete_dense_domain_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        schema: "ObjectMeasurementRowCompletionSchema",
        object_identity: MeasurementObjectRowIdentity,
        label_payload: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValueSequence | None:
        """Return rows unchanged when they already exactly cover the dense domain."""
        if object_identity is not MeasurementObjectRowIdentity.LABEL_ID or not rows:
            return None
        projection_request = self.completion_projection_request(rows, schema)
        row_keys: list[tuple[int, CellProfilerRuntimeValues]] = []
        axis_keys: list[CellProfilerRuntimeValues] = []
        for row in rows:
            object_id = projection_request.object_label(row)
            if object_id is None:
                return None
            axis_key = projection_request.axis_key(row)
            row_keys.append((object_id, axis_key))
            if axis_key not in axis_keys:
                axis_keys.append(axis_key)
        object_ids_by_axis = {
            axis_key: schema.object_ids_for_axis(
                label_payload=label_payload,
                object_identity=object_identity,
                axis_key=axis_key,
            )
            for axis_key in axis_keys
        }
        if any(not object_ids for object_ids in object_ids_by_axis.values()):
            return None
        required_row_keys = ObjectMeasurementProjectedRowKeys.required_axis_domain(
            tuple(axis_keys),
            object_ids_by_axis,
        )
        if tuple(row_keys) != tuple(required_row_keys):
            return None
        return list(rows)

    def completion_schema(
        self,
        rows: CellProfilerRuntimeValueSequence,
        func: CellProfilerFunction,
        *,
        label_payload: CellProfilerRuntimeValue,
    ) -> "ObjectMeasurementRowCompletionSchema":
        """Return row-completion schema after policy-specific identity projection."""
        schema = ObjectMeasurementRowCompletionSchema.from_rows(rows, func)
        identity_axis_fields = self.row_identity_axis_fields(
            schema.axis_fields,
            label_payload=label_payload,
        )
        return ObjectMeasurementRowCompletionSchema.for_completion_fields(
            field_names=schema.field_names,
            object_id_field=schema.object_id_field,
            axis_fields=identity_axis_fields,
        )

    def completion_projection_request(
        self,
        rows: CellProfilerRuntimeValueSequence,
        schema: "ObjectMeasurementRowCompletionSchema",
    ) -> "ObjectMeasurementRowIdentityProjectionRequest":
        """Return the identity projection request for object-row completion."""
        return ObjectMeasurementRowIdentityProjectionRequest(
            rows=rows,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            row_policy=self,
        )

    def project_completion_rows(
        self,
        request: "ObjectMeasurementRowIdentityProjectionRequest",
        object_identity: MeasurementObjectRowIdentity,
        *,
        original_row_count: int,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Project emitted rows into this policy's object identity domain."""
        projection = MeasurementObjectRowIdentityProjectionStrategy.for_enum_member(
            object_identity
        ).project_rows(request)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "object_measurement_row_identity_projection",
            0.0,
            policy=type(self).__name__,
            object_identity=object_identity.value,
            rows=original_row_count,
            projected_rows=len(projection.rows),
            axis_count=len(projection.axis_keys),
            max_object_id=projection.row_keys.max_object_id_or_none(),
        )
        return projection

    @staticmethod
    def projected_rows_have_no_object_ids(
        projection: "ObjectMeasurementRowIdentityProjectionResult",
    ) -> bool:
        return bool(projection.rows) and not projection.row_keys.has_object_ids()

    def completion_axis_keys(
        self,
        request: "ObjectMeasurementRowIdentityProjectionRequest",
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        *,
        label_payload: CellProfilerRuntimeValue,
    ) -> tuple[CellProfilerRuntimeValues, ...]:
        """Return measurement axis keys used to complete missing object rows."""
        axis_keys = request.axis_keys_for_label_payload(
            projection,
            label_payload=label_payload,
        )
        if axis_keys:
            return axis_keys
        return ((),)

    def missing_measurement_value(
        self,
        *,
        object_id: int,
        label_payload: CellProfilerRuntimeValue,
        field_name: str,
        positive_label_extent: int | None = None,
    ) -> float:
        """Return the value to use for a missing object measurement field."""
        value_policy = MissingObjectMeasurementValuePolicy(type(self).missing_value_policy)
        strategy = MissingObjectMeasurementValueStrategy.for_enum_member(value_policy)
        return strategy.missing_value(
            MissingObjectMeasurementValueRequest(
                object_id=object_id,
                label_payload=label_payload,
                field_name=field_name,
                positive_label_extent=positive_label_extent,
            )
        )

class DeclaredObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Generated base for modules with declared measurement-row identity."""


class CompactObjectMeasurementRowIdentityPolicy(DeclaredObjectMeasurementRowPolicy):
    """Use CP's compact row identity for emitted measurement rows."""

    row_identity = MeasurementObjectRowIdentity.ROW_ORDINAL


class CompactMeasuredObjectMeasurementRowPolicy(CompactObjectMeasurementRowIdentityPolicy):
    """Complete rows against the compact ordinal domain emitted by CP."""

    def missing_measurement_value(
        self,
        *,
        object_id: int,
        label_payload: CellProfilerRuntimeValue,
        field_name: str,
        positive_label_extent: int | None = None,
    ) -> MissingObjectMeasurementCellValue:
        if type(self) is CompactMeasuredObjectMeasurementRowPolicy:
            declared_domain = dense_object_label_id_domain(label_payload)
            if declared_domain and object_id < len(declared_domain):
                return MISSING_MEASUREMENT_ROW_VALUE
        return super().missing_measurement_value(
            object_id=object_id,
            label_payload=label_payload,
            field_name=field_name,
            positive_label_extent=positive_label_extent,
        )

    def required_object_ids_for_axis(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        projected_rows: CellProfilerRuntimeValueSequence,
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_key: CellProfilerRuntimeValues,
    ) -> tuple[int, ...]:
        """Compact CP rows are dense over emitted row ordinals."""
        del label_payload, object_identity
        projection_request = ObjectMeasurementRowIdentityProjectionRequest(
            rows=projected_rows,
            object_id_field=object_id_field,
            axis_fields=axis_fields,
            row_policy=self,
        )
        object_ids = tuple(
            object_id
            for row in projected_rows
            if projection_request.axis_key(row) == axis_key
            for object_id in (projection_request.object_label(row),)
            if object_id is not None
        )
        if not object_ids:
            return ()
        return tuple(range(1, max(object_ids) + 1))

    def required_object_ids_by_axis(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_keys: Sequence[CellProfilerRuntimeValues],
    ) -> ObjectMeasurementIdsByAxis:
        """Compact CP rows are dense over emitted row ordinals per axis."""
        schema = ObjectMeasurementRowCompletionSchema.for_completion_fields(
            object_id_field=object_id_field,
            axis_fields=axis_fields,
        )
        declared_ids_by_axis = {
            axis_key: schema.object_ids_for_axis(
                label_payload=label_payload,
                object_identity=object_identity,
                axis_key=axis_key,
            )
            for axis_key in axis_keys
        }
        max_object_id_by_axis = {axis_key: 0 for axis_key in axis_keys}
        for axis_key, declared_ids in declared_ids_by_axis.items():
            if declared_ids:
                max_object_id_by_axis[axis_key] = max(declared_ids)
        for object_id, axis_key in projection.row_keys:
            if object_id is None or axis_key not in max_object_id_by_axis:
                continue
            max_object_id_by_axis[axis_key] = max(
                max_object_id_by_axis[axis_key],
                object_id,
            )
        return {
            axis_key: tuple(range(1, max_object_id + 1))
            for axis_key, max_object_id in max_object_id_by_axis.items()
        }


class DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy(
    CompactObjectMeasurementRowIdentityPolicy
):
    """Compact measured rows, then pad to the declared object-label domain."""


class FeatureAnchoredCompactObjectMeasurementRowPolicy(
    CompactMeasuredObjectMeasurementRowPolicy
):
    """Compact rows whose measuredness is anchored by declared feature fields."""

    measured_object_features: ClassVar[tuple[ObjectShapeMeasurementFeature, ...]] = ()

    def row_has_measured_object(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        del object_id_field, axis_fields
        return any(
            self.measurement_value_is_present(row_mapping.get(feature.value))
            for feature in type(self).measured_object_features
        )


class DenseEmittedObjectMeasurementRowsMixin:
    """Policy mixin for modules that already emit their complete object row domain."""

    def complete_rows(
        self,
        rows: MeasurementRowsInput,
        *,
        label_payload: CellProfilerRuntimeValue,
        func: CellProfilerFunction,
    ) -> MeasurementRowsInput:
        if isinstance(rows, ColumnarRows):
            if rows.covers_declared_object_measurement_domain:
                return rows
            return super().complete_rows(
                rows,
                label_payload=label_payload,
                func=func,
            )
        return list(rows)


class DenseColumnarObjectMeasurementRowsMixin:
    """Policy mixin for modules that emit complete rows only through carriers."""

    def complete_rows(
        self,
        rows: MeasurementRowsInput,
        *,
        label_payload: CellProfilerRuntimeValue,
        func: CellProfilerFunction,
    ) -> MeasurementRowsInput:
        if (
            isinstance(rows, ColumnarRows)
            and rows.covers_declared_object_measurement_domain
        ):
            return rows
        return super().complete_rows(
            rows,
            label_payload=label_payload,
            func=func,
        )


class MeasureObjectSizeShapeObjectMeasurementRowPolicy(
    FeatureAnchoredCompactObjectMeasurementRowPolicy
):
    """Object shape rows are object-qualified, not image-source-qualified."""

    module_name = _MEASURE_OBJECT_SIZE_SHAPE_MODULE
    measured_object_features = (
        ObjectShapeMeasurementFeature.AREA,
        ObjectShapeMeasurementFeature.CENTER_X,
        ObjectShapeMeasurementFeature.CENTER_Y,
    )

    def table_source_image_name(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        source_image_name: str | None,
    ) -> str | None:
        del measurement_images, source_image_name
        return None


for _metadata_row_policy_spec in (
    GeneratedLeafClassSpec(
        "DefaultObjectMeasurementRowPolicy",
        CellProfilerObjectMeasurementRowPolicy,
        attributes={
            "registry_key": CellProfilerModulePolicyRegistryKey.DEFAULT.value,
        },
    ),
    CellProfilerModulePolicyMultiBaseLeafSpec(
        class_name="MeasureObjectIntensityDistributionObjectMeasurementRowPolicy",
        base_type=CompactMeasuredObjectMeasurementRowPolicy,
        base_types=(
            DenseEmittedObjectMeasurementRowsMixin,
            CompactMeasuredObjectMeasurementRowPolicy,
        ),
        module_name=_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
    ),
    CellProfilerModulePolicyMultiBaseLeafSpec(
        class_name="MeasureGranularityObjectMeasurementRowPolicy",
        base_type=CellProfilerObjectMeasurementRowPolicy,
        base_types=(
            DenseEmittedObjectMeasurementRowsMixin,
            CellProfilerObjectMeasurementRowPolicy,
        ),
        module_name=_MEASURE_GRANULARITY_MODULE,
    ),
):
    _metadata_row_policy_spec.declare_in(globals())


class MeasureTextureObjectMeasurementRowPolicy(
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy
):
    """Use direct texture rows when CP already emitted the declared dense domain."""

    module_name = _MEASURE_TEXTURE_MODULE
    row_identity = MeasurementObjectRowIdentity.ROW_SEQUENCE
    row_sequence_axis_fields: ClassVar[frozenset[str]] = frozenset(
        (
            MeasurementRowAxisField.SCALE.value,
            MeasurementRowAxisField.DIRECTION.value,
            MeasurementRowAxisField.GRAY_LEVELS.value,
        )
    )

    def row_identity_axis_fields(
        self,
        axis_fields: Sequence[str],
        *,
        label_payload: CellProfilerRuntimeValue | None = None,
    ) -> tuple[str, ...]:
        """Texture compact rows are sequenced by feature axis, not source/slice axes."""
        if not MeasureTextureMissingValueDomain.from_payload(
            label_payload
        ).is_multi_source_plane_domain():
            return tuple(axis_fields)
        return tuple(
            field_name
            for field_name in axis_fields
            if field_name in type(self).row_sequence_axis_fields
        )

    def object_identity_for_label_payload(
        self,
        label_payload: CellProfilerRuntimeValue,
    ) -> MeasurementObjectRowIdentity:
        """Use row-sequence identity only for multi-source plane-domain texture rows."""
        if MeasureTextureMissingValueDomain.from_payload(
            label_payload
        ).is_multi_source_plane_domain():
            return MeasurementObjectRowIdentity.ROW_SEQUENCE
        return MeasurementObjectRowIdentity.ROW_ORDINAL

    def missing_measurement_value(
        self,
        *,
        object_id: int,
        label_payload: CellProfilerRuntimeValue,
        field_name: str,
        positive_label_extent: int | None = None,
    ) -> float:
        missing_domain = MeasureTextureMissingValueDomain.from_payload(label_payload)
        value_policy = missing_domain.missing_value_policy(type(self).missing_value_policy)
        if positive_label_extent is None:
            positive_label_extent = missing_domain.compact_row_ordinal_positive_extent()
        strategy = MissingObjectMeasurementValueStrategy.for_enum_member(value_policy)
        return strategy.missing_value(
            MissingObjectMeasurementValueRequest(
                object_id=object_id,
                label_payload=label_payload,
                field_name=field_name,
                positive_label_extent=positive_label_extent,
            )
        )

    def complete_rows(
        self,
        rows: MeasurementRowsInput,
        *,
        label_payload: CellProfilerRuntimeValue,
        func: CellProfilerFunction,
    ) -> MeasurementRowsInput:
        """Avoid padding work only when emitted rows already match the domain."""
        if isinstance(rows, ColumnarRows):
            return rows
        missing_domain = MeasureTextureMissingValueDomain.from_payload(label_payload)
        schema = ObjectMeasurementRowCompletionSchema.from_rows(rows, func)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "measure_texture_complete_rows",
            0.0,
            rows=len(rows),
            axis_fields=schema.axis_fields,
            object_id_field=schema.object_id_field,
            multi_source_domain=missing_domain.is_multi_source_plane_domain(),
            label_payload_type=type(label_payload).__name__,
            label_shape=RuntimeShapeInspection(
                np.asarray(LABEL_PAYLOAD_FINAL.value(label_payload))
            ).shape_tuple(),
        )
        if (
            rows
            and unwrap(func) is unwrap(measure_texture_objects)
            and not missing_domain.is_multi_source_plane_domain()
        ):
            return list(rows)
        if not schema.axis_fields and rows:
            required_object_ids = schema.object_ids_for_axis(
                label_payload=label_payload,
                object_identity=self.object_identity(),
                axis_key=(),
            )
            emitted_object_ids = tuple(
                int(measurement_row_mapping(row)[schema.object_id_field])
                for row in rows
                if self.row_is_object_scoped(row)
            )
            if emitted_object_ids == required_object_ids:
                return missing_domain.normalize_existing_rows(
                    rows,
                    field_names=schema.field_names,
                    object_id_field=schema.object_id_field,
                    axis_fields=schema.axis_fields,
                )
        if schema.axis_fields and rows:
            complete_axis_rows = TextureAxisMeasurementRows.from_rows(
                rows,
                schema=schema,
                label_payload=label_payload,
                row_policy=self,
            )
            if complete_axis_rows.already_complete:
                return missing_domain.normalize_existing_rows(
                    rows,
                    field_names=schema.field_names,
                    object_id_field=schema.object_id_field,
                    axis_fields=schema.axis_fields,
                )
        completed_rows = super().complete_rows(
            rows,
            label_payload=label_payload,
            func=func,
        )
        if isinstance(completed_rows, ColumnarRows):
            return completed_rows
        return missing_domain.normalize_existing_rows(
            completed_rows,
            field_names=schema.field_names,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
        )


@dataclass(frozen=True, slots=True)
class TextureAxisMeasurementRows:
    """Observed MeasureTexture row coverage by measurement axis."""

    emitted_object_ids_by_axis: ObjectMeasurementIdsByAxisView
    required_object_ids_by_axis: ObjectMeasurementIdsByAxisView

    @classmethod
    def from_rows(
        cls,
        rows: CellProfilerRuntimeValueSequence,
        *,
        schema: ObjectMeasurementRowCompletionSchema,
        label_payload: CellProfilerRuntimeValue,
        row_policy: MeasureTextureObjectMeasurementRowPolicy,
    ) -> "TextureAxisMeasurementRows":
        emitted_ids: dict[CellProfilerRuntimeValues, list[int]] = {}
        projection_request = ObjectMeasurementRowIdentityProjectionRequest(
            rows=rows,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            row_policy=row_policy,
        )
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            object_id = measurement_object_label(
                row_mapping,
                object_id_field=schema.object_id_field,
            )
            if object_id is None:
                continue
            axis_key = projection_request.axis_key_from_mapping(row_mapping)
            if axis_key not in emitted_ids:
                emitted_ids[axis_key] = []
            emitted_ids[axis_key].append(int(object_id))
        axis_keys = tuple(emitted_ids)
        required_ids_by_axis = row_policy.required_object_ids_by_axis(
            label_payload=label_payload,
            projection=ObjectMeasurementRowIdentityProjectionResult(
                rows=tuple(rows),
                row_keys=ObjectMeasurementProjectedRowKeys(
                    tuple(
                        (object_id, axis_key)
                        for axis_key, object_ids in emitted_ids.items()
                        for object_id in object_ids
                    )
                ),
                measured_row_keys=ObjectMeasurementProjectedRowKeys(
                    tuple(
                        (object_id, axis_key)
                        for axis_key, object_ids in emitted_ids.items()
                        for object_id in object_ids
                    )
                ),
                axis_keys=axis_keys,
            ),
            object_identity=row_policy.object_identity_for_label_payload(label_payload),
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            axis_keys=axis_keys,
        )
        return cls(
            emitted_object_ids_by_axis={
                axis_key: tuple(object_ids)
                for axis_key, object_ids in emitted_ids.items()
            },
            required_object_ids_by_axis=required_ids_by_axis,
        )

    @property
    def already_complete(self) -> bool:
        if not self.emitted_object_ids_by_axis:
            return False
        if self.emitted_object_ids_by_axis.keys() != self.required_object_ids_by_axis.keys():
            return False
        for axis_key, object_ids in self.emitted_object_ids_by_axis.items():
            required_object_ids = self.required_object_ids_by_axis[axis_key]
            if len(object_ids) != len(required_object_ids):
                return False
            if frozenset(object_ids) != frozenset(required_object_ids):
                return False
        return True


@dataclass(frozen=True, slots=True)
class MeasureTextureMissingValueDomain:
    """Resolve texture padding semantics from the object-label measurement domain."""

    label_payload: ObjectLabelValue | None

    @classmethod
    def from_payload(
        cls,
        label_payload: CellProfilerRuntimeValue | None,
    ) -> "MeasureTextureMissingValueDomain":
        """Return texture domain semantics only for object-label payloads."""
        if isinstance(label_payload, ObjectLabelValue):
            return cls(label_payload)
        return cls(None)

    def missing_value_policy(
        self,
        default_policy: MissingObjectMeasurementValuePolicy,
    ) -> MissingObjectMeasurementValuePolicy:
        if self.is_multi_source_plane_domain():
            return MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
        return default_policy

    def is_multi_source_plane_domain(self) -> bool:
        payload = self.label_payload
        if payload is None:
            return False
        if payload.domain.scope is not ObjectLabelDomainScope.PLANE:
            return False
        if payload.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return False
        if len(payload.source_image_names) <= 1:
            return False
        labels = np.asarray(LABEL_PAYLOAD_FINAL.value(payload))
        return labels.ndim >= 4

    def normalize_existing_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        field_names: Sequence[str],
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> list[CellProfilerRuntimeValue]:
        if not self.is_multi_source_plane_domain():
            return list(rows)
        extent = self.compact_row_ordinal_positive_extent()
        if extent is None:
            return list(rows)
        extent = max(extent, self.compact_row_ordinal_extent_from_rows(rows))
        identity_fields = {
            object_id_field,
            *MEASUREMENT_OBJECT_ID_FIELDS,
            *axis_fields,
            MEASUREMENT_OBJECT_NAME_FIELD,
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
        }
        measurement_fields = tuple(
            field_name for field_name in field_names if field_name not in identity_fields
        )
        return MeasureTextureExistingRowsNormalizer(
            rows=rows,
            extent=extent,
            field_names=tuple(field_names),
            identity_fields=frozenset(identity_fields),
            first_measurement_field=FirstMeasurementField(
                measurement_fields
            ).value_or_none(),
        ).normalized_rows()

    def compact_row_ordinal_positive_extent(self) -> int | None:
        """Return the declared compact-row extent for multi-source texture labels."""
        if not self.is_multi_source_plane_domain():
            return None
        if not isinstance(self.label_payload, ObjectLabelValue):
            raise TypeError(
                "MeasureTexture row-ordinal extent requires ObjectLabelValue, got "
                f"{type(self.label_payload).__name__}."
            )
        domain = self.label_payload.object_label_domain()
        if domain.declared_object_id_domains:
            return max(len(object_ids) for object_ids in domain.declared_object_id_domains)
        explicit_domain = domain.explicit_id_domain()
        if explicit_domain is not None:
            return len(explicit_domain)
        return None

    @staticmethod
    def compact_row_ordinal_extent_from_rows(rows: CellProfilerRuntimeValueSequence) -> int:
        """Return the largest compact row ordinal already emitted by texture rows."""
        object_ids = tuple(
            object_id
            for row in rows
            for object_id in (measurement_object_label(measurement_row_mapping(row)),)
            if object_id is not None
        )
        if not object_ids:
            return 0
        return max(object_ids)


@dataclass(frozen=True, slots=True)
class FirstMeasurementField:
    """First non-identity measurement field available for row diagnostics."""

    measurement_fields: tuple[str, ...]

    def value_or_none(self) -> str | None:
        match self.measurement_fields:
            case (field_name, *_):
                return field_name
            case _:
                return None


@dataclass(slots=True)
class MeasureTextureExistingRowsNormalizer:
    """Normalize existing MeasureTexture rows for compact multi-source domains."""

    rows: CellProfilerRuntimeValueSequence
    extent: int
    field_names: tuple[str, ...]
    identity_fields: frozenset[str]
    first_measurement_field: str | None
    first_field_value_types: Counter[str] = field(default_factory=Counter)
    first_field_sample_values: list[str] = field(default_factory=list)
    nan_replacements: int = 0
    none_replacements: int = 0
    absent_replacements: int = 0

    def normalized_rows(self) -> list[CellProfilerRuntimeValue]:
        normalized_rows = [self.normalized_row(row) for row in self.rows]
        self.log_profile(len(normalized_rows))
        return normalized_rows

    def normalized_row(self, row: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        row_mapping = measurement_row_mapping(row)
        object_id = measurement_object_label(row_mapping)
        if object_id is None or object_id > self.extent:
            return row
        normalized_row = dict(row_mapping)
        normalized_row[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = (
            MeasurementObjectRowIdentity.ROW_SEQUENCE.value
        )
        self.record_first_field_sample(row_mapping)
        self.replace_existing_measurements(row_mapping, normalized_row)
        self.add_absent_measurements(normalized_row)
        return normalized_row

    def record_first_field_sample(self, row_mapping: MeasurementRowMapping) -> None:
        field_name = self.first_measurement_field
        if field_name is None:
            return
        first_value = MappingValueLookup(row_mapping, field_name).value_or("<ABSENT>")
        self.first_field_value_types[type(first_value).__name__] += 1
        if len(self.first_field_sample_values) < 6:
            self.first_field_sample_values.append(repr(first_value))

    def replace_existing_measurements(
        self,
        row_mapping: MeasurementRowMapping,
        normalized_row: CellProfilerKwargDict,
    ) -> None:
        for field_name, value in row_mapping.items():
            if field_name in self.identity_fields:
                continue
            self.replace_existing_measurement(field_name, value, normalized_row)

    def replace_existing_measurement(
        self,
        field_name: str,
        value: CellProfilerRuntimeValue,
        normalized_row: CellProfilerKwargDict,
    ) -> None:
        if value is None:
            normalized_row[field_name] = 0.0
            self.none_replacements += 1
            return
        if MeasurementScalarLiteral(value).is_padding_measurement_value:
            normalized_row[field_name] = 0.0
            self.nan_replacements += 1

    def add_absent_measurements(self, normalized_row: CellProfilerKwargDict) -> None:
        for field_name in self.field_names:
            if field_name in self.identity_fields or field_name in normalized_row:
                continue
            normalized_row[field_name] = 0.0
            self.absent_replacements += 1

    def log_profile(self, row_count: int) -> None:
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "measure_texture_normalize_rows",
            0.0,
            rows=row_count,
            extent=self.extent,
            field_count=len(self.field_names),
            first_measurement_field=self.first_measurement_field,
            first_field_value_types=dict(self.first_field_value_types),
            first_field_sample_values=tuple(self.first_field_sample_values),
            nan_replacements=self.nan_replacements,
            none_replacements=self.none_replacements,
            absent_replacements=self.absent_replacements,
        )


class CellProfilerMeasurementRowIdentityField(str, Enum):
    """CellProfiler row fields that carry ownership/axis identity, not values."""

    SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value
    OBJECT_LABEL = MEASUREMENT_OBJECT_LABEL_FIELD
    OBJECT_NAME = MEASUREMENT_OBJECT_NAME_FIELD
    SOURCE_IMAGE_NAME = MEASUREMENT_SOURCE_IMAGE_NAME_FIELD


@dataclass(frozen=True, slots=True)
class SourceImagePairCollection:
    """Cardinality authority for source-image pairs."""

    source_pairs: tuple[CellProfilerSourceImagePair, ...]

    def single_source_image_name(self) -> str | None:
        match self.source_pairs:
            case (source_pair,):
                return source_pair.source_image_name
            case _:
                return None


class MeasureColocalizationObjectMeasurementRowPolicy(
    CellProfilerObjectMeasurementRowPolicy
):
    """Expand composed source stacks into source-pair object measurements."""

    module_name = _MEASURE_COLOCALIZATION_MODULE
    identity_fields: ClassVar[frozenset[str]] = frozenset(
        field.value for field in CellProfilerMeasurementRowIdentityField
    )

    def invocations(
        self,
        measurement_image: CellProfilerMeasurementImage,
        kwargs: CellProfilerKwargs,
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        source_pairs = measurement_image.source_image_pairs()
        if not source_pairs:
            return super().invocations(measurement_image, kwargs)
        return tuple(
            SourcePairObjectMeasurementInvocation(
                kwargs={
                    **kwargs,
                    **source_pair.invocation_kwargs(
                        first_channel_kwarg="channel_1",
                        second_channel_kwarg="channel_2",
                    ),
                },
                source_pair=source_pair,
            )
            for source_pair in source_pairs
        )

    def project_rows(
        self,
        rows: MeasurementRowsInput,
        invocation: ObjectMeasurementInvocation,
    ) -> MeasurementRowsInput:
        if invocation.source_pair is None:
            return rows if isinstance(rows, ColumnarRows) else list(rows)
        if isinstance(rows, ColumnarRows):
            return CellProfilerSourcePairFeature.project_columnar_rows_for_pair(
                rows,
                invocation.source_pair,
                retain_field=type(self).identity_fields.__contains__,
            )
        return [self.project_row(row, invocation.source_pair) for row in rows]

    def project_row(
        self,
        row: CellProfilerRuntimeValue,
        source_pair: CellProfilerSourceImagePair,
    ) -> CellProfilerKwargDict:
        """Return one row with CellProfiler source-pair feature names."""
        row_mapping = measurement_row_mapping(row)
        identity_fields = type(self).identity_fields
        return CellProfilerSourcePairFeature.project_row_for_pair(
            row_mapping,
            source_pair,
            retain_field=identity_fields.__contains__,
        )

    def table_source_image_name(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        source_image_name: str | None,
    ) -> str | None:
        del source_image_name
        source_pairs = tuple(
            source_pair
            for measurement_image in measurement_images
            for source_pair in measurement_image.source_image_pairs()
        )
        return SourceImagePairCollection(source_pairs).single_source_image_name()


class TrackObjectsObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """TrackObjects emits object rows and image-level tracking counts together."""

    module_name = _TRACK_OBJECTS_MODULE
    explicit_row_ownership_required = True

    def row_is_object_scoped(self, row: CellProfilerRuntimeValue) -> bool:
        row_mapping = measurement_row_mapping(row)
        return measurement_row_has_object_identity(row_mapping)

    def image_row_source_image_name(
        self,
        source_image_name: str | None,
    ) -> str | None:
        del source_image_name
        return MeasurementScope.IMAGE.value


for _row_policy_spec in (
    CellProfilerModulePolicyMultiBaseLeafSpec(
        class_name=f"{_MEASURE_OBJECT_INTENSITY_MODULE}ObjectMeasurementRowPolicy",
        base_type=DeclaredObjectMeasurementRowPolicy,
        base_types=(
            DenseColumnarObjectMeasurementRowsMixin,
            DeclaredObjectMeasurementRowPolicy,
        ),
        module_name=_MEASURE_OBJECT_INTENSITY_MODULE,
        attributes={
            "missing_value_policy": (
                MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
            ),
        },
    ),
):
    _row_policy_spec.declare_in(globals())
