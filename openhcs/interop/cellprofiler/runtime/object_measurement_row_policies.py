"""CellProfiler object-measurement row ownership and completion policies."""

from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import ClassVar
from metaclass_registry import RegistryFamily, RegistryKeyAttribute
from openhcs.core.artifacts import ArtifactSpecCollection
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
    MeasurementRowOwnership,
    measurement_object_label,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
    MeasurementScalarLiteral,
    dense_object_label_id_domain,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    ImagePayloadMetadataCompositionMode,
    MeasurementTable,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerMeasurementImage,
    CellProfilerSourceImagePair,
)
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementRecord,
)
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_row_source_names_required,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    MeasurementObjectRowIdentityProjectionStrategy,
    MissingObjectMeasurementValuePolicy,
    MissingObjectMeasurementValueRequest,
    MissingObjectMeasurementValueStrategy,
    ObjectMeasurementIdsByAxis,
    ObjectMeasurementProjectedRowKeys,
    ObjectMeasurementRowCompletionSchema,
    ObjectMeasurementRowIdentityProjectionRequest,
    ObjectMeasurementRowIdentityProjectionResult,
    ObjectMeasurementRowOrdinalProjectionState,
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
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryLookup,
    CellProfilerModulePolicyRegistryKey,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)


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
                *MeasurementRowAxisField.object_id_field_names(),
                *self.axis_fields,
                *MeasurementRowAxisField.object_ownership_field_names(),
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
        cls, record: CellProfilerMeasurementRecord
    ) -> "ExplicitMeasurementRowOwnershipGroups":
        object_rows: ObjectMeasurementRowsByName = {}
        image_rows: ObjectMeasurementRowsByName = {}
        unowned_rows: list[CellProfilerRuntimeValue] = []
        rows = (
            record.rows.iter_row_mappings()
            if isinstance(record.rows, ColumnarRows)
            else record.rows
        )
        for row in rows:
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
    measurement_record_excluded_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    @lru_cache(maxsize=None)
    def for_module(
        cls, module_name: str
    ) -> CellProfilerObjectMeasurementRowPolicy | None:
        """Return the row policy carried by the module declaration."""
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

        module_type = CellProfilerModule.for_module(module_name)
        if module_type is not None and issubclass(module_type, cls):
            return module_type()
        policy_type = CellProfilerModulePolicyRegistryLookup(
            cls.__registry__, module_name, cls.fallback_registry_key
        ).policy_type_or_none()
        if policy_type is None:
            return None
        return policy_type()

    def object_identity(self) -> MeasurementObjectRowIdentity:
        """Return the object identity projection for rows emitted by this module."""
        return MeasurementObjectRowIdentity(type(self).row_identity)

    def object_identity_for_label_payload(
        self, label_payload: CellProfilerRuntimeValue
    ) -> MeasurementObjectRowIdentity:
        """Return row identity for a concrete object-measurement label domain."""
        del label_payload
        return self.object_identity()

    def object_identity_for_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        label_payload: CellProfilerRuntimeValue,
    ) -> MeasurementObjectRowIdentity:
        """Return row identity for the concrete rows emitted by a module."""
        explicit_identity = self.explicit_object_identity_for_rows(rows)
        if explicit_identity is not None:
            return explicit_identity
        return self.object_identity_for_label_payload(label_payload)

    @staticmethod
    def explicit_object_identity_for_rows(
        rows: CellProfilerRuntimeValueSequence,
    ) -> MeasurementObjectRowIdentity | None:
        """Return the uniform explicit object-row identity declared by rows."""
        field_name = MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value
        if isinstance(rows, ColumnarRows):
            if field_name not in rows.columns:
                return None
            identity_values = rows.columns[field_name]
            identities = tuple(
                dict.fromkeys(
                    MeasurementObjectRowIdentity(value) for value in identity_values
                )
            )
        else:
            identities = tuple(
                dict.fromkeys(
                    MeasurementObjectRowIdentity(row_mapping[field_name])
                    for row in rows
                    for row_mapping in (measurement_row_mapping(row),)
                    if field_name in row_mapping
                )
            )
        if not identities:
            return None
        if len(identities) != 1:
            raise ValueError(
                "Object measurement rows declare multiple object-row identities: "
                f"{tuple(identity.value for identity in identities)!r}."
            )
        return identities[0]

    def annotates_projected_object_identity(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        object_identity: MeasurementObjectRowIdentity,
    ) -> bool:
        """Return whether exported rows need an explicit object-identity marker."""
        del row_mapping
        return object_identity is not MeasurementObjectRowIdentity.LABEL_ID

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
        self, rows: MeasurementRowsInput, invocation: ObjectMeasurementInvocation
    ) -> MeasurementRowsInput:
        """Return emitted rows projected into this module's feature namespace."""
        del invocation
        if isinstance(rows, ColumnarRows):
            return rows
        return list(rows)

    def split_scoped_rows(
        self, rows: MeasurementRowsInput
    ) -> tuple[MeasurementRowsInput, CellProfilerRuntimeValueSequence]:
        """Partition object-scoped measurement rows from image-scoped rows."""
        if isinstance(rows, ColumnarRows):
            return (rows, ())
        object_rows: list[CellProfilerRuntimeValue] = []
        non_object_rows: list[CellProfilerRuntimeValue] = []
        for row in rows:
            if self.row_is_object_scoped(row):
                object_rows.append(row)
            else:
                non_object_rows.append(row)
        return (object_rows, non_object_rows)

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
        self, measurement_images: tuple[CellProfilerMeasurementImage, ...]
    ) -> ImagePayloadMetadataCompositionMode | None:
        """Return source metadata topology for this policy's measurement rows."""
        if any(
            (
                self.row_source_owner(measurement_image, measurement_images) is not None
                for measurement_image in measurement_images
            )
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
                measurement_image, measurement_images
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
        ):
            return None
        return object_inputs[0].name

    def requires_explicit_row_ownership(self) -> bool:
        """Return whether emitted rows carry mixed measurement ownership."""
        return type(self).explicit_row_ownership_required

    def record_partitions(
        self, record: CellProfilerMeasurementRecord
    ) -> tuple[CellProfilerMeasurementRecord, ...]:
        """Return single-owner measurement table partitions for a record."""
        if self.requires_explicit_row_ownership():
            return self.explicit_owner_record_partitions(
                record, require_declared_ownership=True
            )
        if self.record_rows_declare_ownership(record):
            return self.explicit_owner_record_partitions(
                record, require_declared_ownership=False
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
    def record_rows_declare_ownership(record: CellProfilerMeasurementRecord) -> bool:
        """Return whether row-level object/source ownership is present."""
        if isinstance(record.rows, ColumnarRows):
            columns = tuple((str(column) for column in record.rows.columns))
            return any(
                field_name in columns
                for field_name in MeasurementRowAxisField.object_ownership_field_names()
            )
        return any(
            (
                measurement_row_object_name(row_mapping) is not None
                or measurement_row_source_image_name(row_mapping) is not None
                for row in record.rows
                for row_mapping in (measurement_row_mapping(row),)
            )
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
                f"{type(self).__name__} requires every mixed-scope measurement row to declare object or source-image ownership."
            )
        return (
            record.with_ownership(
                rows=rows,
                object_name=record.object_name,
                source_image_name=record.source_context.source_image_name,
                source_image_payload=record.source_context.source_image_payload,
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
                f"{type(self).__name__} requires one table-level object owner for mixed measurement rows, got {tuple(groups.object_rows)}."
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
                    source_image_name=record.source_context.source_image_name,
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
                row, object_name=object_name, source_image_name=source_image_name
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
            source_image_name=resolved_source_image_name
        ).annotate_row(row)

    def image_row_source_image_name(self, source_image_name: str | None) -> str | None:
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
                row_mapping, object_id_field, tuple(axis_fields)
            )
        )

    def row_has_result_payload(
        self, request: ObjectMeasurementResultPayloadRequest
    ) -> bool:
        """Return whether a row carries result values, not just identity padding."""
        metadata_fields = request.metadata_fields
        return any(
            (
                field_name not in metadata_fields
                and self.measurement_value_is_present(value)
                for field_name, value in request.row_mapping.items()
            )
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
            object_id_field=MeasurementRowAxisField.OBJECT_LABEL.value,
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
            object_id_field=object_id_field, axis_fields=axis_fields
        )
        object_ids_by_label_axis: ObjectMeasurementIdsByAxis = {}
        object_ids_by_axis: ObjectMeasurementIdsByAxis = {}
        for axis_key in axis_keys:
            label_axis_key = schema.label_domain_axis_key(
                label_payload=label_payload, axis_key=axis_key
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
        schema = self.completion_schema(rows, func, label_payload=label_payload)
        object_identity = self.object_identity_for_rows(
            rows, label_payload=label_payload
        )
        completed_rows = self.already_complete_dense_domain_rows(
            rows,
            schema=schema,
            object_identity=object_identity,
            label_payload=label_payload,
        )
        if completed_rows is not None:
            return self.complete_object_domain_rows(completed_rows)
        projection_request = self.completion_projection_request(rows, schema)
        projection = self.project_completion_rows(
            projection_request,
            object_identity,
            label_payload=label_payload,
            original_row_count=len(rows),
        )
        projected_rows = projection.rows
        if self.projected_rows_have_no_object_ids(projection):
            return list(projected_rows)
        axis_keys = self.completion_axis_keys(
            projection_request, projection, label_payload=label_payload
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
            axis_keys=axis_keys, object_ids_by_axis=object_ids_by_axis
        )
        missing_row_keys = bounded_projection.row_keys.missing_from_axis_domain(
            axis_keys, object_ids_by_axis
        )
        if not missing_row_keys:
            return self.complete_object_domain_rows(
                bounded_projection.ordered_rows(
                    object_ids=object_ids, axis_keys=axis_keys
                )
            )
        missing_rows = schema.missing_rows(
            missing_row_keys=missing_row_keys,
            label_payload=label_payload,
            row_policy=self,
            measured_positive_extent_by_axis=bounded_projection.measured_positive_extent_by_axis(),
        )
        return self.complete_object_domain_rows(
            bounded_projection.ordered_rows(
                rows=(*bounded_projection.rows, *missing_rows),
                row_keys=bounded_projection.row_keys.appended(missing_row_keys),
                object_ids=object_ids,
                axis_keys=axis_keys,
            )
        )

    def complete_object_domain_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
    ) -> MeasurementSparseColumnarRows:
        """Return rows marked as covering this policy's declared object domain."""
        return MeasurementSparseColumnarRows.from_rows(
            rows,
            declared_object_measurement_domain_covered=True,
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
        if any((not object_ids for object_ids in object_ids_by_axis.values())):
            return None
        required_row_keys = ObjectMeasurementProjectedRowKeys.required_axis_domain(
            tuple(axis_keys), object_ids_by_axis
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
            schema.axis_fields, label_payload=label_payload
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
        label_payload: CellProfilerRuntimeValue,
        original_row_count: int,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Project emitted rows into this policy's object identity domain."""
        del label_payload
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
        return bool(projection.rows) and (not projection.row_keys.has_object_ids())

    def completion_axis_keys(
        self,
        request: "ObjectMeasurementRowIdentityProjectionRequest",
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        *,
        label_payload: CellProfilerRuntimeValue,
    ) -> tuple[CellProfilerRuntimeValues, ...]:
        """Return measurement axis keys used to complete missing object rows."""
        axis_keys = request.axis_keys_for_label_payload(
            projection, label_payload=label_payload
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
        value_policy = MissingObjectMeasurementValuePolicy(
            type(self).missing_value_policy
        )
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


class CompactMeasuredObjectMeasurementRowPolicy(
    CompactObjectMeasurementRowIdentityPolicy
):
    """Complete rows against the compact ordinal domain emitted by CP."""

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
            (
                object_id
                for row in projected_rows
                if projection_request.axis_key(row) == axis_key
                for object_id in (projection_request.object_label(row),)
                if object_id is not None
            )
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
            object_id_field=object_id_field, axis_fields=axis_fields
        )
        declared_ids_by_axis = {
            axis_key: schema.object_ids_for_axis(
                label_payload=label_payload,
                object_identity=object_identity,
                axis_key=axis_key,
            )
            for axis_key in axis_keys
        }
        max_object_id_by_axis = {
            axis_key: 0
            for axis_key, declared_ids in declared_ids_by_axis.items()
            if not declared_ids
        }
        for object_id, axis_key in projection.row_keys:
            if object_id is None:
                continue
            declared_ids = declared_ids_by_axis.get(axis_key, ())
            if declared_ids:
                if object_id not in declared_ids:
                    raise ValueError(
                        f"{type(self).__name__} projected object {object_id} "
                        f"outside declared object domain {declared_ids!r} for "
                        f"axis {axis_key!r}."
                    )
                continue
            if axis_key not in max_object_id_by_axis:
                continue
            max_object_id_by_axis[axis_key] = max(
                max_object_id_by_axis[axis_key], object_id
            )
        return {
            axis_key: (
                declared_ids_by_axis[axis_key]
                if declared_ids_by_axis[axis_key]
                else tuple(range(1, max_object_id_by_axis[axis_key] + 1))
            )
            for axis_key in axis_keys
        }


class DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy(
    CompactObjectMeasurementRowIdentityPolicy
):
    """Compact measured rows, then pad to the declared object-label domain."""

    def project_completion_rows(
        self,
        request: "ObjectMeasurementRowIdentityProjectionRequest",
        object_identity: MeasurementObjectRowIdentity,
        *,
        label_payload: CellProfilerRuntimeValue,
        original_row_count: int,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Project label IDs to ordinals from the declared label domain."""
        if object_identity is not MeasurementObjectRowIdentity.ROW_ORDINAL:
            return super().project_completion_rows(
                request,
                object_identity,
                label_payload=label_payload,
                original_row_count=original_row_count,
            )
        schema = ObjectMeasurementRowCompletionSchema.for_completion_fields(
            object_id_field=request.object_id_field,
            axis_fields=request.axis_fields,
        )
        ordinal_by_axis_label: dict[
            CellProfilerRuntimeValues, dict[int, int] | None
        ] = {}
        ordinal_state = ObjectMeasurementRowOrdinalProjectionState()
        rows: list[CellProfilerRuntimeValue] = []
        row_keys: list[tuple[int, CellProfilerRuntimeValues]] = []
        measured_row_keys: list[tuple[int, CellProfilerRuntimeValues]] = []
        axis_keys: list[CellProfilerRuntimeValues] = []
        row_entries: list[
            tuple[
                CellProfilerRuntimeValue,
                MeasurementRowMapping,
                CellProfilerRuntimeValues,
                bool,
            ]
        ] = []
        for row in request.rows:
            row_mapping = measurement_row_mapping(row)
            axis_key = request.axis_key_from_mapping(row_mapping)
            if axis_key not in ordinal_by_axis_label:
                explicit_label_ids = schema.explicit_label_ids_for_axis(
                    label_payload=label_payload,
                    axis_key=axis_key,
                )
                if explicit_label_ids is None:
                    ordinal_by_axis_label[axis_key] = None
                else:
                    ordinal_by_axis_label[axis_key] = {
                        label_id: ordinal
                        for ordinal, label_id in enumerate(explicit_label_ids, start=1)
                    }
            if axis_key not in axis_keys:
                axis_keys.append(axis_key)
            measured = self.row_has_measured_object(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            )
            row_entries.append((row, row_mapping, axis_key, measured))
            if measured and ordinal_by_axis_label[axis_key] is None:
                ordinal_state.register_measured_object(
                    row_mapping,
                    axis_key=axis_key,
                    object_id_field=request.object_id_field,
                )
        for row, row_mapping, axis_key, measured in row_entries:
            if not measured and not self.retains_unmeasured_compact_row(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            ):
                continue
            source_object_id = request.object_label(row)
            if source_object_id is None:
                raise ValueError(
                    f"{type(self).__name__} cannot project a compact "
                    "declared-domain row without an object ID."
                )
            explicit_ordinal_by_label = ordinal_by_axis_label[axis_key]
            if explicit_ordinal_by_label is None:
                if measured:
                    ordinal = ordinal_state.ordinal_for_measured_object(
                        row_mapping,
                        axis_key=axis_key,
                        object_id_field=request.object_id_field,
                    )
                else:
                    ordinal = ordinal_state.next_unbound_ordinal(axis_key)
            elif source_object_id not in explicit_ordinal_by_label:
                raise ValueError(
                    f"{type(self).__name__} cannot project source object "
                    f"{source_object_id} on axis {axis_key!r}; it is absent from "
                    "the declared label domain."
                )
            else:
                ordinal = explicit_ordinal_by_label[source_object_id]
            rows.append(
                MeasurementObjectRowIdentityProjectionStrategy
                .for_enum_member(object_identity)
                .row_with_object_id(request, row, ordinal)
            )
            row_key = (ordinal, axis_key)
            row_keys.append(row_key)
            if measured:
                measured_row_keys.append(row_key)
        projection = ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(rows),
            row_keys=ObjectMeasurementProjectedRowKeys(tuple(row_keys)),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_row_keys)
            ),
            axis_keys=tuple(axis_keys),
        )
        object_ids = tuple(
            sorted(dict.fromkeys(object_id for object_id, _axis_key in row_keys))
        )
        object_order = {
            object_id: index for index, object_id in enumerate(object_ids)
        }
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        ordered_entries = tuple(
            sorted(
                enumerate(zip(projection.rows, projection.row_keys, strict=True)),
                key=lambda item: projection.row_order_key(
                    item[1][1],
                    item[0],
                    object_order=object_order,
                    axis_order=axis_order,
                ),
            )
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "declared_domain_object_measurement_row_identity_projection",
            0.0,
            policy=type(self).__name__,
            object_identity=object_identity.value,
            rows=original_row_count,
            projected_rows=len(rows),
            axis_count=len(axis_keys),
            max_object_id=projection.row_keys.max_object_id_or_none(),
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(row for _index, (row, _row_key) in ordered_entries),
            row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(row_key for _index, (_row, row_key) in ordered_entries)
            ),
            measured_row_keys=projection.measured_row_keys,
            axis_keys=tuple(axis_keys),
        )


class PreserveCompleteDenseDomainRowsMixin:
    """Preserve rows that already cover the concrete dense label domain."""

    def already_complete_dense_domain_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        schema: "ObjectMeasurementRowCompletionSchema",
        object_identity: MeasurementObjectRowIdentity,
        label_payload: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValueSequence | None:
        completed_rows = super().already_complete_dense_domain_rows(
            rows,
            schema=schema,
            object_identity=MeasurementObjectRowIdentity.LABEL_ID,
            label_payload=label_payload,
        )
        if completed_rows is not None:
            return completed_rows
        return super().already_complete_dense_domain_rows(
            rows,
            schema=schema,
            object_identity=object_identity,
            label_payload=label_payload,
        )


class FeatureAnchoredCompactObjectMeasurementRowPolicy(
    CompactMeasuredObjectMeasurementRowPolicy
):
    """Compact rows whose measuredness is anchored by declared feature fields."""

    measured_object_features: ClassVar[tuple[RuntimeMeasurementFeature, ...]] = ()
    retained_unmeasured_compact_features: ClassVar[
        tuple[RuntimeMeasurementFeature, ...]
    ] = ()

    def object_identity_for_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        label_payload: CellProfilerRuntimeValue,
    ) -> MeasurementObjectRowIdentity:
        """Use compact object numbering for shape rows."""
        del rows, label_payload
        return self.object_identity()

    def annotates_projected_object_identity(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        object_identity: MeasurementObjectRowIdentity,
    ) -> bool:
        """Annotate only rows carrying declared compact feature fields."""
        if object_identity is MeasurementObjectRowIdentity.LABEL_ID:
            return False
        return self.row_uses_declared_compact_identity(row_mapping)

    @classmethod
    def rows_use_declared_compact_identity(
        cls, rows: CellProfilerRuntimeValueSequence
    ) -> bool:
        return any(
            (
                cls.row_uses_declared_compact_identity(measurement_row_mapping(row))
                for row in rows
            )
        )

    @classmethod
    def row_uses_declared_compact_identity(
        cls, row_mapping: MeasurementRowMapping
    ) -> bool:
        compact_fields = frozenset(
            (
                feature.value
                for feature in (
                    *cls.measured_object_features,
                    *cls.retained_unmeasured_compact_features,
                )
            )
        )
        return any((field_name in compact_fields for field_name in row_mapping))

    def row_has_measured_object(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        anchor_fields = tuple(
            (feature.value for feature in type(self).measured_object_features)
        )
        if any((field_name in row_mapping for field_name in anchor_fields)):
            return any(
                (
                    self.measurement_value_is_present(row_mapping.get(field_name))
                    for field_name in anchor_fields
                )
            )
        retained_fields = tuple(
            (
                feature.value
                for feature in type(self).retained_unmeasured_compact_features
            )
        )
        if any((field_name in row_mapping for field_name in retained_fields)):
            return False
        return super().row_has_measured_object(
            row_mapping, object_id_field=object_id_field, axis_fields=axis_fields
        )


class DenseColumnarObjectMeasurementRowsMixin:
    """Policy mixin for columnar rows that already match the label-id domain."""

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
            and (
                self.object_identity_for_rows(rows, label_payload=label_payload)
                is MeasurementObjectRowIdentity.LABEL_ID
            )
        ):
            return rows
        return super().complete_rows(rows, label_payload=label_payload, func=func)


class DefaultObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Default CellProfiler object-row policy."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value
