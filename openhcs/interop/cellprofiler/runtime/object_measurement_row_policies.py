"""CellProfiler object-measurement row ownership and completion policies."""

from __future__ import annotations
from abc import ABC
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ObjectLabelsArtifactType,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementSparseColumnarRows,
    MeasurementRowOwnership,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScalarLiteral,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadataCompositionMode,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerMeasurementImage,
    CellProfilerSourceImagePair,
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
    ObjectMeasurementRowIdentityProjectionResult,
    ObjectMeasurementRowOrdinalProjectionState,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    CellProfilerModuleAuthority,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeCallableKwargs
from collections.abc import Mapping
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )


@dataclass(frozen=True, slots=True)
class ObjectMeasurementInvocation:
    """One semantic object-measurement function invocation."""

    kwargs: RuntimeCallableKwargs
    source_pair: CellProfilerSourceImagePair | None = None

    def lowered_kwargs(self) -> dict[str, RuntimeCallableArgument]:
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

    def lowered_kwargs(self) -> dict[str, RuntimeCallableArgument]:
        assert self.source_pair is not None
        return {
            **self.kwargs,
            **self.source_pair.invocation_kwargs(
                first_channel_kwarg=self.first_channel_kwarg,
                second_channel_kwarg=self.second_channel_kwarg,
            ),
        }


class CellProfilerObjectMeasurementRowPolicy(
    CellProfilerModuleAuthority,
    ABC,
):
    """Nominal export-row policy for object-scoped measurement modules."""

    row_identity: ClassVar[MeasurementObjectRowIdentity] = (
        MeasurementObjectRowIdentity.LABEL_ID
    )
    missing_value_policy: ClassVar[MissingObjectMeasurementValuePolicy] = (
        MissingObjectMeasurementValuePolicy.NAN
    )
    explicit_row_ownership_required: ClassVar[bool] = False
    measurement_record_excluded_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def runtime_object_measurement_row_policy(cls):
        """Return the MRO-selected policy implemented by this module class."""
        return cls()

    @classmethod
    def complete_table_measurement_rows(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: ColumnarRows,
    ) -> ColumnarRows:
        """Complete raw table rows over the module's declared object domain."""
        from openhcs.core.artifacts import ObjectLabelsArtifactType

        object_inputs = ArtifactSpecCollection(
            request.callable_contract.artifact_inputs.specs
        ).of_artifact_type(ObjectLabelsArtifactType)
        if len(ArtifactSpecCollection(object_inputs).ref_set()) != 1:
            return rows
        return cls.runtime_object_measurement_row_policy().complete_rows(
            rows,
            label_payload=request.artifact_source_payload(
                request.exact_input_edge(object_inputs[0])
            ),
        )

    @classmethod
    def measurement_record_source_image_name(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: ColumnarRows,
    ) -> str | None:
        """Apply the row policy's table-level image ownership to direct records."""

        source_image_name = super().measurement_record_source_image_name(request, rows)
        row_policy = cls.runtime_object_measurement_row_policy()
        return row_policy.table_source_image_name(
            (),
            (
                request.source.source_image_name
                if row_policy.requires_explicit_row_ownership()
                else source_image_name
            ),
        )

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Retain the image-table owner for explicitly mixed-owner records."""

        if cls.runtime_object_measurement_row_policy().requires_explicit_row_ownership():
            return False
        return super().clear_source_when_rows_declare_object_name()

    def object_identity(self) -> MeasurementObjectRowIdentity:
        """Return the object identity projection for rows emitted by this module."""
        return MeasurementObjectRowIdentity(type(self).row_identity)

    def object_identity_for_label_payload(
        self, label_payload: RuntimeCallableArgument
    ) -> MeasurementObjectRowIdentity:
        """Return row identity for a concrete object-measurement label domain."""
        if not isinstance(label_payload, ObjectLabelValue):
            raise TypeError(
                "Object measurement row identity requires an ObjectLabelValue, "
                f"got {type(label_payload).__name__}."
            )
        return label_payload.measurement_object_row_identity(self.object_identity())

    def object_identity_for_rows(
        self,
        rows: ColumnarRows,
        *,
        label_payload: RuntimeCallableArgument,
    ) -> MeasurementObjectRowIdentity:
        """Return row identity for the concrete rows emitted by a module."""
        if rows.object_row_identity is not None:
            return rows.object_row_identity
        return self.object_identity_for_label_payload(label_payload)

    def row_identity_axis_fields(
        self,
        axis_fields: Sequence[str],
        *,
        label_payload: RuntimeCallableArgument | None = None,
    ) -> tuple[str, ...]:
        """Return row-axis fields that partition object-row identity."""
        del label_payload
        return tuple(axis_fields)

    def invocations(
        self,
        measurement_image: CellProfilerMeasurementImage,
        kwargs: RuntimeCallableKwargs,
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        """Return semantic function invocations for this measurement image."""
        del measurement_image
        return (ObjectMeasurementInvocation(kwargs=kwargs),)

    def project_rows(
        self, rows: ColumnarRows, invocation: ObjectMeasurementInvocation
    ) -> ColumnarRows:
        """Return emitted rows projected into this module's feature namespace."""
        del invocation
        return rows

    def split_scoped_rows(
        self, rows: ColumnarRows
    ) -> tuple[ColumnarRows, ColumnarRows]:
        """Partition object-scoped measurement rows from image-scoped rows."""
        if (
            rows.covers_declared_object_measurement_domain
            and rows.object_row_identity is not None
        ):
            return (
                rows,
                MeasurementSparseColumnarRows.from_rows((), fields=rows.fields),
            )
        object_rows: list[Mapping[str, RuntimeCallableArgument]] = []
        non_object_rows: list[Mapping[str, RuntimeCallableArgument]] = []
        for row_mapping in rows.iter_row_mappings():
            if self.row_is_object_scoped(row_mapping):
                object_rows.append(row_mapping)
            else:
                non_object_rows.append(row_mapping)
        if not non_object_rows:
            return (
                rows,
                MeasurementSparseColumnarRows.from_rows(
                    (),
                    fields=rows.fields,
                ),
            )
        if not object_rows:
            return (
                MeasurementSparseColumnarRows.from_rows(
                    (),
                    fields=rows.fields,
                    object_row_identity=rows.object_row_identity,
                ),
                rows,
            )
        return (
            MeasurementSparseColumnarRows.from_rows(
                tuple(object_rows),
                fields=rows.fields,
                declared_object_measurement_domain_covered=(
                    rows.covers_declared_object_measurement_domain
                ),
                object_row_identity=rows.object_row_identity,
            ),
            MeasurementSparseColumnarRows.from_rows(
                tuple(non_object_rows),
                fields=rows.fields,
            ),
        )

    def table_source_image_name(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        source_image_name: str | None,
    ) -> str | None:
        """Return table-level source ownership for rows emitted by this policy."""
        if not measurement_images:
            if self.requires_explicit_row_ownership():
                return self.image_row_source_image_name(source_image_name)
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
            or len(ArtifactSpecCollection(object_inputs).ref_set()) != 1
            or measurement_row_source_names_required(measurement_images)
            or self.requires_explicit_row_ownership()
        ):
            return object_name
        return None

    def table_object_owner(
        self,
        active_inputs: tuple[ArtifactSpec, ...],
        *,
        contains_image_measurement_rows: bool = False,
    ) -> str | None:
        """Return table-level object ownership for materialized measurements."""
        object_inputs = ArtifactSpecCollection(active_inputs).of_artifact_type(
            ObjectLabelsArtifactType
        )
        if (
            contains_image_measurement_rows
            or len(ArtifactSpecCollection(object_inputs).ref_set()) != 1
        ):
            return None
        return object_inputs[0].name

    def requires_explicit_row_ownership(self) -> bool:
        """Return whether emitted rows carry mixed measurement ownership."""
        return type(self).explicit_row_ownership_required

    def validate_table_ownership(
        self,
        table: MeasurementTable,
    ) -> None:
        """Require explicit ownership when the module declares that invariant."""

        if not self.requires_explicit_row_ownership():
            return
        unowned_indices = tuple(
            index
            for index, row_mapping in enumerate(table.rows.iter_row_mappings())
            if (
                measurement_row_object_name(row_mapping) is None
                and measurement_row_source_image_name(row_mapping) is None
            )
        )
        if unowned_indices:
            raise ValueError(
                f"{type(self).__name__} requires every mixed-scope measurement "
                "row to declare object or source-image ownership; unowned row "
                f"indices are {unowned_indices!r}."
            )

    def annotate_record_rows(
        self,
        rows: ColumnarRows,
        *,
        object_name: str | None,
        source_image_name: str | None,
    ) -> ColumnarRows:
        """Return rows with row-level ownership declared when policy requires it."""
        if not self.requires_explicit_row_ownership():
            return rows
        annotated_rows = tuple(
            self.annotate_record_row(
                row_mapping,
                object_name=object_name,
                source_image_name=source_image_name,
            )
            for row_mapping in rows.iter_row_mappings()
        )
        ownership_fields = tuple(
            FieldSpec(field_name, str)
            for field_name, value in (
                (MeasurementRowAxisField.OBJECT_NAME.value, object_name),
                (MeasurementRowAxisField.SOURCE_IMAGE_NAME.value, source_image_name),
            )
            if value is not None
            and field_name not in {field.name for field in rows.fields}
        )
        return MeasurementSparseColumnarRows.from_rows(
            annotated_rows,
            fields=FieldSpec.merge_exact(
                (rows.fields, ownership_fields),
                context="owned object measurement fields",
            ),
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=rows.object_row_identity,
        )

    def annotate_record_row(
        self,
        row: RuntimeCallableArgument,
        *,
        object_name: str | None,
        source_image_name: str | None,
    ) -> RuntimeCallableArgument:
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

    def row_is_object_scoped(self, row: Mapping[str, RuntimeCallableArgument]) -> bool:
        """Return whether a raw emitted row belongs to the object domain."""
        return any(
            field_name in row
            for field_name in MeasurementRowAxisField.object_id_field_names()
        )

    def row_has_measured_object(
        self,
        row_mapping: Mapping[str, RuntimeCallableArgument],
        *,
        metadata_fields: Collection[str],
    ) -> bool:
        """Return whether a source row should consume an object row identity."""
        return self.row_has_result_payload(
            row_mapping,
            metadata_fields=metadata_fields,
        )

    def row_has_result_payload(
        self,
        row_mapping: Mapping[str, RuntimeCallableArgument],
        *,
        metadata_fields: Collection[str],
    ) -> bool:
        """Return whether a row carries result values, not just identity padding."""
        return self.measurement_values_have_result_payload(
            value
            for field_name, value in row_mapping.items()
            if field_name not in metadata_fields
        )

    def measurement_values_have_result_payload(
        self,
        values: Iterable[RuntimeCallableArgument],
    ) -> bool:
        """Return whether a sequence contains any present measurement value."""
        return any(self.measurement_value_is_present(value) for value in values)

    def measurement_value_is_present(self, value: RuntimeCallableArgument) -> bool:
        """Return whether a measurement cell is an observed value, not padding."""
        return MeasurementScalarLiteral(value).is_present_measurement_value

    def retains_unmeasured_compact_row(
        self,
        row_mapping: Mapping[str, RuntimeCallableArgument],
        *,
        schema: ObjectMeasurementRowCompletionSchema,
    ) -> bool:
        """Return whether compact row projection should keep an unmeasured row."""
        del row_mapping, schema
        return True

    def required_object_ids_for_axis(
        self,
        *,
        label_payload: RuntimeCallableArgument,
        projected_rows: ColumnarRows,
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_key: tuple[RuntimeCallableArgument, ...],
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
        label_payload: RuntimeCallableArgument,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_keys: Sequence[tuple[RuntimeCallableArgument, ...]],
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
        rows: ColumnarRows,
        *,
        label_payload: RuntimeCallableArgument,
    ) -> ColumnarRows:
        """Pad per-object measurement rows across this policy's object domain."""
        rows.validate_fields()
        object_identity = self.object_identity_for_rows(
            rows, label_payload=label_payload
        )
        schema = self.completion_schema(
            rows,
            label_payload=label_payload,
        )
        completed_rows = self.already_complete_dense_domain_rows(
            rows,
            schema=schema,
            object_identity=object_identity,
            label_payload=label_payload,
        )
        if completed_rows is not None:
            return self.complete_object_domain_rows(
                completed_rows,
                object_identity=object_identity,
            )
        projection = self.project_completion_rows(
            rows,
            schema,
            object_identity,
            label_payload=label_payload,
        )
        projected_rows = projection.rows
        if self.projected_rows_have_no_object_ids(projection):
            return MeasurementSparseColumnarRows.from_rows(
                projected_rows.row_mappings(),
                fields=projected_rows.fields,
                object_row_identity=object_identity,
            )
        axis_keys = self.completion_axis_keys(
            schema, projection, label_payload=label_payload
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
            return MeasurementSparseColumnarRows.from_rows(
                projected_rows.row_mappings(),
                fields=projected_rows.fields,
                object_row_identity=object_identity,
            )
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
                ),
                object_identity=object_identity,
            )
        missing_projection_rows = schema.missing_columnar_rows(
            missing_row_keys=missing_row_keys,
            label_payload=label_payload,
            row_policy=self,
            object_row_identity=object_identity,
        )
        completed_projection_rows = ConcatenatedColumnarRows(
            (bounded_projection.rows, missing_projection_rows)
        )
        return self.complete_object_domain_rows(
            bounded_projection.ordered_rows(
                rows=completed_projection_rows,
                row_keys=bounded_projection.row_keys.appended(missing_row_keys),
                object_ids=object_ids,
                axis_keys=axis_keys,
            ),
            object_identity=object_identity,
        )

    def complete_object_domain_rows(
        self,
        rows: ColumnarRows,
        *,
        object_identity: MeasurementObjectRowIdentity,
    ) -> ColumnarRows:
        """Return rows marked as covering this policy's declared object domain."""
        if (
            rows.covers_declared_object_measurement_domain
            and rows.object_row_identity is object_identity
        ):
            return rows
        return MeasurementProjectedColumnarRows.from_columnar_rows(
            rows,
            row_indices=None,
            declared_object_measurement_domain_covered=True,
            object_row_identity=object_identity,
        )

    def already_complete_dense_domain_rows(
        self,
        rows: ColumnarRows,
        *,
        schema: "ObjectMeasurementRowCompletionSchema",
        object_identity: MeasurementObjectRowIdentity,
        label_payload: RuntimeCallableArgument,
    ) -> ColumnarRows | None:
        """Return rows unchanged when they already exactly cover the dense domain."""
        if (
            object_identity is not MeasurementObjectRowIdentity.LABEL_ID
            or not rows.row_count()
        ):
            return None
        row_keys: list[tuple[int, tuple[RuntimeCallableArgument, ...]]] = []
        axis_keys: list[tuple[RuntimeCallableArgument, ...]] = []
        for row_mapping in rows.iter_row_mappings():
            object_id = schema.object_label(row_mapping)
            if object_id is None:
                return None
            axis_key = schema.axis_key_from_mapping(row_mapping)
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
        return rows

    def completion_schema(
        self,
        rows: ColumnarRows,
        *,
        label_payload: RuntimeCallableArgument,
    ) -> "ObjectMeasurementRowCompletionSchema":
        """Return row-completion schema after policy-specific identity projection."""
        schema = ObjectMeasurementRowCompletionSchema.from_fields(rows.fields)
        identity_axis_fields = self.row_identity_axis_fields(
            schema.axis_fields, label_payload=label_payload
        )
        return ObjectMeasurementRowCompletionSchema(
            fields=schema.fields,
            object_id_field=schema.object_id_field,
            axis_fields=identity_axis_fields,
        )

    def project_completion_rows(
        self,
        rows: ColumnarRows,
        schema: "ObjectMeasurementRowCompletionSchema",
        object_identity: MeasurementObjectRowIdentity,
        *,
        label_payload: RuntimeCallableArgument,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Project emitted rows into this policy's object identity domain."""
        del label_payload
        projection = MeasurementObjectRowIdentityProjectionStrategy.for_enum_member(
            object_identity
        ).project_rows(rows, schema, self)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "object_measurement_row_identity_projection",
            0.0,
            policy=type(self).__name__,
            object_identity=object_identity.value,
            rows=rows.row_count(),
            projected_rows=projection.rows.row_count(),
            axis_count=len(projection.axis_keys),
            max_object_id=projection.row_keys.max_object_id_or_none(),
        )
        return projection

    @staticmethod
    def projected_rows_have_no_object_ids(
        projection: "ObjectMeasurementRowIdentityProjectionResult",
    ) -> bool:
        return bool(projection.rows.row_count()) and (
            not projection.row_keys.has_object_ids()
        )

    def completion_axis_keys(
        self,
        schema: "ObjectMeasurementRowCompletionSchema",
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        *,
        label_payload: RuntimeCallableArgument,
    ) -> tuple[tuple[RuntimeCallableArgument, ...], ...]:
        """Return measurement axis keys used to complete missing object rows."""
        axis_keys = schema.axis_keys_for_label_payload(
            projection, label_payload=label_payload
        )
        if axis_keys:
            return axis_keys
        return ((),)

    def missing_measurement_value(
        self,
        *,
        object_id: int,
        label_payload: RuntimeCallableArgument,
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

    def missing_measurement_values(
        self,
        *,
        object_ids: Sequence[int],
        label_payload: RuntimeCallableArgument,
        field_name: str,
        positive_label_extents: Sequence[int | None],
    ) -> tuple[float, ...]:
        """Return one missing field through one nominal strategy resolution."""
        if not isinstance(label_payload, ObjectLabelValue):
            raise TypeError(
                "Missing object-measurement values require an ObjectLabelValue, "
                f"got {type(label_payload).__name__}."
            )
        value_policy = MissingObjectMeasurementValuePolicy(
            type(self).missing_value_policy
        )
        return MissingObjectMeasurementValueStrategy.for_enum_member(
            value_policy
        ).missing_values(
            object_ids=object_ids,
            label_payload=label_payload,
            field_name=field_name,
            positive_label_extents=positive_label_extents,
        )


class DeclaredObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Generated base for modules with declared measurement-row identity."""


class CompactObjectMeasurementRowIdentityPolicy(DeclaredObjectMeasurementRowPolicy):
    """Use CP's compact row identity for emitted measurement rows."""

    row_identity = MeasurementObjectRowIdentity.ROW_ORDINAL


class DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy(
    CompactObjectMeasurementRowIdentityPolicy
):
    """Compact measured rows, then pad to the declared object-label domain."""

    def project_completion_rows(
        self,
        source_rows: ColumnarRows,
        schema: "ObjectMeasurementRowCompletionSchema",
        object_identity: MeasurementObjectRowIdentity,
        *,
        label_payload: RuntimeCallableArgument,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Project label IDs to ordinals from the declared label domain."""
        if object_identity is not MeasurementObjectRowIdentity.ROW_ORDINAL:
            return super().project_completion_rows(
                source_rows,
                schema,
                object_identity,
                label_payload=label_payload,
            )
        ordinal_by_axis_label: dict[tuple[RuntimeCallableArgument, ...], dict[int, int]] = {}
        projected_rows: list[Mapping[str, RuntimeCallableArgument]] = []
        row_keys: list[tuple[int, tuple[RuntimeCallableArgument, ...]]] = []
        measured_row_keys: list[tuple[int, tuple[RuntimeCallableArgument, ...]]] = []
        axis_keys: list[tuple[RuntimeCallableArgument, ...]] = []
        row_entries: list[
            tuple[
                Mapping[str, RuntimeCallableArgument],
                tuple[RuntimeCallableArgument, ...],
                bool,
            ]
        ] = []
        metadata_fields = schema.metadata_fields
        for row_mapping in source_rows.iter_row_mappings():
            axis_key = schema.axis_key_from_mapping(row_mapping)
            if axis_key not in ordinal_by_axis_label:
                explicit_label_ids = schema.explicit_label_ids_for_axis(
                    label_payload=label_payload,
                    axis_key=axis_key,
                )
                if explicit_label_ids is None:
                    raise ValueError(
                        "Declared-domain row-ordinal projection requires an "
                        f"explicit object-ID domain for axis {axis_key!r}."
                    )
                ordinal_by_axis_label[axis_key] = {
                    label_id: ordinal
                    for ordinal, label_id in enumerate(explicit_label_ids, start=1)
                }
            if axis_key not in axis_keys:
                axis_keys.append(axis_key)
            measured = self.row_has_measured_object(
                row_mapping,
                metadata_fields=metadata_fields,
            )
            row_entries.append((row_mapping, axis_key, measured))
        ordinal_state = ObjectMeasurementRowOrdinalProjectionState(
            ordinal_by_axis_label
        )
        for row_mapping, axis_key, measured in row_entries:
            if not measured and not self.retains_unmeasured_compact_row(
                row_mapping,
                schema=schema,
            ):
                continue
            ordinal = ordinal_state.ordinal_for_declared_object(
                row_mapping,
                axis_key=axis_key,
                object_id_field=schema.object_id_field,
            )
            projected_rows.append(
                MeasurementObjectRowIdentityProjectionStrategy.for_enum_member(
                    object_identity
                ).row_with_object_id(schema, row_mapping, ordinal)
            )
            row_key = (ordinal, axis_key)
            row_keys.append(row_key)
            if measured:
                measured_row_keys.append(row_key)
        projection = ObjectMeasurementRowIdentityProjectionResult(
            rows=MeasurementSparseColumnarRows.from_rows(
                tuple(projected_rows),
                fields=source_rows.fields,
                object_row_identity=object_identity,
            ),
            row_keys=ObjectMeasurementProjectedRowKeys(tuple(row_keys)),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_row_keys)
            ),
            axis_keys=tuple(axis_keys),
        )
        object_ids = tuple(
            sorted(dict.fromkeys(object_id for object_id, _axis_key in row_keys))
        )
        object_order = {object_id: index for index, object_id in enumerate(object_ids)}
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        ordered_entries = tuple(
            sorted(
                enumerate(
                    zip(
                        projection.rows.iter_row_mappings(),
                        projection.row_keys,
                        strict=True,
                    )
                ),
                key=lambda item: projection.row_order_key(
                    item[1][1],
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
            rows=source_rows.row_count(),
            projected_rows=projection.rows.row_count(),
            axis_count=len(axis_keys),
            max_object_id=projection.row_keys.max_object_id_or_none(),
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=MeasurementSparseColumnarRows.from_rows(
                tuple(row for _index, (row, _row_key) in ordered_entries),
                fields=source_rows.fields,
                object_row_identity=object_identity,
            ),
            row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(row_key for _index, (_row, row_key) in ordered_entries)
            ),
            measured_row_keys=projection.measured_row_keys,
            axis_keys=tuple(axis_keys),
        )


class DenseColumnarObjectMeasurementRowsMixin:
    """Policy mixin for columnar rows that already match the label-id domain."""

    def complete_rows(
        self,
        rows: ColumnarRows,
        *,
        label_payload: RuntimeCallableArgument,
    ) -> ColumnarRows:
        if rows.covers_declared_object_measurement_domain and (
            self.object_identity_for_rows(rows, label_payload=label_payload)
            is MeasurementObjectRowIdentity.LABEL_ID
        ):
            return rows
        return super().complete_rows(rows, label_payload=label_payload)
