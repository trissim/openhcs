"""Generic CellProfiler measurement-table declarations."""

from __future__ import annotations
from abc import ABC
from collections.abc import Mapping
from dataclasses import (
    asdict,
    fields as dataclass_fields,
    is_dataclass,
    replace as dataclass_replace,
)
import math
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Self,
    get_args,
    get_origin,
    get_type_hints,
)

from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementSubject,
)
from openhcs.core.runtime_plane_projection import (
    RuntimeSliceIdentityProjectableValue,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementRowsAxisProjection,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.source_image_provenance import SourceImageProvenance

if TYPE_CHECKING:
    from openhcs.core.artifacts import ArtifactSpec
    from openhcs.interop.cellprofiler.runtime.measurement_rows import (
        CellProfilerResultMeasurementRows,
    )
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )
    from openhcs.core.steps.function_runtime import RuntimeCallableKwargs


def measurement_table_for_module(
    request: CellProfilerOutputRecordRequest,
) -> MeasurementTable:
    """Return the native measurement table declared by the backend module."""
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_function_name(
        request.callable_contract.function_name
    )
    if module_type is None:
        raise KeyError(
            "No CellProfiler module declaration owns callable "
            f"{request.callable_contract.function_name!r}."
        )
    table = module_type.measurement_table(request)
    if not isinstance(table, MeasurementTable):
        raise TypeError(
            f"{module_type.__name__}.measurement_table() must return MeasurementTable."
        )
    return table


class CellProfilerMeasurementTableModule(ABC):
    """Generic measurement-table assembly contract for module declarations."""

    measurement_record_excluded_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def measurement_table(
        cls, request: CellProfilerOutputRecordRequest
    ) -> MeasurementTable:
        rows = cls.measurement_record_rows(request)
        object_name = cls.measurement_record_object_name(request, rows)
        source_image_name = cls.measurement_record_source_image_name(request, rows)
        source_metadata = cls.measurement_record_source_metadata(request, rows)
        rows = cls.prepare_measurement_record_rows(
            rows,
            source_image_name=source_image_name,
        )
        if (
            cls.clear_source_when_rows_declare_object_name()
            and cls.rows_only_declare_object_name(rows)
        ):
            source_image_name = None
        return cls.build_measurement_table(
            name=request.spec.name,
            rows=rows,
            object_name=object_name,
            source_image_name=source_image_name,
            source_metadata=source_metadata,
        )

    @classmethod
    def prepare_measurement_record_rows(
        cls,
        rows: ColumnarRows,
        *,
        source_image_name: str | None,
    ) -> ColumnarRows:
        """Apply module-owned row semantics before field materialization."""
        rows = cls.filter_measurement_record_rows(rows)
        return cls.project_measurement_record_rows(
            rows,
            source_image_name=source_image_name,
        )

    @classmethod
    def project_measurement_record_rows(
        cls,
        rows: ColumnarRows,
        *,
        source_image_name: str | None,
    ) -> ColumnarRows:
        """Project producer-owned feature identities into emitted rows."""
        del cls, source_image_name
        return rows

    @classmethod
    def build_measurement_table(
        cls,
        *,
        name: str,
        rows: ColumnarRows,
        object_name: str | None,
        source_image_name: str | None,
        source_metadata: ImagePayloadMetadata,
    ) -> MeasurementTable:
        """Build one canonical module-owned native measurement table."""
        subject = (
            MeasurementSubject(MeasurementScope.OBJECT, object_name)
            if object_name is not None
            else MeasurementSubject(
                MeasurementScope.IMAGE,
                source_image_name or MeasurementScope.IMAGE.value,
            )
        )
        return MeasurementTable(
            name=name,
            rows=rows,
            source_image_name=source_image_name,
            subject=subject,
            measurement_feature_owner=cls,
            source_provenance=cls.measurement_source_provenance_for_rows(
                rows,
                source_metadata,
            ),
        )

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> ColumnarRows:
        row_batches = []
        for projection_type in cls.measurement_row_projection_types():
            row_batches.append(projection_type.for_request(cls, request).rows())
        if not row_batches:
            return MeasurementSparseColumnarRows.from_rows((), fields=())
        if len(row_batches) == 1:
            return row_batches[0]
        return ConcatenatedColumnarRows(tuple(row_batches))

    @classmethod
    def complete_table_measurement_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: ColumnarRows,
    ) -> ColumnarRows:
        """Return raw table rows unchanged unless the module declares completion."""
        del cls, request
        return rows

    @classmethod
    def filter_measurement_record_rows(
        cls,
        rows: ColumnarRows,
    ) -> ColumnarRows:
        """Remove module-declared backend-only fields before CP materialization."""
        excluded_fields = cls.measurement_record_excluded_fields
        if not excluded_fields:
            return rows
        retained_fields = tuple(
            field for field in rows.fields if field.name not in excluded_fields
        )
        return MeasurementProjectedColumnarRows(
            {field.name: rows.column_values(field.name) for field in retained_fields},
            fields=retained_fields,
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=rows.object_row_identity,
        )

    @classmethod
    def measurement_row_projection_types(
        cls,
    ) -> tuple[type["CellProfilerResultMeasurementRows"], ...]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            CellProfilerResultMeasurementRows,
        )

        return tuple(
            dict.fromkeys(
                projection_type
                for owner_type in cls.__mro__
                for projection_type in owner_type.__dict__.values()
                if isinstance(projection_type, type)
                and issubclass(projection_type, CellProfilerResultMeasurementRows)
                and projection_type is not CellProfilerResultMeasurementRows
            )
        )

    @classmethod
    def measurement_record_object_name(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> str | None:
        del rows
        return cls.runtime_object_measurement_row_policy().table_object_owner(
            request.callable_contract.artifact_inputs.specs
        )

    @classmethod
    def measurement_record_source_image_name(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> str | None:
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
        from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
            measurement_source_name_for_specs,
        )

        del rows
        module_type = CellProfilerModule.for_function_name(
            request.callable_contract.function_name
        )
        if module_type is None:
            raise KeyError(
                "No CellProfiler module declaration owns callable "
                f"{request.callable_contract.function_name!r}."
            )
        primary_image_inputs = module_type.primary_image_inputs(
            request.callable_contract.resolve_canonical_raw_callable(),
            request.callable_contract.artifact_inputs.specs,
        )
        current_source_name = request.source.source_image_name
        declared_source_names = frozenset(spec.name for spec in primary_image_inputs)
        source_image_name = (
            current_source_name
            if current_source_name in declared_source_names
            else measurement_source_name_for_specs(primary_image_inputs)
        )
        return source_image_name

    @classmethod
    def measurement_record_source_metadata(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> ImagePayloadMetadata:
        del cls, rows
        return request.source.composed_source_metadata(
            (request.source,)
        ) or image_payload_metadata(request.source.payload)

    @staticmethod
    def rows_only_declare_object_name(rows: ColumnarRows) -> bool:
        object_field = MeasurementRowAxisField.OBJECT_NAME.value
        if object_field not in {field.name for field in rows.fields}:
            return False
        return rows.row_count() > 0 and all(
            object_field in row for row in rows.iter_row_mappings()
        )

    @staticmethod
    def measurement_source_provenance_for_rows(
        rows: ColumnarRows,
        source_metadata: ImagePayloadMetadata,
    ) -> SourceImageProvenance:
        source_plane_count = source_metadata.source_provenance.source_plane_count
        if source_plane_count <= 1:
            return source_metadata.source_provenance
        slice_indices = MeasurementRowsAxisProjection.from_rows(
            rows
        ).present_axis_values(MeasurementRowAxisField.SLICE_INDEX.value)
        if len(slice_indices) != 1:
            return source_metadata.source_provenance
        slice_index = slice_indices[0]
        if slice_index >= source_plane_count:
            raise ValueError(
                f"Measurement slice_index {slice_index} exceeds source plane count "
                f"{source_plane_count}."
            )
        return source_metadata.for_source_plane(slice_index).source_provenance

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Return whether row-owned object measurements clear table source context."""
        return True


class MeasurementFeatureRecord(RuntimeSliceIdentityProjectableValue):
    """Dataclass mixin for CP feature rows derived from record fields."""

    measurement_value_field: ClassVar[MeasurementRowValueField] = (
        MeasurementRowValueField.RESULT_VALUE
    )
    measurement_value_dtype: ClassVar[type[object]] = float

    @staticmethod
    def axis_annotation(field_type: object) -> MeasurementRowAxisField | None:
        """Return the nominal measurement axis declared on one record field."""
        if get_origin(field_type) is not Annotated:
            return None
        for annotation in get_args(field_type)[1:]:
            if isinstance(annotation, MeasurementRowAxisField):
                return annotation
        return None

    def with_runtime_slice_identity(
        self,
        *,
        slice_index: int,
        slice_count: int,
    ) -> Self:
        """Stamp the record field declared as the runtime-slice axis."""
        del slice_count
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} must be a dataclass measurement record."
            )
        field_types = get_type_hints(type(self), include_extras=True)
        slice_fields = tuple(
            field.name
            for field in dataclass_fields(self)
            if self.axis_annotation(field_types[field.name])
            is MeasurementRowAxisField.SLICE_INDEX
        )
        if not slice_fields:
            return self
        if len(slice_fields) != 1:
            raise TypeError(
                f"{type(self).__name__} must declare at most one runtime-slice "
                f"measurement axis field, got {slice_fields!r}."
            )
        return dataclass_replace(self, **{slice_fields[0]: int(slice_index)})


class FieldDerivedMeasurementFeatureModule(CellProfilerMeasurementTableModule):
    """Derives CP measurement feature names from module-owned result fields."""

    measurement_feature_family: ClassVar[str | None] = None
    measurement_feature_token_aliases: ClassVar[tuple[tuple[str, str], ...]] = ()

    @classmethod
    def measurement_feature_family_name(cls) -> str:
        family = cls.measurement_feature_family
        if family is not None:
            return family
        try:
            module_name = cls.module_name
        except AttributeError as exc:
            raise TypeError(
                f"{cls.__name__} must declare module_name or measurement_feature_family."
            ) from exc
        return str(module_name)

    @classmethod
    def declared_measurement_feature_family_parts(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Add the field-derived family to inherited finite feature families."""

        inherited = super().declared_measurement_feature_family_parts()
        family = tuple(
            part for part in cls.measurement_feature_family_name().split("_") if part
        )
        return tuple(dict.fromkeys((family, *inherited)))

    @classmethod
    def measurement_feature_stem(cls, field_name: str) -> str:
        token_aliases = dict(cls.measurement_feature_token_aliases)
        return "".join(
            (
                token_aliases[part]
                if part in token_aliases
                else part[:1].upper() + part[1:]
            )
            for part in str(field_name).split("_")
            if part
        )

    @classmethod
    def measurement_feature_name(
        cls,
        field_name: str,
        *qualified_parts: object,
    ) -> str:
        parts = (
            cls.measurement_feature_family_name(),
            cls.measurement_feature_stem(field_name),
            *(str(part) for part in qualified_parts if part not in (None, "")),
        )
        return "_".join(parts)

    @classmethod
    def mean_measurement_feature_name(
        cls,
        object_name: str,
        feature_name: str,
    ) -> str:
        del cls
        return f"Mean_{object_name}_{feature_name}"

    @classmethod
    def measurement_feature_rows(
        cls,
        *,
        axis_values: Mapping[str, object],
        feature_values: Mapping[str, object],
        qualified_parts: tuple[object, ...] = (),
        value_field: MeasurementRowValueField,
    ) -> list[RuntimeCallableKwargs]:
        return [
            {
                **dict(axis_values),
                MeasurementRowAxisField.FEATURE_NAME.value: cls.measurement_feature_name(
                    str(field_name),
                    *qualified_parts,
                ),
                value_field.value: value,
            }
            for field_name, value in feature_values.items()
        ]

    @classmethod
    def measurement_record_axis_values(
        cls, record: MeasurementFeatureRecord
    ) -> dict[str, object]:
        field_types = get_type_hints(type(record), include_extras=True)
        record_values = asdict(record)
        return {
            axis.value: record_values[field.name]
            for field in dataclass_fields(record)
            for axis in (
                MeasurementFeatureRecord.axis_annotation(field_types[field.name]),
            )
            if axis is not None
        }

    @classmethod
    def measurement_record_field_values(
        cls, record: MeasurementFeatureRecord
    ) -> dict[str, object]:
        field_types = get_type_hints(type(record), include_extras=True)
        record_values = asdict(record)
        return {
            field.name: record_values[field.name]
            for field in dataclass_fields(record)
            if MeasurementFeatureRecord.axis_annotation(field_types[field.name]) is None
        }

    @classmethod
    def measurement_feature_row_fields(
        cls,
        record_type: type[MeasurementFeatureRecord],
    ) -> tuple[FieldSpec, ...]:
        if not is_dataclass(record_type):
            raise TypeError(
                f"{cls.__name__}.measurement_feature_row_fields() requires a "
                f"dataclass record type, got {record_type!r}."
            )
        if not issubclass(record_type, MeasurementFeatureRecord):
            raise TypeError(
                f"{record_type.__name__} must inherit MeasurementFeatureRecord."
            )
        field_types = get_type_hints(record_type, include_extras=True)
        return (
            *(
                FieldSpec.from_annotation(
                    axis.value,
                    field_types[field.name],
                )
                for field in dataclass_fields(record_type)
                for axis in (
                    MeasurementFeatureRecord.axis_annotation(field_types[field.name]),
                )
                if axis is not None
            ),
            FieldSpec(MeasurementRowAxisField.FEATURE_NAME.value, str),
            FieldSpec(
                record_type.measurement_value_field.value,
                record_type.measurement_value_dtype,
            ),
        )

    @classmethod
    def measurement_feature_rows_from_records(
        cls,
        records: tuple[MeasurementFeatureRecord, ...],
        *,
        qualified_parts: tuple[object, ...] = (),
    ) -> list[RuntimeCallableKwargs]:
        rows: list[RuntimeCallableKwargs] = []
        for record in records:
            rows.extend(
                cls.measurement_feature_rows(
                    axis_values=cls.measurement_record_axis_values(record),
                    feature_values=cls.measurement_record_field_values(record),
                    qualified_parts=qualified_parts,
                    value_field=type(record).measurement_value_field,
                )
            )
        return rows

    @classmethod
    def mean_measurement_feature_rows(
        cls,
        *,
        axis_values: Mapping[str, object],
        feature_values: Mapping[str, object],
        object_name: str,
        qualified_parts: tuple[object, ...] = (),
        value_field: MeasurementRowValueField,
    ) -> list[RuntimeCallableKwargs]:
        return [
            {
                **dict(axis_values),
                MeasurementRowAxisField.FEATURE_NAME.value: cls.mean_measurement_feature_name(
                    object_name,
                    cls.measurement_feature_name(str(field_name), *qualified_parts),
                ),
                value_field.value: value,
            }
            for field_name, value in feature_values.items()
        ]

    @classmethod
    def mean_measurement_feature_rows_from_records(
        cls,
        records: tuple[MeasurementFeatureRecord, ...],
        *,
        axis_values: Mapping[str, object],
        object_name: str,
        qualified_parts: tuple[object, ...] = (),
    ) -> list[RuntimeCallableKwargs]:
        if not records:
            return []
        values_by_field: dict[str, list[float]] = {}
        for record in records:
            for field_name, value in cls.measurement_record_field_values(
                record
            ).items():
                values_by_field.setdefault(field_name, []).append(float(value))
        mean_values = {
            field_name: (
                sum(finite_values) / len(finite_values)
                if finite_values
                else float("nan")
            )
            for field_name, values in values_by_field.items()
            for finite_values in ([value for value in values if math.isfinite(value)],)
        }
        return cls.mean_measurement_feature_rows(
            axis_values=axis_values,
            feature_values=mean_values,
            object_name=object_name,
            qualified_parts=qualified_parts,
            value_field=type(records[0]).measurement_value_field,
        )


class TableMeasurementRecordRowsMixin(CellProfilerMeasurementTableModule):
    """Adds raw CellProfiler measurement table rows."""

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> ColumnarRows:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            measurement_table_rows,
        )

        rows = (
            MeasurementSparseColumnarRows.from_rows((), fields=())
            if cls.measurement_row_projection_types()
            else measurement_table_rows(request.output_value)
        )
        rows = cls.complete_table_measurement_rows(request, rows)
        return ConcatenatedColumnarRows(
            (rows, super().measurement_record_rows(request))
        )


class RelationshipMeasurementRecordRowsMixin(CellProfilerMeasurementTableModule):
    """Adds relationship rows derived from the module relationship declaration."""

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> ColumnarRows:
        from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
            RelationshipMeasurementRows,
        )

        return ConcatenatedColumnarRows(
            (
                RelationshipMeasurementRows.for_request(request).rows(),
                super().measurement_record_rows(request),
            )
        )


class NoObjectNameMeasurementRecordMixin(CellProfilerMeasurementTableModule):
    """Suppresses object ownership for emitted measurement rows."""

    @classmethod
    def measurement_record_object_name(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> str | None:
        del request, rows
        return None


class PayloadOnlyMeasurementRecordMixin(CellProfilerMeasurementTableModule):
    """Use payload provenance without a table-level source-image owner."""

    @classmethod
    def measurement_record_source_image_name(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> None:
        del cls, request, rows
        return None


class CurrentPayloadMeasurementRecordMixin(PayloadOnlyMeasurementRecordMixin):
    """Uses the current runtime payload without an image-source name."""

    @classmethod
    def measurement_record_source_metadata(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> ImagePayloadMetadata:
        del cls, rows
        return image_payload_metadata(request.source.payload)

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Current-payload measurements keep payload provenance for row-owned data."""
        return False


class ProducedImageMeasurementRecordMixin(CellProfilerMeasurementTableModule):
    """Uses a declared image output as measurement source."""

    @classmethod
    def primary_image_output_spec(
        cls, request: CellProfilerOutputRecordRequest
    ) -> ArtifactSpec | None:
        from openhcs.core.artifacts import ImageArtifactType

        image_specs = tuple(
            spec
            for spec in request.callable_contract.artifact_outputs.of_artifact_type(
                ImageArtifactType
            )
            if spec.sidecar_role is None
        )
        if not image_specs:
            return None
        if len(image_specs) == 1:
            return image_specs[0]
        raise ValueError(
            f"Produced-image measurement ownership requires exactly one primary image output spec, got {[spec.name for spec in image_specs]!r}."
        )

    @classmethod
    def require_primary_image_output_spec(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> ArtifactSpec:
        spec = cls.primary_image_output_spec(request)
        if spec is None:
            raise ValueError("Measurement ownership requires an image output.")
        return spec

    @classmethod
    def measurement_record_source_image_name(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> str | None:
        del rows
        source_spec = cls.primary_image_output_spec(request)
        return None if source_spec is None else source_spec.name

    @classmethod
    def measurement_record_source_metadata(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> ImagePayloadMetadata:
        del rows
        source_spec = cls.primary_image_output_spec(request)
        return image_payload_metadata(
            request.source.payload
            if source_spec is None
            else request.artifact_output_value(source_spec)
        )


class ProducedImagePayloadMeasurementRecordMixin(
    PayloadOnlyMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
):
    """Uses a declared image output payload without an image-source name."""


class DeclaredImageOutputPayloadMeasurementRecordMixin(
    PayloadOnlyMeasurementRecordMixin,
):
    """Uses exact declared image-output provenance as measurement context."""

    @classmethod
    def measurement_record_source_metadata(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> ImagePayloadMetadata:
        from openhcs.core.artifacts import ImageArtifactType
        from openhcs.core.runtime_image_values import (
            ImagePayloadMetadata,
            image_payload_metadata,
        )
        from openhcs.core.source_matching import SourceImageSetIdentityPolicy
        from openhcs.core.source_plane_alignment import (
            SourcePayloadPlaneIdentitySequence,
            SourcePlaneIdentitySequenceAlignment,
        )

        del rows
        image_output_specs = tuple(
            spec
            for spec in request.callable_contract.artifact_outputs.of_artifact_type(
                ImageArtifactType
            )
            if spec.sidecar_role is None
        )
        if not image_output_specs:
            raise ValueError(
                f"{cls.__name__} requires at least one declared image output."
            )
        source_payloads = tuple(
            request.artifact_output_value(spec) for spec in image_output_specs
        )
        identity_policy = SourceImageSetIdentityPolicy.from_source_bindings(
            request.adapter.request.source_binding_plan
        )
        identity_axes = tuple(
            SourcePayloadPlaneIdentitySequence(
                payload,
                identity_policy,
            ).runtime_axis_identities()
            for payload in source_payloads
        )
        missing_identity_specs = tuple(
            image_output_specs[index].ref()
            for index, identity_axis in enumerate(identity_axes)
            if not identity_axis
        )
        if missing_identity_specs:
            raise ValueError(
                f"{cls.__name__} image outputs do not carry producer-declared "
                f"source identity: {missing_identity_specs!r}."
            )
        unaligned_indexes = SourcePlaneIdentitySequenceAlignment.unaligned_axis_indexes(
            identity_axes
        )
        if unaligned_indexes:
            raise ValueError(
                f"{cls.__name__} image outputs do not share one source image-set "
                "axis: "
                f"reference={image_output_specs[0].ref()!r}; "
                f"unaligned={tuple(image_output_specs[index].ref() for index in unaligned_indexes)!r}."
            )
        source_metadata = ImagePayloadMetadata.compose(
            source_payloads,
            source_metadata=tuple(
                image_payload_metadata(payload) for payload in source_payloads
            ),
        ).collapse_leading_plane_axis()
        return source_metadata


class SourceQualifiedInputPayloadMeasurementRecordMixin(
    CellProfilerMeasurementTableModule
):
    """Uses the aligned source axis of row-declared input artifacts."""

    @classmethod
    def measurement_record_source_image_name(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> str | None:
        if cls.measurement_record_source_names(rows):
            return None
        return super().measurement_record_source_image_name(request, rows)

    @staticmethod
    def measurement_record_source_names(rows: ColumnarRows) -> tuple[str, ...]:
        source_field = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
        return tuple(
            dict.fromkeys(
                str(row_mapping[source_field])
                for row_mapping in rows.iter_row_mappings()
                if source_field in row_mapping
            )
        )

    @classmethod
    def measurement_record_source_metadata(
        cls, request: CellProfilerOutputRecordRequest, rows: ColumnarRows
    ) -> ImagePayloadMetadata:
        from openhcs.core.artifacts import (
            ArtifactSpecCollection,
        )

        source_names = cls.measurement_record_source_names(rows)
        if not source_names:
            return super().measurement_record_source_metadata(request, rows)

        declared_inputs = ArtifactSpecCollection(request.callable_contract.artifact_inputs.specs)
        source_specs = []
        for source_name in source_names:
            spec = declared_inputs.by_name(source_name)
            if spec is None:
                raise ValueError(
                    f"{cls.__name__} emitted source-qualified measurement rows for "
                    f"{source_name!r}, but no declared input artifact has that name."
                )
            source_specs.append(spec)

        return request.measurement_source_metadata(tuple(source_specs))

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Source-qualified object rows keep their declared provenance."""
        return False
