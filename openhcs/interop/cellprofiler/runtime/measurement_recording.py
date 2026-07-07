"""Generic CellProfiler measurement-record declarations."""

from __future__ import annotations
from abc import ABC
from collections.abc import Mapping
from dataclasses import asdict, fields as dataclass_fields, is_dataclass
import math
from typing import Annotated, ClassVar, get_args, get_origin, get_type_hints

from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    measurement_row_mapping,
)


def measurement_record_for_module(
    request: CellProfilerOutputRecordRequest,
) -> CellProfilerMeasurementRecord:
    """Return the measurement record declared by the backend module."""
    from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
        CellProfilerMeasurementRecord,
    )
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_module(request.module_name)
    if module_type is None:
        return DefaultMeasurementRecordModule.measurement_record(request)
    record = module_type.measurement_record(request)
    if not isinstance(record, CellProfilerMeasurementRecord):
        raise TypeError(
            f"{module_type.__name__}.measurement_record() must return a CellProfilerMeasurementRecord."
        )
    return record


def filter_measurement_record_rows_by_fields(
    rows: MeasurementRowsInput,
    excluded_fields: frozenset[str],
) -> MeasurementRowsInput:
    """Remove backend-only fields from measurement-record rows."""
    if not excluded_fields:
        return rows

    from openhcs.core.measurement_row_materialization import (
        MeasurementProjectedColumnarRows,
    )
    from openhcs.core.runtime_values import ColumnarRows

    if isinstance(rows, ColumnarRows):
        return MeasurementProjectedColumnarRows(
            {
                str(column): rows.column_values(column)
                for column in rows.columns
                if str(column) not in excluded_fields
            },
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
        )

    filtered_rows: list[CellProfilerKwargs] = []
    for row in rows:
        row_mapping = measurement_row_mapping(row)
        filtered_rows.append(
            {
                field_name: value
                for field_name, value in row_mapping.items()
                if field_name not in excluded_fields
            }
        )
    return filtered_rows


class CellProfilerMeasurementRecordModule(ABC):
    """Generic measurement-record assembly contract for module declarations."""

    measurement_record_excluded_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def measurement_record(
        cls, request: CellProfilerOutputRecordRequest
    ) -> CellProfilerMeasurementRecord:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementFieldSchema,
            CellProfilerMeasurementRecord,
        )

        rows = cls.measurement_record_rows(request)
        rows = cls.filter_measurement_record_rows(request, rows)
        object_name = cls.measurement_record_object_name(request, rows)
        source_context = cls.measurement_record_source_context(request, rows)
        clear_source = cls.clear_source_when_rows_declare_object_name()
        if (
            clear_source
            and CellProfilerMeasurementFieldSchema.rows_declare_object_name(rows)
        ):
            object_name = None
            source_context = source_context.without_source()
        rows, fields = cls.measurement_record_fields(request, rows)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=object_name,
            source_context=source_context,
            fields=fields,
            clear_source_when_rows_declare_object_name=clear_source,
        )

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> list[CellProfilerKwargs]:
        rows: list[CellProfilerKwargs] = []
        for projection_type in cls.measurement_row_projection_types():
            rows.extend(projection_type.for_request(cls, request).rows())
        return rows

    @classmethod
    def filter_measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: MeasurementRowsInput,
    ) -> MeasurementRowsInput:
        """Remove module-declared backend-only fields before CP materialization."""
        excluded_fields = cls.measurement_record_excluded_field_names(request)
        return filter_measurement_record_rows_by_fields(rows, excluded_fields)

    @classmethod
    def measurement_record_excluded_field_names(
        cls, request: CellProfilerOutputRecordRequest
    ) -> frozenset[str]:
        """Return backend-only fields declared by the record class or module."""
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

        module_type = CellProfilerModule.for_module(request.module_name)
        module_excluded_fields: frozenset[str] = frozenset()
        if module_type is not None and issubclass(
            module_type, CellProfilerMeasurementRecordModule
        ):
            module_excluded_fields = module_type.measurement_record_excluded_fields
        return frozenset(
            (
                *cls.measurement_record_excluded_fields,
                *module_excluded_fields,
            )
        )

    @classmethod
    def measurement_row_projection_types(
        cls,
    ) -> tuple[type["CellProfilerMeasurementRows"], ...]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            CellProfilerMeasurementRows,
        )

        return tuple(
            dict.fromkeys(
                projection_type
                for owner_type in cls.__mro__
                for projection_type in owner_type.__dict__.values()
                if isinstance(projection_type, type)
                and issubclass(projection_type, CellProfilerMeasurementRows)
                and projection_type is not CellProfilerMeasurementRows
            )
        )

    @classmethod
    def measurement_record_object_name(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> str | None:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            _measurement_object_name,
        )

        del rows
        return _measurement_object_name(request.declared_input_specs)

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )
        from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
            measurement_source_name_for_specs,
        )

        del rows
        source_image_name = request.source.source_image_name
        if source_image_name is None:
            source_image_name = measurement_source_name_for_specs(
                request.runtime_plan.primary_image_inputs
            )
        return CellProfilerMeasurementSourceContext(
            source_image_name=source_image_name,
            source_image_payload=request.source.payload,
        )

    @classmethod
    def measurement_record_fields(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> tuple[
        list[CellProfilerKwargs], tuple[CellProfilerMeasurementFieldSchema, ...] | None
    ]:
        return (rows, request.fields_for_rows(rows))

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Return whether row-owned object measurements clear table source context."""
        return True


class MeasurementFeatureRecord:
    """Dataclass mixin for CP feature rows derived from record fields."""

    measurement_value_field: ClassVar[MeasurementRowValueField] = (
        MeasurementRowValueField.RESULT_VALUE
    )


class FieldDerivedMeasurementFeatureModule(CellProfilerMeasurementRecordModule):
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
    def measurement_feature_stem(cls, field_name: str) -> str:
        token_aliases = dict(cls.measurement_feature_token_aliases)
        return "".join(
            token_aliases[part] if part in token_aliases else part[:1].upper() + part[1:]
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

    @staticmethod
    def measurement_record_axis_annotation(field_type: object) -> MeasurementRowAxisField | None:
        if get_origin(field_type) is not Annotated:
            return None
        for annotation in get_args(field_type)[1:]:
            if isinstance(annotation, MeasurementRowAxisField):
                return annotation
        return None

    @classmethod
    def measurement_feature_rows(
        cls,
        *,
        axis_values: Mapping[str, object],
        feature_values: Mapping[str, object],
        qualified_parts: tuple[object, ...] = (),
        value_field: MeasurementRowValueField,
    ) -> list[CellProfilerKwargs]:
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
    def measurement_record_axis_values(cls, record: MeasurementFeatureRecord) -> dict[str, object]:
        field_types = get_type_hints(type(record), include_extras=True)
        record_values = asdict(record)
        return {
            axis.value: record_values[field.name]
            for field in dataclass_fields(record)
            for axis in (
                cls.measurement_record_axis_annotation(field_types[field.name]),
            )
            if axis is not None
        }

    @classmethod
    def measurement_record_field_values(cls, record: MeasurementFeatureRecord) -> dict[str, object]:
        field_types = get_type_hints(type(record), include_extras=True)
        record_values = asdict(record)
        return {
            field.name: record_values[field.name]
            for field in dataclass_fields(record)
            if cls.measurement_record_axis_annotation(field_types[field.name]) is None
        }

    @classmethod
    def measurement_feature_row_fields(
        cls,
        record_type: type[MeasurementFeatureRecord],
    ) -> tuple[str, ...]:
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
                axis.value
                for field in dataclass_fields(record_type)
                for axis in (
                    cls.measurement_record_axis_annotation(field_types[field.name]),
                )
                if axis is not None
            ),
            record_type.measurement_value_field.value,
        )

    @classmethod
    def measurement_feature_rows_from_records(
        cls,
        records: tuple[MeasurementFeatureRecord, ...],
        *,
        qualified_parts: tuple[object, ...] = (),
    ) -> list[CellProfilerKwargs]:
        rows: list[CellProfilerKwargs] = []
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
    def source_qualified_measurement_feature_rows_from_records(
        cls,
        records: tuple[MeasurementFeatureRecord, ...],
    ) -> list[CellProfilerKwargs]:
        rows: list[CellProfilerKwargs] = []
        source_field = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
        for record in records:
            source_name = cls.measurement_record_axis_values(record).get(source_field)
            rows.extend(
                cls.measurement_feature_rows_from_records(
                    (record,),
                    qualified_parts=(source_name,),
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
    ) -> list[CellProfilerKwargs]:
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
    ) -> list[CellProfilerKwargs]:
        if not records:
            return []
        values_by_field: dict[str, list[float]] = {}
        for record in records:
            for field_name, value in cls.measurement_record_field_values(record).items():
                values_by_field.setdefault(field_name, []).append(float(value))
        mean_values = {
            field_name: (
                sum(finite_values) / len(finite_values)
                if finite_values
                else float("nan")
            )
            for field_name, values in values_by_field.items()
            for finite_values in (
                [value for value in values if math.isfinite(value)],
            )
        }
        return cls.mean_measurement_feature_rows(
            axis_values=axis_values,
            feature_values=mean_values,
            object_name=object_name,
            qualified_parts=qualified_parts,
            value_field=type(records[0]).measurement_value_field,
        )


class TableMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds raw CellProfiler measurement table rows."""

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            measurement_table_rows,
        )
        from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
            CellProfilerObjectMeasurementRowPolicy,
        )

        rows = measurement_table_rows(request.output_value)
        if issubclass(cls, CellProfilerObjectMeasurementRowPolicy):
            object_inputs = request.runtime_plan.object_label_inputs
            if len(object_inputs) == 1:
                row_policy = request.runtime_plan.object_measurement_row_policy
                rows = row_policy.complete_rows(
                    rows,
                    label_payload=request.object_label_source_payload_for_spec(
                        object_inputs[0]
                    ),
                    func=request.func,
                )
        return [*rows, *super().measurement_record_rows(request)]


class RelationshipMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds relationship rows derived from the module relationship declaration."""

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
            RelationshipMeasurementRows,
        )

        return [
            *RelationshipMeasurementRows.for_request(request).rows(),
            *super().measurement_record_rows(request),
        ]


class OutputObjectLocationMeasurementRecordRowsMixin(
    CellProfilerMeasurementRecordModule
):
    """Adds location rows for the emitted object artifact."""

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            ObjectLocationMeasurementRows,
        )

        object_name = request.single_output_object_name()
        return [
            *ObjectLocationMeasurementRows(
                request.output_values[object_name],
                object_name=object_name,
                domain_scope=request.object_label_output_domain_scope(),
            ).rows(),
            *super().measurement_record_rows(request),
        ]


class TrackingMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds tracking rows annotated through the runtime row policy."""

    @classmethod
    def measurement_record_rows(
        cls, request: CellProfilerOutputRecordRequest
    ) -> list[CellProfilerKwargs]:
        from openhcs.core.runtime_semantics import MeasurementScope
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            _measurement_object_name,
            measurement_table_rows,
        )

        row_policy = request.runtime_plan.object_measurement_row_policy
        return [
            *row_policy.annotate_record_rows(
                measurement_table_rows(request.output_value),
                object_name=_measurement_object_name(request.declared_input_specs),
                source_image_name=request.source.source_image_name
                or MeasurementScope.IMAGE.value,
            ),
            *super().measurement_record_rows(request),
        ]


class NoObjectNameMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Suppresses object ownership for emitted measurement rows."""

    @classmethod
    def measurement_record_object_name(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> str | None:
        del request, rows
        return None


class NoSourceMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Suppresses image-source qualification for emitted measurement rows."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del request, rows
        return CellProfilerMeasurementSourceContext()


class CurrentSourceMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses the current runtime source image and payload."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del rows
        return CellProfilerMeasurementSourceContext(
            source_image_name=request.source.source_image_name,
            source_image_payload=request.source.payload,
        )


class CurrentPayloadMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses the current runtime payload without an image-source name."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )
        from openhcs.core.runtime_values import image_payload_metadata

        del rows
        return CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=request.source.payload,
            source_metadata=image_payload_metadata(request.source.payload),
        )

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Current-payload measurements keep payload provenance for row-owned data."""
        return False


class ObjectLabelOutputSourceMeasurementRecordMixin(
    CellProfilerMeasurementRecordModule
):
    """Uses the declared object-label output source payload as row provenance."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )
        from openhcs.core.runtime_values import image_payload_metadata

        del rows
        source_payload = (
            request.output_values[request.single_output_object_name()]
            if request.adapter is None
            else request.runtime_plan.object_label_output_source_context_policy.source_context(
                request
            ).source_payload
        )
        return CellProfilerMeasurementSourceContext(
            source_metadata=image_payload_metadata(source_payload)
        )


class ProducedImageMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses a declared image output as measurement source."""

    @classmethod
    def primary_image_measurement_source(
        cls, request: CellProfilerOutputRecordRequest
    ) -> CellProfilerImageMeasurementSource:
        from openhcs.core.artifacts import ArtifactType, ImageArtifactType
        from openhcs.interop.cellprofiler.runtime.measurement_image_sources import (
            ProducedArtifactImageMeasurementSource,
            UnqualifiedRuntimeImageMeasurementSource,
        )
        from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
            ImagePayloadPure2DOutputAggregator,
        )

        image_specs = tuple(
            (
                spec
                for spec in request.contract.declared_output_collection().of_artifact_type(
                    ImageArtifactType
                )
                if spec.sidecar_role is None
            )
        )
        if not image_specs:
            return UnqualifiedRuntimeImageMeasurementSource()
        if len(image_specs) == 1:
            return ProducedArtifactImageMeasurementSource(image_specs[0])
        accepted_image_output_types = (
            ImagePayloadPure2DOutputAggregator.accepted_output_types()
        )
        retained_image_names = {
            name
            for name, value in request.output_values.items()
            if accepted_image_output_types
            and isinstance(value, accepted_image_output_types)
        }
        retained_specs = tuple(
            (spec for spec in image_specs if spec.name in retained_image_names)
        )
        if len(retained_specs) == 1:
            return ProducedArtifactImageMeasurementSource(retained_specs[0])
        raise ValueError(
            f"Produced-image measurement ownership requires exactly one primary image output spec, got {[spec.name for spec in image_specs]!r}."
        )

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del rows
        source_image = cls.primary_image_measurement_source(request)
        return CellProfilerMeasurementSourceContext(
            source_image_name=source_image.source_image_name(request),
            source_image_payload=source_image.source_image_payload(request),
        )


class ProducedImagePayloadMeasurementRecordMixin(ProducedImageMeasurementRecordMixin):
    """Uses a declared image output payload without an image-source name."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del rows
        source_image = cls.primary_image_measurement_source(
            request
        ).require_produced_artifact()
        return CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=source_image.source_image_payload(request),
        )


class DeclaredImageOutputPayloadMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses current payload provenance renamed to declared image outputs."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.core.artifacts import ImageArtifactType
        from openhcs.core.runtime_values import image_payload_metadata
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del rows
        image_output_names = tuple(
            spec.name
            for spec in request.contract.declared_output_collection().of_artifact_type(
                ImageArtifactType
            )
            if spec.sidecar_role is None
        )
        source_metadata = image_payload_metadata(request.source.payload)
        if image_output_names:
            source_metadata = source_metadata.with_source_provenance(
                source_metadata.source_provenance.with_source_image_names(
                    image_output_names
                )
            )
        return CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=request.source.payload,
            source_metadata=source_metadata,
        )


class SourceQualifiedInputPayloadMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses row-declared input artifacts as source-qualified measurement provenance."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
        from openhcs.core.runtime_semantics import MeasurementRowAxisField
        from openhcs.core.runtime_values import (
            ImagePayloadMetadataCompositionRequest,
            image_payload_metadata,
        )
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        source_field = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
        source_names = tuple(
            dict.fromkeys(
                str(row_mapping[source_field])
                for row in rows
                for row_mapping in (measurement_row_mapping(row),)
                if source_field in row_mapping
            )
        )
        if not source_names:
            return super().measurement_record_source_context(request, rows)

        declared_inputs = request.contract.declared_input_collection()
        source_payloads = []
        source_metadata = []
        for source_name in source_names:
            spec = declared_inputs.by_name(source_name)
            if spec is None:
                raise ValueError(
                    f"{cls.__name__} emitted source-qualified measurement rows for "
                    f"{source_name!r}, but no declared input artifact has that name."
                )
            if spec.artifact_type is ImageArtifactType:
                payload = request.input_image_source_payload(spec)
            elif spec.artifact_type is ObjectLabelsArtifactType:
                payload = request.object_label_source_payload_for_spec(spec)
            else:
                raise ValueError(
                    f"{cls.__name__} emitted source-qualified measurement rows for "
                    f"{source_name!r}, but declared input {spec.name!r} has unsupported "
                    f"artifact type {spec.artifact_type.value!r}."
                )
            if payload is None:
                raise ValueError(
                    f"{cls.__name__} could not resolve source payload for declared "
                    f"input artifact {source_name!r}."
                )
            metadata = image_payload_metadata(payload)
            source_payloads.append(payload)
            source_metadata.append(
                metadata.with_source_provenance(
                    metadata.source_provenance.with_source_image_names((source_name,))
                )
            )

        return CellProfilerMeasurementSourceContext(
            source_image_name=None,
            source_image_payload=request.source.payload,
            source_metadata=ImagePayloadMetadataCompositionRequest(
                source_payloads,
                source_metadata_override=source_metadata,
            ).metadata(),
        )

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Source-qualified object rows keep their declared provenance."""
        return False


class SourceNameOnlyMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses only the runtime source image name."""

    @classmethod
    def measurement_record_source_context(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del rows
        return CellProfilerMeasurementSourceContext(
            source_image_name=request.source.source_image_name
        )


class NoFieldsMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Suppresses explicit measurement field schema."""

    @classmethod
    def measurement_record_fields(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> tuple[
        list[CellProfilerKwargs], tuple[CellProfilerMeasurementFieldSchema, ...] | None
    ]:
        del request
        return (rows, None)


class ColumnarFieldsMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Derives field schema from columnar materialization."""

    @classmethod
    def measurement_record_fields(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> tuple[
        list[CellProfilerKwargs], tuple[CellProfilerMeasurementFieldSchema, ...] | None
    ]:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            MeasurementRowColumnarMaterialization,
        )

        del request
        return MeasurementRowColumnarMaterialization.from_rows(rows).table()


class FieldsFromRowsMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Derives field schema directly from declared row carriers."""

    @classmethod
    def measurement_record_fields(
        cls, request: CellProfilerOutputRecordRequest, rows: list[CellProfilerKwargs]
    ) -> tuple[
        list[CellProfilerKwargs], tuple[CellProfilerMeasurementFieldSchema, ...] | None
    ]:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementFieldSchema,
        )

        del request
        return (rows, CellProfilerMeasurementFieldSchema.from_rows(rows))


class DefaultMeasurementRecordModule(
    TableMeasurementRecordRowsMixin,
    RelationshipMeasurementRecordRowsMixin,
    CellProfilerMeasurementRecordModule,
):
    """Default measurement-record declaration used by most modules."""
