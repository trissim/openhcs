"""Generic CellProfiler measurement-record declarations."""

from __future__ import annotations

from abc import ABC


def measurement_record_for_module(
    request: CellProfilerOutputRecordRequest,
) -> CellProfilerMeasurementRecord:
    """Return the measurement record declared by the backend module."""
    from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
        CellProfilerMeasurementRecord,
    )
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    module_type = CellProfilerModule.for_module(request.module_name)
    if module_type is None:
        return DefaultMeasurementRecordModule.measurement_record(request)
    record = module_type.measurement_record(request)
    if not isinstance(record, CellProfilerMeasurementRecord):
        raise TypeError(
            f"{module_type.__name__}.measurement_record() must return "
            "a CellProfilerMeasurementRecord."
        )
    return record


class CellProfilerMeasurementRecordModule(ABC):
    """Generic measurement-record assembly contract for module declarations."""

    @classmethod
    def measurement_record(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementFieldSchema,
            CellProfilerMeasurementRecord,
        )

        rows = cls.measurement_record_rows(request)
        object_name = cls.measurement_record_object_name(request, rows)
        source_context = cls.measurement_record_source_context(request, rows)
        clear_source = cls.clear_source_when_rows_declare_object_name()
        if clear_source and CellProfilerMeasurementFieldSchema.rows_declare_object_name(rows):
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
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> list[CellProfilerKwargs]:
        del request
        return []

    @classmethod
    def measurement_record_object_name(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> str | None:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            _measurement_object_name,
        )

        del rows
        return _measurement_object_name(request.declared_input_specs)

    @classmethod
    def measurement_record_source_context(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
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
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> tuple[
        list[CellProfilerKwargs],
        tuple[CellProfilerMeasurementFieldSchema, ...] | None,
    ]:
        return rows, request.fields_for_rows(rows)

    @classmethod
    def clear_source_when_rows_declare_object_name(cls) -> bool:
        """Return whether row-owned object measurements clear table source context."""
        return True


class TableMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds raw CellProfiler measurement table rows."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
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
                        object_inputs[0],
                    ),
                    func=request.func,
                )
        return [
            *rows,
            *super().measurement_record_rows(request),
        ]


class RelationshipMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds relationship rows derived from the module relationship declaration."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
            RelationshipMeasurementRows,
        )

        return [
            *RelationshipMeasurementRows.for_request(request).rows(),
            *super().measurement_record_rows(request),
        ]


class OutputObjectThresholdMeasurementRecordRowsMixin(
    CellProfilerMeasurementRecordModule
):
    """Adds threshold rows qualified by the emitted object artifact."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            ThresholdMeasurementRows,
        )

        return [
            *ThresholdMeasurementRows(
                request.output_value,
                object_name=request.single_output_object_name(),
            ).rows(),
            *super().measurement_record_rows(request),
        ]


class ProducedImageThresholdMeasurementRecordRowsMixin(
    CellProfilerMeasurementRecordModule
):
    """Adds threshold rows qualified by the emitted image artifact."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            ThresholdMeasurementRows,
        )

        source_image = cls.primary_image_measurement_source(
            request
        ).require_produced_artifact()
        return [
            *ThresholdMeasurementRows(
                request.output_value,
                object_name=source_image.artifact_spec.name,
            ).rows(),
            *super().measurement_record_rows(request),
        ]


class ClassifyObjectsMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds object-classification rows."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> list[CellProfilerKwargs]:
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            ClassifyObjectsMeasurementRows,
            _measurement_object_name,
        )

        return [
            *ClassifyObjectsMeasurementRows(
                request.output_value,
                object_name=_measurement_object_name(request.declared_input_specs),
            ).rows(),
            *super().measurement_record_rows(request),
        ]


class AlignOutputMeasurementRecordRowsMixin(CellProfilerMeasurementRecordModule):
    """Adds Align rows for declared image outputs."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> list[CellProfilerKwargs]:
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpecCollection
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            AlignMeasurementRows,
        )

        output_names = tuple(
            spec.name
            for spec in ArtifactSpecCollection(request.declared_outputs).of_kind(
                ArtifactKind.IMAGE
            )
        )
        return [
            *AlignMeasurementRows(
                request.output_value,
                output_names=output_names,
            ).rows(),
            *super().measurement_record_rows(request),
        ]


class OutputObjectLocationMeasurementRecordRowsMixin(
    CellProfilerMeasurementRecordModule
):
    """Adds location rows for the emitted object artifact."""

    @classmethod
    def measurement_record_rows(
        cls,
        request: CellProfilerOutputRecordRequest,
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
        cls,
        request: CellProfilerOutputRecordRequest,
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
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> str | None:
        del request, rows
        return None


class NoSourceMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Suppresses image-source qualification for emitted measurement rows."""

    @classmethod
    def measurement_record_source_context(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
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
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
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
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
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


class ObjectLabelOutputSourceMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses the declared object-label output source payload as row provenance."""

    @classmethod
    def measurement_record_source_context(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )
        from openhcs.core.runtime_values import image_payload_metadata

        del rows
        source_payload = (
            request.output_values[request.single_output_object_name()]
            if request.adapter is None
            else request.object_label_output_source_payload()
        )
        return CellProfilerMeasurementSourceContext(
            source_metadata=image_payload_metadata(source_payload),
        )


class ProducedImageMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses a declared image output as measurement source."""

    @classmethod
    def primary_image_measurement_source(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerImageMeasurementSource:
        from openhcs.core.artifacts import ArtifactKind
        from openhcs.interop.cellprofiler.runtime.measurement_image_sources import (
            ProducedArtifactImageMeasurementSource,
            UnqualifiedRuntimeImageMeasurementSource,
        )
        from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
            ImagePayloadPure2DOutputAggregator,
        )

        image_specs = tuple(
            spec
            for spec in request.contract.declared_output_collection().of_kind(
                ArtifactKind.IMAGE
            )
            if spec.sidecar_role is None
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
            spec for spec in image_specs if spec.name in retained_image_names
        )
        if len(retained_specs) == 1:
            return ProducedArtifactImageMeasurementSource(retained_specs[0])
        raise ValueError(
            "Produced-image measurement ownership requires exactly one primary image "
            f"output spec, got {[spec.name for spec in image_specs]!r}."
        )

    @classmethod
    def measurement_record_source_context(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
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
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
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


class SourceNameOnlyMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Uses only the runtime source image name."""

    @classmethod
    def measurement_record_source_context(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> CellProfilerMeasurementSourceContext | None:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementSourceContext,
        )

        del rows
        return CellProfilerMeasurementSourceContext(
            source_image_name=request.source.source_image_name,
        )


class NoFieldsMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Suppresses explicit measurement field schema."""

    @classmethod
    def measurement_record_fields(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> tuple[
        list[CellProfilerKwargs],
        tuple[CellProfilerMeasurementFieldSchema, ...] | None,
    ]:
        del request
        return rows, None


class ColumnarFieldsMeasurementRecordMixin(CellProfilerMeasurementRecordModule):
    """Derives field schema from columnar materialization."""

    @classmethod
    def measurement_record_fields(
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> tuple[
        list[CellProfilerKwargs],
        tuple[CellProfilerMeasurementFieldSchema, ...] | None,
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
        cls,
        request: CellProfilerOutputRecordRequest,
        rows: list[CellProfilerKwargs],
    ) -> tuple[
        list[CellProfilerKwargs],
        tuple[CellProfilerMeasurementFieldSchema, ...] | None,
    ]:
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementFieldSchema,
        )

        del request
        return rows, CellProfilerMeasurementFieldSchema.from_rows(rows)


class DefaultMeasurementRecordModule(
    TableMeasurementRecordRowsMixin,
    RelationshipMeasurementRecordRowsMixin,
    CellProfilerMeasurementRecordModule,
):
    """Default measurement-record declaration used by most modules."""
