"""Nominal artifact declarations authorities for CellProfiler modules."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)
from openhcs.core.callable_contract import (
    CallableContract,
    ImagePayloadConsumption,
)
from openhcs.core.config import StepSourceBindingsConfig
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    InputGroupLineageSourceRelation,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    ObjectMeasurementSubjectRelation,
    MeasurementsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.equivalence.policy import normalize_runtime_identifier
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.source_bindings import (
    SourceBindingsConfig,
)
from openhcs.interop.cellprofiler.settings_binder import (
    MeasurementFeatureSettingBinding,
    SettingToKeywordBinding,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    CurrentPayloadMeasurementRecordMixin,
    RelationshipMeasurementRecordRowsMixin,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
    )
    from openhcs.interop.cellprofiler.measurement_scope import (
        CellProfilerMeasurementTargetScope,
    )
    from openhcs.core.runtime_image_values import ImagePayloadMetadata
    from openhcs.core.runtime_tabular_values import ColumnarRows
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )
    from collections.abc import Callable
    from openhcs.core.steps.function_runtime import (
        RuntimeCallableKwargs,
        RuntimeFunctionOutput,
    )
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.module_measurement_features import ObjectCountFeature


class PlaneRuntimeArtifactModule(ABC):
    """Parent for modules that consume source-aligned runtime artifacts by plane."""


class ArtifactExportModule(CellProfilerModule):
    """Parent for modules whose materialized outputs expose selected artifacts."""

    @classmethod
    def finalize_artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        artifact_outputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Declare every exported input identity as output provenance."""

        outputs = super().finalize_artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
        )
        provenance = tuple(
            ArtifactSpecRelation(source=artifact_input.ref())
            for artifact_input in artifact_inputs.specs
        )
        return tuple(
            replace(output, relations=(*output.relations, *provenance))
            for output in outputs
        )


class PerObjectMeasurementExecutionModule(PlaneRuntimeArtifactModule):
    """Parent for modules invoked once per measured object set."""

    @classmethod
    def validate_callable_artifact_abi(
        cls,
        func: "Callable[..., RuntimeFunctionOutput]",
        contract: CallableContract,
    ) -> None:
        """Require the one measurement sink used by per-object execution."""

        super().validate_callable_artifact_abi(func, contract)
        measurement_outputs = contract.artifact_outputs.of_artifact_type(
            MeasurementsArtifactType
        )
        if len(measurement_outputs) != 1:
            raise ValueError(
                f"{cls.require_module_name()} per-object execution requires exactly "
                "one measurement output, got "
                f"{tuple(spec.ref() for spec in measurement_outputs)!r}."
            )

    @classmethod
    def validate_callable_object_inputs(
        cls,
        *,
        module_name: str,
        object_label_inputs: tuple[ArtifactSpec, ...],
        special_input_names: tuple[str, ...],
    ) -> None:
        """Validate the single object spec selected for each invocation."""

        if special_input_names or not object_label_inputs:
            super().validate_callable_object_inputs(
                module_name=module_name,
                object_label_inputs=object_label_inputs,
                special_input_names=special_input_names,
            )
            return
        for object_input in object_label_inputs:
            super().validate_callable_object_inputs(
                module_name=module_name,
                object_label_inputs=(object_input,),
                special_input_names=(),
            )

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: CallableContract,
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., Any]:
        """Select the object callable when the contract declares object inputs."""
        if cls.function_variants and contract.artifact_inputs.of_artifact_type(
            ObjectLabelsArtifactType
        ):
            return cls.require_callable(cls.function_variants[0])
        return super().resolve_function(
            module,
            contract=contract,
            source_bindings=source_bindings,
        )

    @classmethod
    def executes_per_object_measurements(
        cls,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        return bool(object_inputs)


class SourceSetupCellProfilerModule(CellProfilerModule, ABC):
    """Nominal root for setup modules that only contribute source declarations."""

    function_name = None

    @classmethod
    def emits_function_step(cls) -> bool:
        return False

    @classmethod
    @abstractmethod
    def contribute_source_bindings(
        cls,
        module: "ModuleBlock",
        config: "SourceBindingsConfig",
    ) -> "SourceBindingsConfig":
        """Return source declarations contributed by this setup module."""


class ObjectArtifactInputModule(CellProfilerModule):
    """Parent for modules that consume object-label artifacts through declared settings."""

    @classmethod
    def measurement_record_source_metadata(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: "ColumnarRows",
    ) -> "ImagePayloadMetadata":
        """Compose image-feature ownership with exact object-input provenance."""

        object_inputs = ArtifactSpecCollection(
            request.callable_contract.artifact_inputs.specs
        ).of_artifact_type(ObjectLabelsArtifactType)
        if not object_inputs:
            return super().measurement_record_source_metadata(request, rows)
        return request.measurement_source_metadata(object_inputs)


class ObjectArtifactOutputModule(
    CellProfilerModule,
):
    """Parent for modules that emit object-label artifacts through declared settings."""

    @classmethod
    def measurement_object_output_specs_for_request(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> tuple[ArtifactSpec, ...]:
        """Resolve exact measured object outputs from the compiled measurement spec."""

        measured_refs = tuple(
            dict.fromkeys(
                relation.source
                for relation in request.spec.relations
                if type(relation) is ArtifactSpecRelation
                and relation.source.plan_type is ArtifactOutputPlan
                and relation.source.artifact_type is ObjectLabelsArtifactType
            )
        )
        declared_outputs = request.callable_contract.artifact_outputs
        measured_outputs: list[ArtifactSpec] = []
        for ref in measured_refs:
            spec = declared_outputs.by_ref(ref)
            if spec is None:
                raise ValueError(
                    f"{cls.__name__} measurement relation refers to undeclared "
                    f"object output {ref!r}."
                )
            measured_outputs.append(spec)
        return tuple(measured_outputs)

    @classmethod
    def finalize_artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        artifact_outputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Attach exact object outputs to the module's measurement artifact."""

        outputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_outputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                artifact_outputs=artifact_outputs,
            )
        )
        object_outputs = outputs.of_artifact_type(ObjectLabelsArtifactType)
        if not object_outputs:
            return outputs.specs
        measurement_outputs = outputs.of_artifact_type(MeasurementsArtifactType)
        if not measurement_outputs:
            return outputs.specs
        if len(measurement_outputs) != 1:
            raise ValueError(
                f"{cls.__name__} requires one measurement output for its object "
                f"outputs, got {measurement_outputs!r}."
            )
        measurement = measurement_outputs[0]
        measurement = replace(
            measurement,
            relations=(
                *measurement.relations,
                *(ArtifactSpecRelation(source=spec.ref()) for spec in object_outputs),
            ),
        )
        resolved_outputs = tuple(
            measurement if spec.ref() == measurement.ref() else spec for spec in outputs
        )
        main_flow_outputs = cls.main_flow_output_specs(
            tuple(spec for spec in resolved_outputs if spec.participates_in_main_flow)
        )
        main_flow_refs = frozenset(spec.ref() for spec in main_flow_outputs)
        artifact_outputs = tuple(
            spec for spec in resolved_outputs if spec.ref() not in main_flow_refs
        )
        if any(
            spec.artifact_type is ObjectLabelsArtifactType for spec in main_flow_outputs
        ):
            return (*main_flow_outputs, *artifact_outputs)
        first_object_index = next(
            index
            for index, spec in enumerate(artifact_outputs)
            if spec.artifact_type is ObjectLabelsArtifactType
        )
        measurement_index = next(
            index
            for index, spec in enumerate(artifact_outputs)
            if spec.ref() == measurement.ref()
        )
        if measurement_index < first_object_index:
            return (*main_flow_outputs, *artifact_outputs)
        return (
            *main_flow_outputs,
            *(
                spec
                for spec in artifact_outputs[:first_object_index]
                if spec.ref() != measurement.ref()
            ),
            measurement,
            *(
                spec
                for spec in artifact_outputs[first_object_index:]
                if spec.ref() != measurement.ref()
            ),
        )

    @classmethod
    def measurement_record_rows(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> "ColumnarRows":
        """Emit canonical count and location rows for declared object outputs."""
        from openhcs.core.runtime_measurements import MeasurementRowAxisField
        from openhcs.core.runtime_tabular_values import FieldSpec
        from openhcs.core.measurement_row_materialization import (
            ConcatenatedColumnarRows,
            MeasurementProjectedColumnarRows,
            MeasurementRowsAxisProjection,
        )
        from openhcs.interop.cellprofiler.measurement_lookup import (
            CellProfilerMeasurementFeature,
        )
        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            ObjectLocationMeasurementRows,
        )

        row_batches = []
        for object_spec in cls.measurement_object_output_specs_for_request(request):
            location_rows = ObjectLocationMeasurementRows(
                request.artifact_output_value(object_spec),
                object_name=object_spec.name,
                domain_scope=request.object_label_output_domain_scope(),
            )
            count_feature = CellProfilerMeasurementFeature.object_count(
                object_spec.name
            ).name
            label_plane_domains = location_rows.label_plane_domains()
            count_fields = (
                FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
                FieldSpec(count_feature, ObjectCountFeature.measurement_dtype),
            )
            row_batches.extend(
                (
                    MeasurementProjectedColumnarRows(
                        {
                            MeasurementRowAxisField.SLICE_INDEX.value: tuple(
                                range(len(label_plane_domains))
                            ),
                            count_feature: tuple(
                                ObjectCountFeature.measurement_dtype(len(object_ids))
                                for _label_plane, object_ids in label_plane_domains
                            ),
                        },
                        fields=count_fields,
                    ),
                    location_rows.rows(),
                )
            )
        row_batches.append(super().measurement_record_rows(request))
        rows = ConcatenatedColumnarRows(tuple(row_batches))
        plane_index = request.adapter.request.plane_projection.plane_index
        if plane_index is not None:
            rows = MeasurementRowsAxisProjection.from_rows(
                rows
            ).project_runtime_slice_index(plane_index)
        return rows

    @classmethod
    def complete_table_measurement_rows(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: "ColumnarRows",
    ) -> "ColumnarRows":
        """Object outputs derive measurements from labels, not backend summaries."""
        from openhcs.core.measurement_row_materialization import (
            MeasurementSparseColumnarRows,
        )

        object_outputs = cls.measurement_object_output_specs_for_request(request)
        return super().complete_table_measurement_rows(
            request,
            (
                MeasurementSparseColumnarRows.from_rows((), fields=())
                if object_outputs
                else rows
            ),
        )

    @classmethod
    def measurement_record_object_name(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: "ColumnarRows",
    ) -> None:
        """Keep count rows image-scoped while location rows declare their owner."""
        del request, rows
        return None

    @classmethod
    def measurement_record_source_metadata(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: "ColumnarRows",
    ) -> "ImagePayloadMetadata":
        """Compose inherited ownership with exact object-output provenance."""

        object_outputs = cls.measurement_object_output_specs_for_request(request)
        if not object_outputs:
            return super().measurement_record_source_metadata(request, rows)
        return request.measurement_source_metadata(object_outputs)


class PriorMeasurementArtifactInputModule(CellProfilerModule):
    """Parent for modules that consume feature-addressed prior measurements."""

    @classmethod
    @lru_cache(maxsize=None)
    def measurement_feature_setting_bindings(
        cls,
    ) -> tuple[MeasurementFeatureSettingBinding, ...]:
        """Return feature-reference bindings owned by this module's MRO."""

        return tuple(
            dict.fromkeys(
                binding
                for owner_type in cls.__mro__
                for binding in owner_type.__dict__.values()
                if isinstance(binding, MeasurementFeatureSettingBinding)
            )
        )

    @classmethod
    def prior_measurement_feature_names(
        cls,
        module: "ModuleBlock",
    ) -> tuple[str, ...]:
        """Return ordered feature names that require prior measurement rows."""

        from openhcs.interop.cellprofiler.measurement_lookup import (
            CellProfilerMeasurementFeature,
            CellProfilerMeasurementFeatureKind,
        )

        return tuple(
            dict.fromkeys(
                feature_name
                for binding in cls.measurement_feature_setting_bindings()
                for feature_name in binding.feature_names(module)
                for feature in (CellProfilerMeasurementFeature.parse(feature_name),)
                if (
                    feature is not None
                    and feature.kind
                    is not CellProfilerMeasurementFeatureKind.OBJECT_COUNT
                )
            )
        )

    @classmethod
    def prior_measurement_lineage_refs(
        cls,
        *,
        feature_name: str,
        direct_inputs: ArtifactSpecCollection,
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[frozenset[ArtifactSpecRef], frozenset[ArtifactSpecRef]]:
        """Return object and source refs constraining one selected feature."""

        from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
        from openhcs.interop.cellprofiler.measurement_dialect import (
            CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        from openhcs.interop.cellprofiler.measurement_lookup import (
            CellProfilerMeasurementFeature,
        )

        object_refs = {
            spec.for_plan_type(ArtifactInputPlan).ref()
            for spec in direct_inputs.specs
            if spec.artifact_type is ObjectLabelsArtifactType
        }
        source_aliases = frozenset(
            alias
            for alias in MeasurementFeatureQuery(
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).source_candidates
        )
        feature_object_names = frozenset(
            feature.object_name
            for feature in (CellProfilerMeasurementFeature.parse(feature_name),)
            if feature is not None and feature.object_name is not None
        )
        source_refs: set[ArtifactSpecRef] = set()
        for spec in (
            *step_context.available_artifacts.specs,
            *step_context.main_flow_artifacts.specs,
        ):
            if spec.artifact_type is MeasurementsArtifactType:
                continue
            normalized_name = normalize_runtime_identifier(spec.name)
            ref = spec.for_plan_type(ArtifactInputPlan).ref()
            if spec.name in feature_object_names:
                object_refs.add(ref)
            if (
                normalized_name in source_aliases
                or normalized_name.replace("_", "") in source_aliases
            ):
                source_refs.add(ref)
        return frozenset(object_refs), frozenset(source_refs)

    @classmethod
    def prior_measurement_artifact_inputs(
        cls,
        module: "ModuleBlock",
        *,
        step_context: "ArtifactDeclarationStepContext",
        direct_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Select prior measurement artifacts by their declared lineage."""

        feature_names = cls.prior_measurement_feature_names(module)
        if not feature_names:
            return ()
        selected: list[ArtifactSpec] = []
        for feature_name in feature_names:
            object_refs, source_refs = cls.prior_measurement_lineage_refs(
                feature_name=feature_name,
                direct_inputs=direct_inputs,
                step_context=step_context,
            )
            feature_matches = tuple(
                producer
                for producer in step_context.available_artifact_producers
                if producer.spec.plan_type is ArtifactOutputPlan
                and producer.spec.artifact_type is MeasurementsArtifactType
                and producer.spec.measurement_feature_owner is not None
                and producer.spec.measurement_feature_owner.owns_measurement_feature_name(
                    feature_name
                )
                and (
                    not object_refs
                    or any(
                        relation.source in object_refs
                        for relation in producer.spec.relations
                    )
                )
                and (
                    not source_refs
                    or any(
                        relation.source in source_refs
                        for relation in producer.spec.relations
                    )
                )
            )
            if not feature_matches:
                raise ValueError(
                    f"{cls.__name__} cannot resolve prior measurement feature "
                    f"{feature_name!r} from declaration-owned artifact producers."
                )
            group_lineage_refs = source_refs or object_refs
            for producer in feature_matches:
                measurement_input = producer.spec.for_plan_type(ArtifactInputPlan)
                lineage_sources = tuple(
                    source
                    for source in producer.spec.group_scope_sources()
                    if source in group_lineage_refs
                )
                if len(lineage_sources) > 1:
                    raise ValueError(
                        f"{cls.__name__} prior measurement feature "
                        f"{feature_name!r} resolves multiple group-lineage sources "
                        f"{lineage_sources!r}."
                    )
                if lineage_sources:
                    measurement_input = measurement_input.with_group_scope_relation(
                        InputGroupLineageSourceRelation(lineage_sources[0])
                    )
                selected.append(measurement_input)
        selected = list(dict.fromkeys(selected))
        if selected:
            return tuple(selected)
        raise ValueError(
            f"{cls.__name__} cannot resolve prior measurement features "
            f"{feature_names!r} from declaration-owned artifact producers."
        )

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        direct_inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        prior_measurement_inputs = cls.prior_measurement_artifact_inputs(
            module,
            step_context=step_context,
            direct_inputs=direct_inputs,
        )
        return (
            *direct_inputs.specs,
            *prior_measurement_inputs,
        )


class MeasurementArtifactOutputModule(CellProfilerModule):
    """Parent for modules that emit the standard measurement artifact."""

    @classmethod
    def measurement_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Preserve all dependencies while inheriting the invocation domain."""

        inherited = super().measurement_output_relations(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        provenance_relations = tuple(
            ArtifactSpecRelation(source=artifact_input.ref())
            for artifact_input in artifact_inputs.specs
        )
        invocation_domain_inputs = cls.invocation_domain_inputs(
            cls.require_callable(invocation_key.function_name),
            artifact_inputs.specs,
        )
        return (
            *provenance_relations,
            *(
                GroupLineageSourceRelation(source=artifact_input.ref())
                for artifact_input in invocation_domain_inputs
            ),
            *inherited,
        )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        inherited_outputs = super().artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        return (
            *inherited_outputs,
            cls.measurement_output_artifact(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
        )


class SourceQualifiedMeasurementFeatureModule(CellProfilerModule):
    """Trait for modules whose emitted measurement features include source names."""

    @classmethod
    def source_qualified_measurement_feature_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeature], ...]:
        """Return finite module-owned feature enums carrying source identity."""

        return cls.measurement_feature_types()

    @classmethod
    def declared_source_qualified_measurement_feature_family_parts(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Declare every module-owned feature family as source-qualified."""

        families = cls.declared_measurement_feature_family_parts()
        if not families:
            raise TypeError(
                f"{cls.__name__} must declare at least one measurement feature "
                "family to use SourceQualifiedMeasurementFeatureModule."
            )
        return families

    @classmethod
    def source_qualified_measurement_category(cls) -> str:
        """Return the canonical CP category declared first by the module."""
        if not cls.measurement_category_prefixes:
            raise TypeError(
                f"{cls.__name__} must declare a measurement category prefix."
            )
        return "".join(
            part[:1].upper() + part[1:] for part in cls.measurement_category_prefixes[0]
        )

    @classmethod
    def source_qualified_feature_name(
        cls,
        field_name: str,
        source_image_name: str,
    ) -> str:
        """Resolve one exact raw field through module-owned feature members."""
        features = tuple(
            feature
            for feature_type in cls.source_qualified_measurement_feature_types()
            for feature in feature_type
        )
        exact_names = {
            "_".join(
                (
                    cls.source_qualified_measurement_category(),
                    feature.value,
                    source_image_name,
                )
            ): feature
            for feature in features
        }
        if field_name in exact_names:
            return field_name
        matches = tuple(
            feature
            for feature in features
            if feature.measurement_row_field_name == field_name
        )
        if len(matches) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one measurement feature for raw "
                f"field {field_name!r}, got {[feature.value for feature in matches]!r}."
            )
        return "_".join(
            (
                cls.source_qualified_measurement_category(),
                matches[0].value,
                source_image_name,
            )
        )


class SourceQualifiedWideMeasurementRowsModule(
    SourceQualifiedMeasurementFeatureModule,
):
    """Project raw wide rows using the module's source-qualified feature schema."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("module_name") is None:
            return
        projection_types = cls.measurement_row_projection_types()
        if projection_types:
            raise TypeError(
                f"{cls.__name__} cannot combine wide source-qualified field "
                "projection with module-owned measurement row projectors "
                f"{tuple(projection_type.__qualname__ for projection_type in projection_types)!r}."
            )

    @classmethod
    def project_measurement_record_rows(
        cls,
        rows: "ColumnarRows",
        *,
        source_image_name: str | None,
    ) -> "ColumnarRows":
        """Project wide raw fields to exact source-qualified CP features."""
        from types import MappingProxyType

        from openhcs.core.measurement_row_materialization import (
            MeasurementProjectedColumnarRows,
            is_structural_missing_measurement_cell,
            wide_measurement_feature_columns,
        )
        from openhcs.core.runtime_measurements import MeasurementRowAxisField
        from openhcs.core.runtime_tabular_values import FieldSpec
        from openhcs.core.runtime_tabular_values import ColumnarRows

        if not isinstance(rows, ColumnarRows):
            raise TypeError(
                f"{cls.__name__} requires schema-bearing ColumnarRows, "
                f"got {type(rows).__name__}."
            )
        columns = MappingProxyType(
            {str(column): rows.column_values(str(column)) for column in rows.columns}
        )
        feature_columns = wide_measurement_feature_columns(columns)
        if not feature_columns:
            return rows
        source_field = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
        source_names = tuple(
            dict.fromkeys(
                str(value)
                for value in columns.get(source_field, ())
                if value is not None
                and not is_structural_missing_measurement_cell(value)
                and str(value)
            )
        )
        if not source_names and source_image_name is not None:
            source_names = (source_image_name,)
        if len(source_names) != 1:
            raise ValueError(
                f"{cls.__name__} requires one exact source image for wide "
                f"measurement fields, got {source_names!r}."
            )
        source_image_name = source_names[0]
        feature_field_names = frozenset(
            field_name for field_name, _values in feature_columns
        )
        projected_columns = {
            (
                cls.source_qualified_feature_name(field_name, source_image_name)
                if field_name in feature_field_names
                else field_name
            ): values
            for field_name, values in columns.items()
        }
        if len(projected_columns) != len(columns):
            raise ValueError(
                f"{cls.__name__} source-qualified feature projection produced "
                "duplicate field identities."
            )
        projected_fields = tuple(
            FieldSpec(
                (
                    cls.source_qualified_feature_name(field.name, source_image_name)
                    if field.name in feature_field_names
                    else field.name
                ),
                dtype=field.dtype,
                required=field.required,
            )
            for field in rows.fields
        )
        return MeasurementProjectedColumnarRows(
            MappingProxyType(projected_columns),
            fields=projected_fields,
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=rows.object_row_identity,
        )


class ParentChildLineageArtifactOutputModule(
    RelationshipMeasurementRecordRowsMixin,
    CurrentPayloadMeasurementRecordMixin,
    CellProfilerModule,
):
    """Parent for modules whose lineage projects Parent/Children measurements."""

    @classmethod
    def finalize_artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        artifact_outputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Declare that parent-child measurement rows consume lineage outputs."""

        outputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_outputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                artifact_outputs=artifact_outputs,
            )
        )
        relationship_outputs = tuple(
            dict.fromkeys(
                spec
                for spec, declaration in outputs.relation_refs(
                    ObjectRelationshipDeclaration
                )
                if declaration.projects_parent_child_measurements()
            )
        )
        if not relationship_outputs:
            raise ValueError(
                f"{cls.__name__} must declare at least one output carrying a "
                "parent-child relationship."
            )
        measurement_outputs = outputs.of_artifact_type(MeasurementsArtifactType)
        if len(measurement_outputs) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one measurement output for "
                f"parent-child lineage, got {measurement_outputs!r}."
            )
        measurement = measurement_outputs[0]
        measurement = replace(
            measurement,
            relations=(
                *measurement.relations,
                *(
                    ArtifactSpecRelation(source=relationship_output.ref())
                    for relationship_output in relationship_outputs
                ),
            ),
        )
        return tuple(
            measurement if spec.ref() == measurement.ref() else spec
            for spec in outputs.specs
        )

    @classmethod
    def complete_table_measurement_rows(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: "ColumnarRows",
    ) -> "ColumnarRows":
        """Relationship rows replace the internal relationship summary carrier."""

        del rows
        from openhcs.core.measurement_row_materialization import (
            MeasurementSparseColumnarRows,
        )

        return super().complete_table_measurement_rows(
            request,
            MeasurementSparseColumnarRows.from_rows((), fields=()),
        )

    @classmethod
    def parent_child_relationship_output_artifact(
        cls,
        module: "ModuleBlock",
        *,
        step_context: "ArtifactDeclarationStepContext",
        parent: ArtifactSpec,
        child: ArtifactSpec,
        lineage_source: ArtifactSpec,
    ) -> ArtifactSpec:
        """Declare one parent-child relationship with exact group lineage."""

        declaration = ObjectRelationshipDeclaration.parent_child(
            source=parent.ref(),
            target=child.ref(),
            producer_module_number=module.module_num,
        )
        return ArtifactSpec.output(
            declaration.artifact_name(),
            ObjectLineageArtifactType,
            relations=(
                SourceStackLineageSourceRelation(source=lineage_source.ref()),
                declaration,
            ),
        )


class ObjectLineageTransformContractModule(
    PlaneRuntimeArtifactModule,
    ParentChildLineageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
):
    """Parent for one-object-input modules that emit object lineage."""

    input_objects_binding: ClassVar[SettingToKeywordBinding]
    output_objects_binding: ClassVar[SettingToKeywordBinding]

    @classmethod
    def artifact_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        binding: SettingToKeywordBinding,
        name: str,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Anchor transformed labels to their exact parent object artifact."""

        if binding is not cls.output_objects_binding:
            return super().artifact_output_relations(
                module,
                binding=binding,
                name=name,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                output_position=output_position,
            )
        parent_names = cls.artifact_names_for_binding(
            module,
            cls.input_objects_binding,
        )
        if len(parent_names) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one parent object artifact, "
                f"got {parent_names!r}."
            )
        parent = artifact_inputs.require_by_name_and_artifact_type(
            parent_names[0],
            ObjectLabelsArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=parent.ref()),)

    @classmethod
    def artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        outputs = ArtifactSpecCollection(
            ArtifactSpecCollection(
                super().artifact_contract_outputs(
                    module,
                    invocation_key=invocation_key,
                    step_context=step_context,
                    artifact_inputs=artifact_inputs,
                )
            ).unique(conflict_context=f"{cls.__name__} lineage outputs")
        )
        parent_names = cls.artifact_names_for_binding(
            module,
            cls.input_objects_binding,
        )
        if len(parent_names) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one parent object artifact, "
                f"got {parent_names!r}."
            )
        lineage_source = artifact_inputs.require_by_name_and_artifact_type(
            parent_names[0],
            ObjectLabelsArtifactType,
        )
        object_outputs = outputs.of_artifact_type(ObjectLabelsArtifactType)
        if len(object_outputs) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one object output, got "
                f"{object_outputs!r}."
            )
        relationship = cls.parent_child_relationship_output_artifact(
            module,
            step_context=step_context,
            parent=lineage_source,
            child=object_outputs[0],
            lineage_source=lineage_source,
        )
        measurement_outputs = outputs.of_artifact_type(MeasurementsArtifactType)
        if len(measurement_outputs) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one measurement output, "
                f"got {measurement_outputs!r}."
            )
        measurement_ref = measurement_outputs[0].ref()
        return tuple(
            item
            for output in outputs
            for item in (
                (output, relationship) if output.ref() == measurement_ref else (output,)
            )
        )


class ImageMeasurementInputModule(
    MeasurementArtifactOutputModule,
):
    """Parent for measurement modules that consume image measurement inputs."""

    image_measurement_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select images to measure",
        aliases=(
            "Select an image to measure",
            "Select the image to measure",
            "Select the images to measure",
        ),
    )
    image_measurement_binding = SettingToKeywordBinding.input(
        image_measurement_setting,
        ImageArtifactType,
        repeated=True,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        image_measurement_binding,
    )

    @classmethod
    def measurement_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Declare exact image subjects for image-scoped measurements."""

        object_inputs = artifact_inputs.of_artifact_type(ObjectLabelsArtifactType)
        image_subjects = (
            ()
            if cls.executes_per_object_measurements(object_inputs)
            else tuple(
                ImageMeasurementSubjectRelation(source=spec.ref())
                for spec in artifact_inputs.of_artifact_type(ImageArtifactType)
            )
        )
        return (
            *super().measurement_output_relations(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
            *image_subjects,
        )

    @classmethod
    def invocation_module_blocks(
        cls,
        module: "ModuleBlock",
    ) -> tuple["ModuleBlock", ...]:
        """Expose each natural measurement image as one public invocation."""

        blocks = super().invocation_module_blocks(module)
        if (
            CallableContract.from_callable(
                cls.require_callable()
            ).image_payload_consumption
            is ImagePayloadConsumption.COMPOSED
        ):
            return blocks
        (binding,) = cls.declared_artifact_bindings(
            plan_type=ArtifactInputPlan, artifact_type=ImageArtifactType
        )
        return cls.split_invocation_blocks_for_binding(
            blocks,
            binding,
        )

    @classmethod
    def executes_per_image_measurements(
        cls,
        func: "Callable[..., RuntimeFunctionOutput]",
        object_inputs: tuple[ArtifactSpec, ...],
        *,
        callable_contract: CallableContract,
    ) -> bool:
        """Execute natural image measurements once per declared image input."""

        del func
        return (
            not cls.executes_per_object_measurements(object_inputs)
            and not any(
                spec.parameter_name is not None
                for spec in callable_contract.artifact_inputs
            )
            and callable_contract.image_payload_consumption
            is not ImagePayloadConsumption.COMPOSED
        )


class ObjectMeasurementInputModule(
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
):
    """Parent for measurement modules that consume object-label measurement inputs."""

    object_measurement_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select object sets to measure",
        aliases=("Select objects to measure", "Select an object to measure"),
    )
    object_measurement_binding = SettingToKeywordBinding.input(
        object_measurement_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="labels",
        repeated=True,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        object_measurement_binding,
    )

    @classmethod
    def measurement_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Declare exact object subjects for object-scoped measurements."""

        return (
            *super().measurement_output_relations(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
            *(
                ObjectMeasurementSubjectRelation(source=spec.ref())
                for spec in artifact_inputs.of_artifact_type(ObjectLabelsArtifactType)
            ),
        )

    @classmethod
    def invocation_module_blocks(
        cls,
        module: "ModuleBlock",
    ) -> tuple["ModuleBlock", ...]:
        """Expose each measured object set as one scalar-label invocation."""

        return cls.split_invocation_blocks_for_binding(
            super().invocation_module_blocks(module),
            cls.object_measurement_binding,
        )


class ScopedMeasurementModule(
    ImageMeasurementInputModule, ObjectMeasurementInputModule
):
    """Module declaration parent for CellProfiler modules with target-scope settings."""

    measurement_scope_binding: ClassVar[SettingToKeywordBinding]
    measurement_scope_default: ClassVar["CellProfilerMeasurementTargetScope"]

    @classmethod
    def object_measurement_invocation_kwargs(
        cls,
        runtime_kwargs: "RuntimeCallableKwargs",
        *,
        include_image_measurements: bool,
    ) -> "RuntimeCallableKwargs":
        """Emit image-scope rows once while measuring every declared object set."""

        from openhcs.interop.cellprofiler.measurement_scope import (
            CellProfilerMeasurementTargetScope,
            coerce_cellprofiler_measurement_target_scope,
        )

        parameter_name = cls.measurement_scope_binding.require_parameter_name()
        scope = coerce_cellprofiler_measurement_target_scope(
            runtime_kwargs.get(parameter_name),
            cls.measurement_scope_default,
        )
        if (
            include_image_measurements
            or scope is not CellProfilerMeasurementTargetScope.BOTH
        ):
            return runtime_kwargs
        return {
            **runtime_kwargs,
            parameter_name: CellProfilerMeasurementTargetScope.OBJECT,
        }

    @classmethod
    def declared_setting_bindings(cls) -> tuple[SettingToKeywordBinding, ...]:
        """Compose the public measurement-scope behavior binding."""

        return (cls.measurement_scope_binding, *super().declared_setting_bindings())

    @classmethod
    def measurement_scope(
        cls, module: "ModuleBlock"
    ) -> "CellProfilerMeasurementTargetScope":
        """Return the declaration-owned typed scope from the parsed setting."""

        from openhcs.interop.cellprofiler.measurement_scope import (
            coerce_cellprofiler_measurement_target_scope,
        )
        from openhcs.interop.cellprofiler.setting_names import optional_setting_value

        raw_value = optional_setting_value(
            module, cls.measurement_scope_binding.setting_name
        )
        if raw_value is None:
            return cls.measurement_scope_default
        parser = cls.measurement_scope_binding.parse
        if parser is None:
            raise TypeError(
                f"{cls.__name__}.measurement_scope_binding must declare a parser."
            )
        return coerce_cellprofiler_measurement_target_scope(
            parser(raw_value), cls.measurement_scope_default
        )

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Declare object inputs only for scopes that measure objects."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        if not cls.measurement_scope(module).measurement_scope_selection.includes(
            MeasurementScope.OBJECT
        ):
            return tuple(
                binding
                for binding in bindings
                if not (
                    binding.require_artifact_plan_type() is ArtifactInputPlan
                    and binding.require_artifact_type() is ObjectLabelsArtifactType
                )
            )
        return bindings
