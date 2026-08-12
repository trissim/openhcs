"""Object-filtering semantics for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
    CellProfilerMeasurementFeatureKind,
)
from openhcs.interop.cellprofiler.setting_names import (
    RepeatedSettingSequence,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
)
from openhcs.interop.cellprofiler.settings_binder import parse_cellprofiler_bool
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import inspect
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.memory.decorators import numpy
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.measurement_feature_queries import (
    measurement_values_for_feature,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    MeasurementsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.runtime_object_label_domains import (
    DenseObjectLabelExtentDomainDeclaration,
    ObjectLabelIdDomainStrategy,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter
from openhcs.core.runtime_measurements import (
    ObjectLabelMeasurementValues,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    PlaneRuntimeArtifactModule,
    PriorMeasurementArtifactInputModule,
    ParentChildLineageArtifactOutputModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_names,
    setting_values,
)
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.settings_binder import (
    MeasurementFeatureSettingBinding,
    SettingToKeywordBinding,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    ObjectLabelsInputBindingMixin,
    ObjectLabelsRuntimeParameter,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectMeasurementVectorBinding,
)
from openhcs.core.steps.function_runtime import (
    RuntimeCallableArgument,
    RuntimeCallableKwargs,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)

if TYPE_CHECKING:
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting


class _FilterMeasurementValuesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound selected measurement vector."""

    parameter_name = "measurement_values"
    annotation_type = np.ndarray | None
    parameter_default = None


class _FilterMeasurementTablesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound measurement tables used by multi-feature filters."""

    parameter_name = "measurement_tables"
    annotation_type = tuple[MeasurementTable, ...]
    parameter_default = ()


class _FilterEnclosingLabelsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound enclosing-object labels."""

    parameter_name = "enclosing_object_labels"
    annotation_type = ObjectLabelValue | None
    parameter_default = None


class _FilterParentChildRelationshipsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound measurement relationship rows."""

    parameter_name = "parent_child_relationships"
    annotation_type = tuple[
        ObjectRelationship | DirectedObjectRelationshipPayload,
        ...,
    ]
    parameter_default = ()


class FilterObjectsRemovedObjectSourceRelation(SourceStackLineageSourceRelation):
    """Mark the complementary FilterObjects output derived from one input set."""

    relation_key: ClassVar[str] = "filter_objects_removed_object_source"
    target_artifact_type = ObjectLabelsArtifactType

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "FilterObjects removed-object output requires an object-label "
                f"source, got {self.source.artifact_type.value}:{self.source.name}."
            )


@dataclass(frozen=True, slots=True)
class FilterObjectsRuntimeInputPlan:
    """Runtime object-label partition for one FilterObjects invocation."""

    object_specs: tuple[ArtifactSpec, ...]
    enclosing_spec: ArtifactSpec | None
    measurement_features: tuple[str, ...]
    measurement_relationship_specs: tuple[ArtifactSpec, ...] = ()

    @classmethod
    def from_request(
        cls, request: RuntimeInputBindingRequest
    ) -> "FilterObjectsRuntimeInputPlan":
        declared_inputs = request.declared_inputs
        object_inputs = declared_inputs.of_artifact_type(ObjectLabelsArtifactType)
        raw_measurement_features = request.kwargs.get(
            FilterObjectsModule.measurement_feature_binding.require_parameter_name()
        )
        measurement_features = (
            ()
            if raw_measurement_features is None
            else tuple(str(value) for value in raw_measurement_features)
        )
        output_specs = request.adapter.request.require_callable_contract().artifact_outputs
        output_objects = output_specs.of_artifact_type(ObjectLabelsArtifactType)
        removed_output_relations = tuple(
            (spec, relation)
            for spec, relation in output_specs.relation_refs(
                FilterObjectsRemovedObjectSourceRelation
            )
            if spec.artifact_type is ObjectLabelsArtifactType
        )
        if len(removed_output_relations) > 1:
            raise ValueError(
                "FilterObjects declares multiple removed-object outputs: "
                f"{tuple(spec.name for spec, _relation in removed_output_relations)!r}."
            )
        removed_refs = {spec.ref() for spec, _relation in removed_output_relations}
        object_specs = cls._object_specs_from_output_lineage(
            object_inputs,
            tuple(spec for spec in output_objects if spec.ref() not in removed_refs),
            module_name=request.adapter.request.require_callable_contract().module_name,
        )
        if (
            removed_output_relations
            and removed_output_relations[0][1].source != object_specs[0].ref()
        ):
            removed_spec, removed_relation = removed_output_relations[0]
            raise ValueError(
                f"FilterObjects removed output {removed_spec.name!r} derives from "
                f"{removed_relation.source.name!r}, but the primary filtered input "
                f"is {object_specs[0].name!r}."
            )
        filtered_refs = {spec.ref() for spec in object_specs}
        remaining_object_specs = tuple(
            spec for spec in object_inputs if spec.ref() not in filtered_refs
        )
        enclosing_spec = cls._enclosing_object_spec(
            remaining_object_specs,
            module_name=request.adapter.request.require_callable_contract().module_name,
        )
        measurement_relationship_specs: list[ArtifactSpec] = []
        relationship_declarations = declared_inputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if object_specs:
            for (
                child_object_name
            ) in CellProfilerMeasurementFeature.child_count_object_names(
                measurement_features
            ):
                child_spec = declared_inputs.by_name_and_artifact_type(
                    child_object_name,
                    ObjectLabelsArtifactType,
                )
                relationship = None
                if child_spec is not None:
                    matches = tuple(
                        spec
                        for spec, declaration in relationship_declarations
                        if declaration.source == object_specs[0].ref()
                        and declaration.target == child_spec.ref()
                    )
                    if len(matches) > 1:
                        raise ValueError(
                            "FilterObjects child-count endpoints select multiple "
                            "relationship inputs: "
                            f"{tuple(spec.name for spec in matches)!r}."
                        )
                    relationship = matches[0] if matches else None
                if relationship is not None:
                    measurement_relationship_specs.append(relationship)
        return cls(
            object_specs=object_specs,
            enclosing_spec=enclosing_spec,
            measurement_features=measurement_features,
            measurement_relationship_specs=ArtifactSpecCollection(
                measurement_relationship_specs
            ).unique(conflict_context="CellProfiler input spec"),
        )

    @classmethod
    def _object_specs_from_output_lineage(
        cls,
        object_inputs: tuple[ArtifactSpec, ...],
        output_specs: tuple[ArtifactSpec, ...],
        *,
        module_name: str,
    ) -> tuple[ArtifactSpec, ...]:
        output_objects = ArtifactSpecCollection(output_specs).of_artifact_type(
            ObjectLabelsArtifactType
        )
        if not output_objects:
            raise ValueError(
                f"{module_name} requires declared object output lineage to bind object inputs."
            )
        object_specs: list[ArtifactSpec] = []
        for output_spec in output_objects:
            source = cls._single_object_input_lineage_source(output_spec)
            if source is None:
                raise ValueError(
                    f"{module_name} object output {output_spec.name!r} has no "
                    "GroupLineageSourceRelation to an input object artifact."
                )
            source_spec = ArtifactSpecCollection(object_inputs).by_ref(source)
            if source_spec is None:
                raise ValueError(
                    f"{module_name} object output {output_spec.name!r} references "
                    f"input object {source.name!r}, but the runtime contract declares "
                    f"object inputs={[spec.name for spec in object_inputs]!r}."
                )
            object_specs.append(source_spec)
        return tuple(object_specs)

    @staticmethod
    def _single_object_input_lineage_source(
        spec: ArtifactSpec,
    ) -> ArtifactSpecRef | None:
        sources = tuple(
            relation.source
            for relation in spec.relations
            if isinstance(relation, GroupLineageSourceRelation)
            and relation.source.artifact_type is ObjectLabelsArtifactType
        )
        if len(sources) > 1:
            raise ValueError(
                f"Artifact {spec.name!r} has multiple object lineage sources."
            )
        return sources[0] if sources else None

    @staticmethod
    def _enclosing_object_spec(
        remaining_object_specs: tuple[ArtifactSpec, ...],
        *,
        module_name: str,
    ) -> ArtifactSpec | None:
        match remaining_object_specs:
            case ():
                return None
            case (enclosing_spec,):
                return enclosing_spec
            case _:
                raise ValueError(
                    f"{module_name} cannot infer a unique enclosing object input "
                    f"from extra object inputs={[spec.name for spec in remaining_object_specs]!r}."
                )

    @property
    def primary_object_spec(self) -> ArtifactSpec | None:
        if not self.object_specs:
            return None
        return self.object_specs[0]

    def bind_measurement_inputs(
        self, request: RuntimeInputBindingRequest
    ) -> dict[str, RuntimeCallableArgument]:
        """Return FilterObjects measurement bindings owned by this runtime plan."""
        scoped_request = request.with_object_inputs(self.object_specs)
        measurement_values = self.measurement_vector(scoped_request)
        if measurement_values is not None:
            return {
                _FilterMeasurementValuesRuntimeParameter.require_parameter_name(): measurement_values
            }
        measurement_tables = self.measurement_tables(scoped_request)
        if measurement_tables is None:
            return {}
        return {
            _FilterMeasurementTablesRuntimeParameter.require_parameter_name(): measurement_tables
        }

    def measurement_vector(
        self, request: RuntimeInputBindingRequest
    ) -> RuntimeCallableArgument | None:
        object_spec = self.primary_object_spec
        if object_spec is None:
            return None
        feature_names = self.measurement_features
        if len(feature_names) != 1:
            return None
        feature_name = str(feature_names[0])
        labels = request.label_payload_for(object_spec)
        return (
            CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=feature_name,
                labels=labels,
            )
            .vector()
            .runtime_value
        )

    def measurement_tables(
        self, request: RuntimeInputBindingRequest
    ) -> tuple[MeasurementTable, ...] | None:
        object_spec = self.primary_object_spec
        if object_spec is None:
            return None
        feature_names = self.measurement_features
        if not feature_names:
            return None
        labels = request.label_payload_for(object_spec)
        tables_by_identity: dict[int, MeasurementTable] = {}
        for feature_name in feature_names:
            binding = CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=str(feature_name),
                labels=labels,
            )
            tables = binding.measurement_tables(request.adapter, match_group=False)
            for table in tables:
                table_identity = id(table)
                if table_identity not in tables_by_identity:
                    tables_by_identity[table_identity] = table
        if not tables_by_identity:
            raise ValueError(
                f"{request.adapter.request.require_callable_contract().module_name} declared measurement inputs do not "
                f"provide features {feature_names!r} for object {object_spec.name!r}."
            )
        return tuple(tables_by_identity.values())


class FilterObjectsInputPolicy(ObjectLabelsInputBindingMixin):
    """Bind ordered primary/additional object rows for FilterObjects."""

    supported_non_object_input_kinds = frozenset(
        {MeasurementsArtifactType, ObjectLineageArtifactType}
    )

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        plan = FilterObjectsRuntimeInputPlan.from_request(request)
        bound = super().bind_runtime_inputs(
            request.with_object_inputs(plan.object_specs)
        )
        bound.update(plan.bind_measurement_inputs(request))
        measurement_tables = tuple(
            bound.get(_FilterMeasurementTablesRuntimeParameter.require_parameter_name())
            or ()
        )
        measurement_values = bound.get(
            _FilterMeasurementValuesRuntimeParameter.require_parameter_name()
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "filterobjects_bound_measurements",
            0.0,
            module=request.adapter.request.require_callable_contract().module_name,
            table_count=len(measurement_tables),
            has_measurement_values=measurement_values is not None,
            measurement_values_type=(
                "none"
                if measurement_values is None
                else type(measurement_values).__name__
            ),
            measurement_features=len(plan.measurement_features),
        )
        if plan.enclosing_spec is not None:
            parameter_name = (
                _FilterEnclosingLabelsRuntimeParameter.require_parameter_name()
            )
            bound[parameter_name] = request.label_argument_for(
                plan.enclosing_spec, parameter_name
            )
        if plan.measurement_relationship_specs:
            bound[
                _FilterParentChildRelationshipsRuntimeParameter.require_parameter_name()
            ] = tuple(
                (
                    request.current_plane_relationship_for(relationship_spec)
                    for relationship_spec in plan.measurement_relationship_specs
                )
            )
        return bound


class FilterObjectsModule(
    PlaneRuntimeArtifactModule,
    FilterObjectsInputPolicy,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    ParentChildLineageArtifactOutputModule,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
):
    module_name = "FilterObjects"
    function_name = "filter_objects"
    validated = True
    confidence = 1.0
    input_setting = SettingNameFamily(
        "Select the object to filter",
        aliases=("Select the objects to filter", "Select the input objects"),
    )
    output_setting = "Name the output objects"
    mode_setting = SettingNameFamily(
        "Filter using classifier rules or measurements?",
        aliases=("Select the filtering mode",),
    )
    method_setting = "Select the filtering method"
    measurement_setting = "Select the measurement to filter by"
    use_minimum_setting = "Filter using a minimum measurement value?"
    minimum_setting = "Minimum value"
    use_maximum_setting = "Filter using a maximum measurement value?"
    maximum_setting = "Maximum value"
    measurement_feature_binding = MeasurementFeatureSettingBinding(
        measurement_setting,
        "measurement_features",
    )
    measurement_minimum_binding = SettingToKeywordBinding(
        minimum_setting,
        "measurement_min_values",
    )
    measurement_maximum_binding = SettingToKeywordBinding(
        maximum_setting,
        "measurement_max_values",
    )
    measurement_use_minimum_binding = SettingToKeywordBinding(
        use_minimum_setting,
        "measurement_use_minimum",
    )
    measurement_use_maximum_binding = SettingToKeywordBinding(
        use_maximum_setting,
        "measurement_use_maximum",
    )
    measurement_rule_bindings = (
        measurement_feature_binding,
        measurement_minimum_binding,
        measurement_maximum_binding,
        measurement_use_minimum_binding,
        measurement_use_maximum_binding,
    )
    additional_input_setting = "Select additional object to relabel"
    additional_output_setting = "Name the relabeled objects"
    enclosing_object_setting = "Select the objects that contain the filtered objects"
    per_object_assignment_setting = "Assign overlapping child to"
    measurement_count_setting = "Measurement count"
    additional_object_count_setting = "Additional object count"
    classifier_location_setting = "Select the location of the rules or classifier file"
    classifier_file_setting = "Rules or classifier file name"
    classifier_class_setting = "Class number"
    keep_removed_objects_setting = "Keep removed objects as a separate set?"
    removed_objects_name_setting = "Name the objects removed by the filter"
    ignored_settings = (
        classifier_location_setting,
        classifier_file_setting,
        classifier_class_setting,
    )
    input_binding = SettingToKeywordBinding.input(
        input_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name=ObjectLabelsRuntimeParameter.require_parameter_name(),
    )
    additional_input_binding = SettingToKeywordBinding.input(
        additional_input_setting,
        ObjectLabelsArtifactType,
        repeated=True,
        runtime_parameter_name=ObjectLabelsRuntimeParameter.require_parameter_name(),
    )
    enclosing_object_binding = SettingToKeywordBinding.input(
        enclosing_object_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name=(
            _FilterEnclosingLabelsRuntimeParameter.require_parameter_name()
        ),
    )
    output_binding = SettingToKeywordBinding.output(
        output_setting, ObjectLabelsArtifactType
    )
    additional_output_binding = SettingToKeywordBinding.output(
        additional_output_setting, ObjectLabelsArtifactType, repeated=True
    )
    removed_output_binding = SettingToKeywordBinding.output(
        removed_objects_name_setting, ObjectLabelsArtifactType
    )
    additional_object_count_binding = SettingToKeywordBinding(
        additional_object_count_setting,
        "additional_object_count",
        int,
    )
    emit_removed_objects_binding = SettingToKeywordBinding(
        keep_removed_objects_setting,
        "emit_removed_objects",
        parse_cellprofiler_bool,
    )
    setting_bindings = (
        input_binding,
        additional_input_binding,
        enclosing_object_binding,
        output_binding,
        additional_output_binding,
        removed_output_binding,
        SettingToKeywordBinding(mode_setting, "mode"),
        SettingToKeywordBinding(method_setting, "filter_method"),
        SettingToKeywordBinding(
            per_object_assignment_setting,
            "per_object_assignment",
        ),
        additional_object_count_binding,
        emit_removed_objects_binding,
    )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        """Expose only the object topology declared by this module value."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        additional_objects = cls.additional_object_count(module)
        enclosing_objects = cls.filter_method(module).requires_enclosing_object
        removed_objects = cls.emits_removed_objects(module)
        return tuple(
            binding
            for binding in bindings
            if additional_objects
            or binding
            not in (cls.additional_input_binding, cls.additional_output_binding)
            if enclosing_objects or binding is not cls.enclosing_object_binding
            if removed_objects or binding is not cls.removed_output_binding
        )

    @classmethod
    def topology_parameter_value(
        cls,
        module: "ModuleBlock",
        binding: SettingToKeywordBinding,
    ) -> object:
        """Resolve one topology value from its setting or public callable default."""

        values = setting_values(module, binding.setting_name)
        if len(values) > 1:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares "
                f"{len(values)} rows for scalar FilterObjects topology setting "
                f"{setting_names(binding.setting_name)[0]!r}."
            )
        if values:
            return values[0] if binding.parse is None else binding.parse(values[0])
        parameter_name = binding.require_parameter_name()
        parameter = inspect.signature(cls.require_callable()).parameters[parameter_name]
        if parameter.default is inspect.Parameter.empty:
            raise ValueError(
                f"Module {module.name}({module.module_num}) omits required "
                f"FilterObjects topology setting "
                f"{setting_names(binding.setting_name)[0]!r}."
            )
        return parameter.default

    @classmethod
    def additional_object_count(cls, module: "ModuleBlock") -> int:
        """Return the exact public additional-object topology cardinality."""

        values = setting_values(
            module,
            cls.additional_object_count_binding.setting_name,
        )
        if len(values) != 1:
            raise ValueError(
                f"Module {module.name}({module.module_num}) requires exactly one "
                f"FilterObjects {cls.additional_object_count_setting!r} setting "
                f"row, got {len(values)}."
            )
        parser = cls.additional_object_count_binding.parse
        if parser is None:
            raise TypeError(
                "FilterObjects additional-object count binding requires a parser."
            )
        count = parser(values[0])
        if not isinstance(count, int):
            raise TypeError(
                "FilterObjects additional-object count must resolve to int, got "
                f"{type(count).__name__}."
            )
        if count < 0:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares a negative "
                f"FilterObjects additional-object count {count}."
            )
        return count

    @classmethod
    def emits_removed_objects(cls, module: "ModuleBlock") -> bool:
        """Return whether the public invocation declares a removed-object output."""

        value = cls.topology_parameter_value(
            module,
            cls.emit_removed_objects_binding,
        )
        if not isinstance(value, bool):
            raise TypeError(
                "FilterObjects emit_removed_objects must resolve to bool, got "
                f"{type(value).__name__}."
            )
        return value

    @classmethod
    def removed_object_name(cls, module: "ModuleBlock") -> str | None:
        """Return the declared complementary output name when enabled."""

        if not cls.emits_removed_objects(module):
            return None
        name = required_setting_value(module, cls.removed_objects_name_setting)
        cls.SymbolRequirement(name, cls.removed_objects_name_setting).validate(module)
        return name

    @dataclass(frozen=True, slots=True)
    class SymbolRequirement:
        """Fail-loud FilterObjects symbol-setting validation."""

        value: str
        setting_name: str | SettingNameFamily

        def validate(self, module: "ModuleBlock") -> None:
            if normalized_symbol_name(self.value) is not None:
                return
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an empty FilterObjects symbol in setting {self.setting_name!r}."
            )

    @dataclass(frozen=True, slots=True)
    class ObjectPair:
        """Shared input/output object-name pair for FilterObjects rows."""

        input_object_name: str
        output_object_name: str

    @dataclass(frozen=True, slots=True)
    class AdditionalObjectRow(ObjectPair):
        """One additional object set relabeled using the primary filter mask."""

        def validated(
            self, module: "ModuleBlock"
        ) -> "FilterObjectsModule.AdditionalObjectRow":
            FilterObjectsModule.SymbolRequirement(
                self.input_object_name, FilterObjectsModule.additional_input_setting
            ).validate(module)
            FilterObjectsModule.SymbolRequirement(
                self.output_object_name, FilterObjectsModule.additional_output_setting
            ).validate(module)
            return self

    @dataclass(frozen=True, slots=True)
    class MeasurementRule:
        """One measurement limit rule used by FilterObjects."""

        feature_name: str
        use_minimum: bool
        min_value: float | None
        use_maximum: bool
        max_value: float | None

        @classmethod
        def from_block(
            cls, module: "ModuleBlock", block: Sequence["ModuleSetting"]
        ) -> "FilterObjectsModule.MeasurementRule":
            return cls(
                feature_name=block_setting_value(
                    block, FilterObjectsModule.measurement_setting
                ),
                use_minimum=parse_cellprofiler_bool(
                    block_setting_value(
                        block, FilterObjectsModule.use_minimum_setting, default="No"
                    )
                ),
                min_value=FilterObjectsModule.optional_float(
                    block_setting_value(block, FilterObjectsModule.minimum_setting)
                ),
                use_maximum=parse_cellprofiler_bool(
                    block_setting_value(
                        block, FilterObjectsModule.use_maximum_setting, default="No"
                    )
                ),
                max_value=FilterObjectsModule.optional_float(
                    block_setting_value(block, FilterObjectsModule.maximum_setting)
                ),
            ).validated(module)

        def validated(
            self, module: "ModuleBlock"
        ) -> "FilterObjectsModule.MeasurementRule":
            if self.feature_name.strip():
                return self
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an empty FilterObjects measurement rule."
            )

    @dataclass(frozen=True, slots=True)
    class Plan(ObjectPair):
        """Complete typed FilterObjects artifact and runtime plan."""

        additional_rows: tuple["FilterObjectsModule.AdditionalObjectRow", ...]
        enclosing_object_name: str | None
        per_object_assignment: str
        removed_object_name: str | None

        @property
        def input_object_names(self) -> tuple[str, ...]:
            ordered_names = (
                *(pair.input_object_name for pair in self.object_pairs),
                *(
                    ()
                    if self.enclosing_object_name is None
                    else (self.enclosing_object_name,)
                ),
            )
            return tuple(dict.fromkeys(ordered_names))

        @property
        def object_pairs(self) -> tuple["FilterObjectsModule.ObjectPair", ...]:
            return (self, *self.additional_rows)

        def identity_kwargs(self) -> RuntimeCallableKwargs:
            """Return only topology identities that cannot follow main-flow order."""

            kwargs: RuntimeCallableKwargs = {}
            if self.additional_rows:
                kwargs[
                    FilterObjectsModule.additional_input_binding.require_parameter_name()
                ] = tuple(row.input_object_name for row in self.additional_rows)
                kwargs[
                    FilterObjectsModule.additional_output_binding.require_parameter_name()
                ] = tuple(row.output_object_name for row in self.additional_rows)
            if self.removed_object_name is not None:
                kwargs[
                    FilterObjectsModule.removed_output_binding.require_parameter_name()
                ] = self.removed_object_name
            return kwargs

        def output_artifact_specs(
            self,
            module: "ModuleBlock",
            measurement_artifact_name: str,
            artifact_inputs: ArtifactSpecCollection,
            step_context: "ArtifactDeclarationStepContext",
        ) -> tuple[ArtifactSpec, ...]:
            object_sources = tuple(
                (
                    (
                        pair,
                        artifact_inputs.require_by_name_and_artifact_type(
                            pair.input_object_name,
                            ObjectLabelsArtifactType,
                        ),
                    )
                    for pair in self.object_pairs
                )
            )
            object_outputs = tuple(
                (
                    source,
                    ArtifactSpec.output(
                        pair.output_object_name,
                        ObjectLabelsArtifactType,
                        relations=(
                            SourceStackLineageSourceRelation(source=source.ref()),
                        ),
                    ),
                )
                for pair, source in object_sources
            )
            if self.removed_object_name is not None:
                removed_source = object_sources[0][1]
                object_outputs = (
                    *object_outputs,
                    (
                        removed_source,
                        ArtifactSpec.output(
                            self.removed_object_name,
                            ObjectLabelsArtifactType,
                            relations=(
                                FilterObjectsRemovedObjectSourceRelation(
                                    source=removed_source.ref()
                                ),
                            ),
                        ),
                    ),
                )
            relationship_outputs = tuple(
                FilterObjectsModule.parent_child_relationship_output_artifact(
                    module,
                    step_context=step_context,
                    parent=source,
                    child=output,
                    lineage_source=source,
                )
                for source, output in object_outputs
            )
            measurement_dependencies = (
                *(output for _source, output in object_outputs),
                *relationship_outputs,
            )
            return (
                ArtifactSpec.output(
                    measurement_artifact_name,
                    MeasurementsArtifactType,
                    measurement_feature_owner=FilterObjectsModule,
                    relations=(
                        SourceStackLineageSourceRelation(
                            source=object_sources[0][1].ref()
                        ),
                        *(
                            ArtifactSpecRelation(source=output.ref())
                            for output in measurement_dependencies
                        ),
                    ),
                ),
                *(output for _source, output in object_outputs),
                *relationship_outputs,
            )

    @classmethod
    def _filter_kwargs(cls, module: "ModuleBlock") -> "RuntimeCallableKwargs":
        """Return absorbed-function kwargs for a typed FilterObjects plan."""
        plan = cls.plan(module)
        measurement_rules = cls.measurement_rules(module)
        return {
            "mode": cls.mode_value(module),
            "filter_method": optional_setting_value(module, cls.method_setting)
            or "Limits",
            cls.measurement_feature_binding.require_parameter_name(): tuple(
                (rule.feature_name for rule in measurement_rules)
            ),
            cls.measurement_minimum_binding.require_parameter_name(): tuple(
                (rule.min_value for rule in measurement_rules)
            ),
            cls.measurement_maximum_binding.require_parameter_name(): tuple(
                (rule.max_value for rule in measurement_rules)
            ),
            cls.measurement_use_minimum_binding.require_parameter_name(): tuple(
                (rule.use_minimum for rule in measurement_rules)
            ),
            cls.measurement_use_maximum_binding.require_parameter_name(): tuple(
                (rule.use_maximum for rule in measurement_rules)
            ),
            "per_object_assignment": plan.per_object_assignment,
            **plan.identity_kwargs(),
        }

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind behavior and consume the module's repeated topology rows."""

        bound = cls._bind_declared_settings(module, binder=binder)
        bound = bound.with_kwargs(cls._filter_kwargs(module)).with_consumed_settings(
            *(binding.setting_name for binding in cls.measurement_rule_bindings),
            cls.measurement_count_setting,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Reconstruct interleaved measurement-limit rows from public tuples."""

        from openhcs.interop.cellprofiler.cellprofiler_literals import (
            cellprofiler_setting_literal,
        )
        from openhcs.interop.cellprofiler.parser import ModuleSetting

        existing_names = cls._normalized_record_setting_names(existing_records)
        rule_setting_names = {
            normalize_cellprofiler_setting_name(name)
            for binding in cls.measurement_rule_bindings
            for name in setting_names(binding.setting_name)
        }
        own_records: tuple[ModuleSetting, ...] = ()
        if not existing_names.intersection(rule_setting_names):
            columns = tuple(
                tuple(value if isinstance(value, (tuple, list)) else (value,))
                for binding in cls.measurement_rule_bindings
                if (
                    value := invocation.kwargs_dict.get(
                        binding.require_parameter_name(), ()
                    )
                )
            )
            if columns:
                cardinalities = {len(column) for column in columns}
                if (
                    len(columns) != len(cls.measurement_rule_bindings)
                    or len(cardinalities) != 1
                ):
                    raise ValueError(
                        "FilterObjects measurement-rule kwargs must declare equal-length tuples."
                    )
                own_records = tuple(
                    ModuleSetting(
                        setting_names(binding.setting_name)[0],
                        (
                            ""
                            if column[row_index] is None
                            else cellprofiler_setting_literal(column[row_index])
                        ),
                    )
                    for row_index in range(len(columns[0]))
                    for binding, column in zip(
                        cls.measurement_rule_bindings, columns, strict=True
                    )
                )
        return (
            *own_records,
            *super()._derived_identity_setting_records(
                invocation=invocation,
                block_position=block_position,
                existing_records=(*existing_records, *own_records),
                step_context=step_context,
            ),
        )

    @classmethod
    def plan(cls, module: "ModuleBlock") -> "FilterObjectsModule.Plan":
        """Return the typed FilterObjects compile/runtime plan."""
        filter_method = cls.filter_method(module)
        plan = cls.Plan(
            input_object_name=required_setting_value(module, cls.input_setting),
            output_object_name=required_setting_value(module, cls.output_setting),
            additional_rows=cls.additional_rows(module),
            enclosing_object_name=normalized_symbol_name(
                optional_setting_value(module, cls.enclosing_object_setting) or ""
            ),
            per_object_assignment=optional_setting_value(
                module, cls.per_object_assignment_setting
            )
            or "Both parents",
            removed_object_name=cls.removed_object_name(module),
        )
        cls.SymbolRequirement(plan.input_object_name, cls.input_setting).validate(
            module
        )
        cls.SymbolRequirement(plan.output_object_name, cls.output_setting).validate(
            module
        )
        if (
            filter_method.requires_enclosing_object
            and plan.enclosing_object_name is None
        ):
            raise ValueError(
                f"Module {module.name}({module.module_num}) uses "
                f"{filter_method.value!r} filtering without an "
                "enclosing object input."
            )
        return plan

    @classmethod
    def filter_method(cls, module: "ModuleBlock") -> "FilterMethod":
        """Return the nominal filtering method declared by one module."""

        return coerce_cellprofiler_enum(
            FilterMethod,
            optional_setting_value(module, cls.method_setting) or FilterMethod.LIMITS,
        )

    @classmethod
    def additional_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["FilterObjectsModule.AdditionalObjectRow", ...]:
        """Return ordered additional relabel rows from parsed settings."""
        row_count = cls.additional_object_count(module)
        input_names = setting_values(module, cls.additional_input_setting)
        output_names = setting_values(module, cls.additional_output_setting)
        if len(input_names) != row_count or len(output_names) != row_count:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares "
                f"additional_object_count={row_count}, but FilterObjects has "
                f"{len(input_names)} additional inputs and {len(output_names)} "
                "additional outputs."
            )
        return tuple(
            cls.AdditionalObjectRow(input_name, output_name).validated(module)
            for input_name, output_name in zip(
                input_names,
                output_names,
                strict=True,
            )
        )

    @classmethod
    def measurement_rules(
        cls, module: "ModuleBlock"
    ) -> tuple["FilterObjectsModule.MeasurementRule", ...]:
        """Return ordered measurement limit rules from parsed settings."""
        if module.iter_settings():
            blocks = repeating_setting_blocks(
                module.iter_settings(), start_name=cls.measurement_setting
            )
            return tuple(
                (cls.MeasurementRule.from_block(module, block) for block in blocks)
            )
        feature_names = setting_values(module, cls.measurement_setting)
        use_minimum = setting_values(module, cls.use_minimum_setting)
        min_values = setting_values(module, cls.minimum_setting)
        use_maximum = setting_values(module, cls.use_maximum_setting)
        max_values = setting_values(module, cls.maximum_setting)
        return tuple(
            (
                cls.MeasurementRule(
                    feature_name=RepeatedSettingSequence(feature_names).at(index),
                    use_minimum=parse_cellprofiler_bool(
                        RepeatedSettingSequence(use_minimum, default="No").at(index)
                    ),
                    min_value=cls.optional_float(
                        RepeatedSettingSequence(min_values).at(index)
                    ),
                    use_maximum=parse_cellprofiler_bool(
                        RepeatedSettingSequence(use_maximum, default="No").at(index)
                    ),
                    max_value=cls.optional_float(
                        RepeatedSettingSequence(max_values).at(index)
                    ),
                ).validated(module)
                for index in range(len(feature_names))
            )
        )

    @classmethod
    def child_count_object_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        """Return child object names needed by Children_<object>_Count rules."""
        return CellProfilerMeasurementFeature.child_count_object_names(
            tuple((rule.feature_name for rule in cls.measurement_rules(module)))
        )

    @classmethod
    def mode_value(cls, module: "ModuleBlock") -> "FilterMode":
        value = optional_setting_value(module, cls.mode_setting)
        if value is None:
            return FilterMode.MEASUREMENTS
        return FilterMode(value.strip().lower())

    @staticmethod
    def optional_float(raw_value: str | None) -> float | None:
        if raw_value is None:
            return None
        stripped = raw_value.strip()
        if not stripped:
            return None
        return float(stripped)

    @classmethod
    def artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
    ):
        plan = cls.plan(module)
        inputs = [
            *super().artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
            ),
        ]
        available_specs = ArtifactSpecCollection(
            (
                *step_context.main_flow_artifacts.specs,
                *step_context.available_artifacts.specs,
            )
        ).unique(conflict_context="active relationship artifact")

        relationship_declarations = ArtifactSpecCollection(
            available_specs
        ).relation_refs(ObjectRelationshipDeclaration)
        source_ref = (
            ArtifactSpecCollection(inputs)
            .require_by_name_and_artifact_type(
                plan.input_object_name,
                ObjectLabelsArtifactType,
            )
            .ref()
            .for_plan_type(ArtifactInputPlan)
        )
        for target_name in cls.child_count_object_names(module):
            target_ref = ArtifactSpec.input(
                target_name,
                ObjectLabelsArtifactType,
            ).ref()
            matches = tuple(
                spec
                for spec, declaration in relationship_declarations
                if declaration.projects_parent_child_measurements()
                and declaration.source.for_plan_type(ArtifactInputPlan) == source_ref
                and declaration.target.for_plan_type(ArtifactInputPlan) == target_ref
            )
            if len(matches) > 1:
                raise ValueError(
                    "FilterObjects endpoints select multiple active relationship "
                    f"artifacts: {source_ref!r} -> {target_ref!r}: "
                    f"{tuple(spec.name for spec in matches)!r}."
                )
            if matches:
                inputs.append(matches[0].for_plan_type(ArtifactInputPlan))
        return tuple(inputs)

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        plan = cls.plan(module)
        return plan.output_artifact_specs(
            module,
            cls.measurement_artifact_name(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
            ),
            artifact_inputs,
            step_context,
        )


class FilterMethod(Enum):
    """CellProfiler FilterObjects measurement selection modes."""

    MINIMAL = "minimal"
    MAXIMAL = "maximal"
    MINIMAL_PER_OBJECT = "minimal_per_object"
    MAXIMAL_PER_OBJECT = "maximal_per_object"
    LIMITS = "limits"

    @property
    def requires_enclosing_object(self) -> bool:
        """Return whether this method selects one child per enclosing object."""

        return self in {self.MINIMAL_PER_OBJECT, self.MAXIMAL_PER_OBJECT}


class FilterMode(Enum):
    """CellProfiler FilterObjects top-level filter modes."""

    MEASUREMENTS = "measurements"
    BORDER = "border"


class PerObjectAssignment(Enum):
    """How per-object FilterObjects assigns child objects to parents."""

    BOTH_PARENTS = "both_parents"
    PARENT_WITH_MOST_OVERLAP = "parent_with_most_overlap"


FilterObjectsParentChildRelationship = (
    ObjectRelationship | DirectedObjectRelationshipPayload
)
FilterObjectsParentChildRelationships = tuple[FilterObjectsParentChildRelationship, ...]


def filter_objects_relationship_endpoint_ids(values: object) -> tuple[int, ...]:
    """Return flattened integer IDs from one relationship endpoint."""
    return tuple(int(value) for value in np.asarray(values).reshape(-1))


@dataclass(frozen=True, slots=True)
class FilterObjectsMeasurementLimitWindow:
    """Object-id retention policy for FilterObjects measurement bounds."""

    values: ObjectLabelMeasurementValues
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool

    @classmethod
    def from_label_indexed_values(
        cls,
        values: np.ndarray,
        *,
        min_value: float | None,
        max_value: float | None,
        use_minimum: bool,
        use_maximum: bool,
    ) -> "FilterObjectsMeasurementLimitWindow":
        object_ids = tuple(range(1, len(values) + 1))
        return cls(
            ObjectLabelMeasurementValues.from_label_indexed_values(object_ids, values),
            min_value=min_value,
            max_value=max_value,
            use_minimum=use_minimum,
            use_maximum=use_maximum,
        )

    @property
    def retained_ids(self) -> list[int]:
        return list(
            self.values.ids_within_limits(
                min_value=self.min_value,
                max_value=self.max_value,
                use_minimum=self.use_minimum,
                use_maximum=self.use_maximum,
            )
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsStats:
    """FilterObjects object-count output row."""

    slice_index: int
    objects_pre_filter: int
    objects_post_filter: int
    objects_removed: int

    @classmethod
    def from_counts(
        cls, *, objects_pre_filter: int, objects_post_filter: int, slice_index: int = 0
    ) -> "FilterObjectsStats":
        return cls(
            slice_index=slice_index,
            objects_pre_filter=objects_pre_filter,
            objects_post_filter=objects_post_filter,
            objects_removed=objects_pre_filter - objects_post_filter,
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsRelabeledPlane:
    """Relabeled object plane plus the direct input-to-output relationship."""

    labels: np.ndarray
    relationship: DirectedObjectRelationshipPayload

    @classmethod
    def from_ordered_parent_ids(
        cls, labels: np.ndarray, *, parent_ids_by_child: Sequence[int]
    ) -> "FilterObjectsRelabeledPlane":
        """Relabel ``labels`` in child order and retain the exact parent mapping."""
        label_array = np.asarray(labels, dtype=np.int32)
        max_label = int(label_array.max()) if label_array.size else 0
        label_mapping = np.zeros(max_label + 1, dtype=np.int32)
        parent_ids: list[int] = []
        child_ids: list[int] = []
        for child_id, parent_id in enumerate(parent_ids_by_child, start=1):
            if 0 < parent_id <= max_label:
                label_mapping[int(parent_id)] = child_id
                parent_ids.append(int(parent_id))
                child_ids.append(child_id)
        return cls(
            labels=label_mapping[label_array],
            relationship=DirectedObjectRelationshipPayload(
                source_ids=tuple(parent_ids), target_ids=tuple(child_ids)
            ),
        )

    @classmethod
    def from_retained_mask(
        cls, labels: np.ndarray, retained_mask: np.ndarray
    ) -> "FilterObjectsRelabeledPlane":
        """Relabel source IDs overlapping retained primary objects."""
        label_array = np.asarray(labels, dtype=np.int32)
        source_ids = np.unique(label_array[np.asarray(retained_mask, dtype=bool)])
        source_ids = source_ids[source_ids > 0]
        return cls.from_ordered_parent_ids(
            label_array,
            parent_ids_by_child=tuple((int(source_id) for source_id in source_ids)),
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsSelectionRequest:
    """Inputs needed to choose retained primary object labels."""

    labels: np.ndarray
    object_ids: tuple[int, ...]
    filter_method: FilterMethod
    measurement_values: ObjectLabelMeasurementValues | None
    measurement_features: tuple[str, ...]
    measurement_min_values: tuple[float | None, ...]
    measurement_max_values: tuple[float | None, ...]
    measurement_use_minimum: tuple[bool, ...]
    measurement_use_maximum: tuple[bool, ...]
    measurement_tables: tuple[MeasurementTable, ...]
    enclosing_labels: np.ndarray | None
    parent_child_relationships: FilterObjectsParentChildRelationships
    per_object_assignment: PerObjectAssignment
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool

    @property
    def num_objects_pre(self) -> int:
        return len(self.object_ids)

    def measurement_values_for_feature(
        self, feature_name: str
    ) -> ObjectLabelMeasurementValues:
        """Resolve one rule from its exact declared runtime source."""
        if self.measurement_values is not None and self.measurement_features == (
            feature_name,
        ):
            return self.measurement_values

        feature = CellProfilerMeasurementFeature.parse(feature_name)
        if (
            feature is not None
            and feature.kind is CellProfilerMeasurementFeatureKind.CHILD_COUNT
        ):
            return self.child_count_measurement_values(feature)
        if not self.measurement_tables:
            raise ValueError(
                "FilterObjects measurement feature "
                f"{feature_name!r} has no declared measurement-table input."
            )
        values = measurement_values_for_feature(
            self.measurement_tables,
            feature_name,
            object_count=self.num_objects_pre,
            object_ids=self.object_ids,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        return ObjectLabelMeasurementValues(self.object_ids, values)

    def child_count_measurement_values(
        self,
        feature: CellProfilerMeasurementFeature,
    ) -> ObjectLabelMeasurementValues:
        """Resolve a child-count rule from its exact named relationship."""

        child_name = feature.object_name
        if child_name is None:
            raise ValueError(
                f"FilterObjects child-count feature {feature.name!r} has no child identity."
            )
        matches = tuple(
            relationship
            for relationship in self.parent_child_relationships
            if isinstance(relationship, ObjectRelationship)
            and relationship.declaration.target.name == child_name
        )
        if len(matches) != 1:
            raise ValueError(
                f"FilterObjects child-count feature {feature.name!r} requires exactly "
                f"one relationship targeting {child_name!r}, got {len(matches)}."
            )
        counts_by_parent_id: dict[int, float] = {
            object_id: 0.0 for object_id in self.object_ids
        }
        for parent_id in filter_objects_relationship_endpoint_ids(
            matches[0].payload.source_ids
        ):
            if parent_id in counts_by_parent_id:
                counts_by_parent_id[parent_id] += 1.0
        return ObjectLabelMeasurementValues.from_value_mapping(
            self.object_ids,
            counts_by_parent_id,
        )

    def first_measurement_values(self) -> ObjectLabelMeasurementValues:
        if self.measurement_values is not None:
            return self.measurement_values
        if self.measurement_features:
            return self.measurement_values_for_feature(self.measurement_features[0])
        raise ValueError(
            "FilterObjects measurement selection requires a declared measurement "
            "feature or runtime-bound measurement values."
        )

    def matching_measurement_rule_ids(self) -> list[int]:
        self.validate_measurement_rule_lengths()
        retained_ids = set(self.object_ids)
        for index, feature_name in enumerate(self.measurement_features):
            keep_ids = FilterObjectsMeasurementLimitWindow(
                values=self.measurement_values_for_feature(feature_name),
                min_value=self.measurement_min_values[index],
                max_value=self.measurement_max_values[index],
                use_minimum=self.measurement_use_minimum[index],
                use_maximum=self.measurement_use_maximum[index],
            )
            retained_ids.intersection_update(keep_ids.retained_ids)
        return sorted(retained_ids)

    def validate_measurement_rule_lengths(self) -> None:
        expected = len(self.measurement_features)
        lengths = {
            len(self.measurement_min_values),
            len(self.measurement_max_values),
            len(self.measurement_use_minimum),
            len(self.measurement_use_maximum),
        }
        if lengths == {expected}:
            return
        raise ValueError("FilterObjects measurement rule kwargs must align by row.")


@dataclass(frozen=True, slots=True)
class FilterSelectionKey:
    """Nominal retained-object selection identity."""

    mode: FilterMode
    method: FilterMethod | None = None

    @property
    def label(self) -> str:
        if self.method is None:
            return self.mode.value
        return f"{self.mode.value}:{self.method.value}"

    @classmethod
    def from_mode_and_method(
        cls,
        mode: FilterMode,
        method: FilterMethod,
    ) -> "FilterSelectionKey":
        """Return the single selection identity defined by the public mode."""

        if mode is FilterMode.BORDER:
            return cls(mode)
        return cls(mode, method)


class FilterSelectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal retained-object selection for each FilterObjects behavior."""

    __registry_key__ = "selection_label"
    __skip_if_no_key__ = True
    selection_label: ClassVar[str | None] = None
    selection_key: ClassVar[FilterSelectionKey | None] = None

    @classmethod
    def for_mode_and_method(
        cls, mode: FilterMode, method: FilterMethod
    ) -> "FilterSelectionStrategy":
        requested_key = FilterSelectionKey.from_mode_and_method(mode, method)
        strategy_type = cls.__registry__.get(requested_key.label)
        if strategy_type is not None:
            return strategy_type()
        raise ValueError(
            f"Unsupported FilterObjects selection {requested_key.label!r}."
        )

    @abstractmethod
    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        """Return one-indexed primary object labels to retain."""


class BorderFilterSelectionStrategy(FilterSelectionStrategy):
    """Remove primary objects touching the image border."""

    selection_key = FilterSelectionKey(FilterMode.BORDER)
    selection_label = selection_key.label

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        return self.discard_border_objects(request.labels)

    @staticmethod
    def discard_border_objects(labels: np.ndarray) -> list[int]:
        from scipy import ndimage as ndi

        interior_pixels = ndi.binary_erosion(np.ones_like(labels, dtype=bool))
        border_labels = set(labels[~interior_pixels])
        keep_labels = list(set(labels.ravel()).difference(border_labels))
        if 0 in keep_labels:
            keep_labels.remove(0)
        keep_labels.sort()
        return keep_labels


class LimitsFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep objects whose measurement falls within configured limits."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.LIMITS)
    selection_label = selection_key.label

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        if request.measurement_features:
            return request.matching_measurement_rule_ids()
        values = request.measurement_values
        if values is None:
            values = request.first_measurement_values()
        return FilterObjectsMeasurementLimitWindow(
            values=values,
            min_value=request.min_value,
            max_value=request.max_value,
            use_minimum=request.use_minimum,
            use_maximum=request.use_maximum,
        ).retained_ids


class ExtremumFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep one object selected by a measurement extremum."""

    keep_max: ClassVar[bool | None] = None

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        keep_max = type(self).keep_max
        if keep_max is None:
            raise TypeError("ExtremumFilterSelectionStrategy must define keep_max.")
        values = request.measurement_values
        if values is None:
            values = request.first_measurement_values()
        return keep_one_object(values, keep_max=keep_max)


class MinimalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the minimum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MINIMAL)
    selection_label = selection_key.label
    keep_max = False


class MaximalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the maximum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MAXIMAL)
    selection_label = selection_key.label
    keep_max = True


class PerObjectFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep one child object per enclosing parent object."""

    selection_key: ClassVar[FilterSelectionKey | None] = None

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        selection_key = type(self).selection_key
        if selection_key is None or selection_key.method is None:
            raise TypeError("PerObjectFilterSelectionStrategy must define method.")
        values = request.first_measurement_values().dense_label_indexed(
            max_label=int(request.labels.max()) if request.labels.size else 0
        )
        return PerObjectAssignmentStrategy.for_assignment(
            request.per_object_assignment
        ).indexes_to_keep(
            PerObjectAssignmentRequest(
                child_labels=request.labels,
                enclosing_labels=require_enclosing_labels(request),
                measurement_values=values,
                child_count=request.num_objects_pre,
                keep_max=selection_key.method is FilterMethod.MAXIMAL_PER_OBJECT,
            )
        )


class MinimalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for minimal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS, FilterMethod.MINIMAL_PER_OBJECT
    )
    selection_label = selection_key.label


class MaximalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for maximal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS, FilterMethod.MAXIMAL_PER_OBJECT
    )
    selection_label = selection_key.label


@dataclass(frozen=True, slots=True)
class PerObjectAssignmentRequest:
    """Inputs for assigning candidate child objects to enclosing parents."""

    child_labels: np.ndarray
    enclosing_labels: np.ndarray
    measurement_values: np.ndarray
    child_count: int
    keep_max: bool

    def __post_init__(self) -> None:
        if self.child_labels.shape != self.enclosing_labels.shape:
            raise ValueError(
                f"FilterObjects per-object child and enclosing labels must have matching shape, got {self.child_labels.shape} and {self.enclosing_labels.shape}."
            )


class PerObjectAssignmentStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal parent-assignment strategy for per-object filtering."""

    __registry_key__ = "assignment_label"
    __skip_if_no_key__ = True
    assignment_label: ClassVar[str | None] = None
    assignment: ClassVar[PerObjectAssignment | None] = None

    @classmethod
    def for_assignment(
        cls, assignment: PerObjectAssignment
    ) -> "PerObjectAssignmentStrategy":
        strategy_type = cls.__registry__.get(assignment.value)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported FilterObjects per-object assignment {assignment.value!r}."
            )
        return strategy_type()

    @abstractmethod
    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        """Return child IDs selected by this exact assignment policy."""

    def best_child_indexes_by_parent(
        self, parent_children: dict[int, set[int]], request: PerObjectAssignmentRequest
    ) -> list[int]:
        selected: set[int] = set()
        for child_ids in parent_children.values():
            child_values = tuple(
                (
                    (
                        child_id,
                        self.measurement_value_for_child(
                            request.measurement_values, child_id
                        ),
                    )
                    for child_id in child_ids
                )
            )
            finite_child_values = tuple(
                (
                    (child_id, value)
                    for child_id, value in child_values
                    if np.isfinite(value)
                )
            )
            if not finite_child_values:
                continue
            selected.add(
                min(
                    finite_child_values,
                    key=(
                        (lambda item: (-item[1], item[0]))
                        if request.keep_max
                        else lambda item: (item[1], item[0])
                    ),
                )[0]
            )
        return sorted(selected)

    @staticmethod
    def measurement_value_for_child(
        measurement_values: np.ndarray, child_id: int
    ) -> float:
        value_index = child_id - 1
        if value_index < 0 or value_index >= len(measurement_values):
            return float("nan")
        return float(measurement_values[value_index])

    @staticmethod
    def overlap_label_pairs(
        request: PerObjectAssignmentRequest,
    ) -> tuple[tuple[int, int], ...]:
        overlap_mask = (request.child_labels > 0) & (request.enclosing_labels > 0)
        child_ids = request.child_labels[overlap_mask].astype(np.int64, copy=False)
        parent_ids = request.enclosing_labels[overlap_mask].astype(np.int64, copy=False)
        return tuple(
            (
                (int(child_id), int(parent_id))
                for child_id, parent_id in zip(child_ids, parent_ids, strict=True)
            )
        )

    @abstractmethod
    def parent_children(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        """Return child labels eligible for each enclosing parent label."""


class BothParentsAssignmentStrategy(PerObjectAssignmentStrategy):
    """Assign an overlapping child as a candidate for every touched parent."""

    assignment = PerObjectAssignment.BOTH_PARENTS
    assignment_label = assignment.value

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        return best_child_indexes_both_parents(
            request.child_labels,
            request.enclosing_labels,
            request.measurement_values,
            request.keep_max,
        )

    def parent_children(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        parent_children: dict[int, set[int]] = {}
        for child_id, parent_id in self.overlap_label_pairs(request):
            parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children


class ParentWithMostOverlapAssignmentStrategy(PerObjectAssignmentStrategy):
    """Assign each child only to its most-overlapped enclosing parent."""

    assignment = PerObjectAssignment.PARENT_WITH_MOST_OVERLAP
    assignment_label = assignment.value

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        return best_child_indexes_parent_with_most_overlap(
            request.child_labels,
            request.enclosing_labels,
            request.measurement_values,
            request.keep_max,
        )

    def parent_children(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        counts_by_child: dict[int, dict[int, int]] = {}
        for child_id, parent_id in self.overlap_label_pairs(request):
            parent_counts = counts_by_child.setdefault(child_id, {})
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        parent_children: dict[int, set[int]] = {}
        for child_id, parent_counts in counts_by_child.items():
            parent_id = min(
                parent_counts,
                key=lambda candidate: (-parent_counts[candidate], candidate),
            )
            parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children


def keep_one_object(
    values: ObjectLabelMeasurementValues, keep_max: bool = True
) -> list[int]:
    """Keep only the object with the maximum or minimum finite measurement."""
    selected_id = values.extremum_id(keep_max=keep_max)
    return [] if selected_id is None else [selected_id]


def require_enclosing_labels(request: FilterObjectsSelectionRequest) -> np.ndarray:
    if request.enclosing_labels is not None:
        return request.enclosing_labels
    raise ValueError(
        "FilterObjects per-object filtering requires enclosing object labels."
    )


def best_child_indexes_both_parents(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    import scipy.ndimage

    values = np.asarray(measurement_values, dtype=np.float64)
    if values.size == 0:
        return []
    child_array = np.asarray(child_labels, dtype=np.int32)
    parent_array = np.asarray(enclosing_labels, dtype=np.int32)
    max_parent = int(parent_array.max()) if parent_array.size else 0
    if max_parent <= 0:
        return []
    pixel_values = np.empty(values.size + 1, dtype=np.float64)
    pixel_values[1:] = values
    pixel_values[0] = -np.inf if keep_max else np.inf
    source_values = pixel_values[child_array]
    parent_range = np.arange(1, max_parent + 1)
    position_fn = (
        scipy.ndimage.maximum_position if keep_max else scipy.ndimage.minimum_position
    )
    positions = position_fn(source_values, parent_array, parent_range)
    positions = np.asarray(
        (positions,) if isinstance(positions, tuple) else positions, dtype=np.uint32
    )
    if positions.size == 0:
        return []
    indexes = tuple(map(tuple, positions.transpose()))
    selected = sorted(set((int(label) for label in child_array[indexes])))
    if selected and selected[0] == 0:
        selected = selected[1:]
    return selected


def best_child_indexes_parent_with_most_overlap(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    max_child = int(np.max(child_labels))
    if max_child <= 0:
        return []
    child_ids, parent_ids, overlap_counts = unique_overlap_counts(
        child_labels, enclosing_labels
    )
    if child_ids.size == 0:
        return []
    return selected_labels_to_list(
        _best_child_selected_mask_parent_with_most_overlap_numba(
            np.ascontiguousarray(child_ids, dtype=np.int32),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(overlap_counts, dtype=np.int64),
            np.ascontiguousarray(measurement_values, dtype=np.float64),
            max_child,
            bool(keep_max),
        )
    )


def unique_overlap_counts(
    child_labels: np.ndarray, enclosing_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    child_array = np.asarray(child_labels, dtype=np.int64)
    parent_array = np.asarray(enclosing_labels, dtype=np.int64)
    max_parent = int(parent_array.max())
    overlap_mask = (child_array > 0) & (parent_array > 0)
    if not np.any(overlap_mask):
        empty = np.array([], dtype=np.int64)
        return (empty, empty, empty)
    encoded_pairs = (
        child_array[overlap_mask] * (max_parent + 1) + parent_array[overlap_mask]
    )
    unique_pairs, overlap_counts = np.unique(encoded_pairs, return_counts=True)
    return (
        unique_pairs // (max_parent + 1),
        unique_pairs % (max_parent + 1),
        overlap_counts,
    )


@njit(cache=True)
def _best_child_selected_mask_parent_with_most_overlap_numba(
    child_ids: np.ndarray,
    parent_ids: np.ndarray,
    overlap_counts: np.ndarray,
    measurement_values: np.ndarray,
    max_child: int,
    keep_max: bool,
) -> np.ndarray:
    selected = np.zeros(max_child + 1, dtype=np.bool_)
    if child_ids.size == 0:
        return selected
    max_parent = int(np.max(parent_ids))
    best_parent_by_child = np.zeros(max_child + 1, dtype=np.int32)
    best_overlap_by_child = np.zeros(max_child + 1, dtype=np.int64)
    for index in range(child_ids.size):
        child_id = int(child_ids[index])
        parent_id = int(parent_ids[index])
        overlap_count = int(overlap_counts[index])
        best_parent = int(best_parent_by_child[child_id])
        best_overlap = int(best_overlap_by_child[child_id])
        if (
            best_parent == 0
            or overlap_count > best_overlap
            or (overlap_count == best_overlap and parent_id < best_parent)
        ):
            best_parent_by_child[child_id] = parent_id
            best_overlap_by_child[child_id] = overlap_count
    best_child_by_parent = np.zeros(max_parent + 1, dtype=np.int32)
    best_value_by_parent = np.empty(max_parent + 1, dtype=np.float64)
    for child_id in range(1, max_child + 1):
        parent_id = int(best_parent_by_child[child_id])
        if parent_id <= 0:
            continue
        value_index = child_id - 1
        if value_index < 0 or value_index >= measurement_values.size:
            continue
        value = float(measurement_values[value_index])
        if not np.isfinite(value):
            continue
        best_child = int(best_child_by_parent[parent_id])
        if best_child == 0:
            best_child_by_parent[parent_id] = child_id
            best_value_by_parent[parent_id] = value
        else:
            best_value = float(best_value_by_parent[parent_id])
            if keep_max:
                if value > best_value or (
                    value == best_value and child_id < best_child
                ):
                    best_child_by_parent[parent_id] = child_id
                    best_value_by_parent[parent_id] = value
            elif value < best_value or (value == best_value and child_id < best_child):
                best_child_by_parent[parent_id] = child_id
                best_value_by_parent[parent_id] = value
    for parent_id in range(1, max_parent + 1):
        child_id = int(best_child_by_parent[parent_id])
        if child_id > 0:
            selected[child_id] = True
    return selected


def selected_labels_to_list(selected: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(selected, dtype=bool)).astype(int).tolist()


def filtered_object_payloads(
    inputs: Sequence[ObjectLabelValue], outputs: Sequence[np.ndarray]
) -> tuple[ObjectLabelValue, ...]:
    """Return dense object payloads preserving the original object domain."""
    return tuple(
        (
            filtered_object_payload(input_value, output_labels)
            for input_value, output_labels in zip(inputs, outputs, strict=True)
        )
    )


def filtered_object_payload(
    input_value: ObjectLabelValue, output_labels: np.ndarray
) -> ObjectLabelValue:
    """Wrap filtered labels with the input object's dense extent domain."""
    return object_label_value_with_dense_labels(
        input_value,
        output_labels,
        domain_declaration=DenseObjectLabelExtentDomainDeclaration(),
    )


def relabel_overlapping_objects(
    labels: np.ndarray, filtered_primary_labels: np.ndarray
) -> np.ndarray:
    """Relabel additional objects by overlap with retained primary objects."""
    retained_mask = filtered_primary_labels > 0
    labels = labels.astype(np.int32)
    if labels.shape != retained_mask.shape:
        raise ValueError(
            "FilterObjects additional object labels must match primary labels."
        )
    return FilterObjectsRelabeledPlane.from_retained_mask(labels, retained_mask).labels


def object_transform_relationships(
    input_label_planes: tuple[np.ndarray, ...],
    relabeled_objects: tuple[np.ndarray, ...],
) -> tuple[DirectedObjectRelationshipPayload, ...]:
    """Derive parent-child payloads between input and filtered object labels."""
    if len(input_label_planes) != len(relabeled_objects):
        raise ValueError(
            "Object transform relationship derivation requires aligned input and output label planes."
        )
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type()
    return tuple(
        (
            relationship_backend.parent_child_payload_from_labels(
                np.asarray(input_labels), np.asarray(output_labels)
            )
            for input_labels, output_labels in zip(
                input_label_planes, relabeled_objects, strict=True
            )
        )
    )


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_inputs(
    ObjectLabelsRuntimeParameter.require_parameter_name(),
    _FilterEnclosingLabelsRuntimeParameter.require_parameter_name(),
)
@runtime_bound_parameters(
    _FilterMeasurementValuesRuntimeParameter,
    _FilterMeasurementTablesRuntimeParameter,
    _FilterParentChildRelationshipsRuntimeParameter,
)
def filter_objects(
    image: np.ndarray,
    mode: FilterMode = FilterMode.MEASUREMENTS,
    filter_method: FilterMethod = FilterMethod.LIMITS,
    object_labels: tuple[ObjectLabelValue, ...] = (),
    measurement_values: np.ndarray | None = None,
    measurement_features: tuple[str, ...] = (),
    measurement_min_values: tuple[float | None, ...] = (),
    measurement_max_values: tuple[float | None, ...] = (),
    measurement_use_minimum: tuple[bool, ...] = (),
    measurement_use_maximum: tuple[bool, ...] = (),
    measurement_tables: tuple[MeasurementTable, ...] = (),
    enclosing_object_labels: ObjectLabelValue | None = None,
    parent_child_relationships: FilterObjectsParentChildRelationships = (),
    per_object_assignment: PerObjectAssignment = PerObjectAssignment.BOTH_PARENTS,
    min_value: float | None = None,
    max_value: float | None = None,
    use_minimum: bool = True,
    use_maximum: bool = True,
    additional_object_count: int = 0,
    emit_removed_objects: bool = False,
    slice_by_slice: bool = True,
) -> (
    tuple[
        np.ndarray,
        DataclassMeasurementColumnarRows,
        ObjectLabelValue,
        DirectedObjectRelationshipPayload,
    ]
    | tuple[
        np.ndarray,
        DataclassMeasurementColumnarRows,
        np.ndarray | DirectedObjectRelationshipPayload,
        ...,
    ]
):
    """Filter dense object labels using CellProfiler-compatible selection policy.

    Args:
        object_labels: Ordered object-label values: the first is filtered and any
            additional sets are relabeled from regions overlapping retained
            primary objects.
        measurement_features: Ordered CellProfiler measurement feature names,
            with one limit rule per entry.
        measurement_min_values: Lower bounds aligned by index with
            ``measurement_features``; ``None`` leaves a rule unbounded below.
        measurement_max_values: Upper bounds aligned by index with
            ``measurement_features``; ``None`` leaves a rule unbounded above.
        measurement_use_minimum: Flags aligned with ``measurement_features`` that
            enable each rule's lower bound.
        measurement_use_maximum: Flags aligned with ``measurement_features`` that
            enable each rule's upper bound.
        enclosing_object_labels: Parent-object labels used by per-parent minimum
            or maximum selection modes.
        min_value: Lower limit for the single runtime measurement vector when no
            repeated ``measurement_features`` rules are declared.
        max_value: Upper limit for the single runtime measurement vector when no
            repeated ``measurement_features`` rules are declared.
        use_minimum: Apply ``min_value`` to the single-vector limit rule.
        use_maximum: Apply ``max_value`` to the single-vector limit rule.
    """
    if len(object_labels) == 0:
        raise ValueError("FilterObjects requires at least one object label input.")
    if additional_object_count != len(object_labels) - 1:
        raise ValueError(
            "FilterObjects additional_object_count must match additional object label inputs."
        )
    primary_labels = object_labels[0]
    paired_values = (
        *object_labels[1:],
        *(() if enclosing_object_labels is None else (enclosing_object_labels,)),
    )
    aligned_values, aligned_adapters = SourceSpatialDomainAdapter.aligned_values(
        (primary_labels, *paired_values)
    )
    labels = np.asarray(aligned_values[0], dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError(
            "FilterObjects requires one explicitly projected 2-D object-label plane."
        )
    additional_label_planes = tuple(
        np.asarray(value, dtype=np.int32)
        for value in aligned_values[1 : len(object_labels)]
    )
    output_adapters = aligned_adapters[: len(object_labels)]
    max_label = labels.max()
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0, objects_post_filter=0
        )
        relabeled_planes = tuple(
            (
                FilterObjectsRelabeledPlane.from_ordered_parent_ids(
                    label_plane, parent_ids_by_child=()
                )
                for label_plane in (labels, *additional_label_planes)
            )
        )
        if emit_removed_objects:
            relabeled_planes = (
                *relabeled_planes,
                FilterObjectsRelabeledPlane.from_ordered_parent_ids(
                    labels, parent_ids_by_child=()
                ),
            )
        output_sources = (
            *object_labels,
            *((object_labels[0],) if emit_removed_objects else ()),
        )
        output_adapters = (
            *output_adapters,
            *((output_adapters[0],) if emit_removed_objects else ()),
        )
        relabeled_objects = filtered_object_payloads(
            output_sources,
            tuple(
                adapter.extract_source_array(
                    plane.labels,
                    spatial_axes_yx=adapter.spatial_axes_yx,
                )
                for plane, adapter in zip(
                    relabeled_planes, output_adapters, strict=True
                )
            ),
        )
        return (
            image,
            DataclassMeasurementColumnarRows((stats,), row_type=FilterObjectsStats),
            *relabeled_objects,
            *(plane.relationship for plane in relabeled_planes),
        )
    object_ids = ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels)
    selection_measurement_values = (
        None
        if measurement_values is None
        else ObjectLabelMeasurementValues.from_positional_values(
            object_ids, measurement_values
        )
    )
    indexes_to_keep = FilterSelectionStrategy.for_mode_and_method(
        mode, filter_method
    ).indexes_to_keep(
        FilterObjectsSelectionRequest(
            labels=labels,
            object_ids=object_ids,
            filter_method=filter_method,
            measurement_values=selection_measurement_values,
            measurement_features=measurement_features,
            measurement_min_values=measurement_min_values,
            measurement_max_values=measurement_max_values,
            measurement_use_minimum=measurement_use_minimum,
            measurement_use_maximum=measurement_use_maximum,
            measurement_tables=measurement_tables,
            enclosing_labels=(
                None
                if enclosing_object_labels is None
                else np.asarray(aligned_values[-1], dtype=np.int32)
            ),
            parent_child_relationships=parent_child_relationships,
            per_object_assignment=per_object_assignment,
            min_value=min_value,
            max_value=max_value,
            use_minimum=use_minimum,
            use_maximum=use_maximum,
        )
    )
    primary_plane = FilterObjectsRelabeledPlane.from_ordered_parent_ids(
        labels, parent_ids_by_child=tuple(indexes_to_keep)
    )
    filtered_labels = primary_plane.labels
    retained_mask = filtered_labels > 0
    additional_relabeled_planes = tuple(
        (
            FilterObjectsRelabeledPlane.from_retained_mask(additional, retained_mask)
            for additional in additional_label_planes
        )
    )
    relabeled_planes = (primary_plane, *additional_relabeled_planes)
    if emit_removed_objects:
        retained_ids = frozenset(int(value) for value in indexes_to_keep)
        relabeled_planes = (
            *relabeled_planes,
            FilterObjectsRelabeledPlane.from_ordered_parent_ids(
                labels,
                parent_ids_by_child=tuple(
                    object_id
                    for object_id in object_ids
                    if object_id not in retained_ids
                ),
            ),
        )
    output_sources = (
        *object_labels,
        *((object_labels[0],) if emit_removed_objects else ()),
    )
    output_adapters = (
        *output_adapters,
        *((output_adapters[0],) if emit_removed_objects else ()),
    )
    relabeled_objects = filtered_object_payloads(
        output_sources,
        tuple(
            adapter.extract_source_array(
                plane.labels,
                spatial_axes_yx=adapter.spatial_axes_yx,
            )
            for plane, adapter in zip(relabeled_planes, output_adapters, strict=True)
        ),
    )
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=len(object_ids), objects_post_filter=len(indexes_to_keep)
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=FilterObjectsStats),
        *relabeled_objects,
        *(plane.relationship for plane in relabeled_planes),
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def filter_objects_by_size(
    image: np.ndarray,
    labels: ObjectLabelValue,
    min_area: float = 0.0,
    max_area: float = float("inf"),
    use_minimum: bool = True,
    use_maximum: bool = True,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Filter objects based on area measurements."""
    source_labels = labels
    label_array = object_label_dense_array(source_labels, dtype=np.int32)
    max_label = label_array.max()
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0, objects_post_filter=0
        )
        return (
            image,
            DataclassMeasurementColumnarRows((stats,), row_type=FilterObjectsStats),
            filtered_object_payload(source_labels, label_array),
        )
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    indexes_to_keep = FilterObjectsMeasurementLimitWindow.from_label_indexed_values(
        region_props.area,
        min_value=min_area,
        max_value=max_area,
        use_minimum=use_minimum,
        use_maximum=use_maximum,
    ).retained_ids
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=len(region_props.label),
        objects_post_filter=len(indexes_to_keep),
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=FilterObjectsStats),
        filtered_object_payload(source_labels, label_mapping[label_array]),
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def filter_border_objects(
    image: np.ndarray, labels: ObjectLabelValue
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Remove objects touching the image border."""
    source_labels = labels
    label_array = object_label_dense_array(source_labels, dtype=np.int32)
    max_label = label_array.max()
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0, objects_post_filter=0
        )
        return (
            image,
            DataclassMeasurementColumnarRows((stats,), row_type=FilterObjectsStats),
            filtered_object_payload(source_labels, label_array),
        )
    object_ids = ObjectLabelIdDomainStrategy.for_value(label_array).present_ids(
        label_array
    )
    indexes_to_keep = BorderFilterSelectionStrategy.discard_border_objects(label_array)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=len(object_ids), objects_post_filter=len(indexes_to_keep)
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=FilterObjectsStats),
        filtered_object_payload(source_labels, label_mapping[label_array]),
    )


__all__ = public_names_from_objects(
    FilterMethod,
    FilterMode,
    FilterObjectsMeasurementLimitWindow,
    "FilterObjectsParentChildRelationship",
    "FilterObjectsParentChildRelationships",
    FilterObjectsSelectionRequest,
    FilterObjectsStats,
    FilterSelectionStrategy,
    "FilterSelectionKey",
    PerObjectAssignment,
    PerObjectAssignmentRequest,
    PerObjectAssignmentStrategy,
    best_child_indexes_both_parents,
    best_child_indexes_parent_with_most_overlap,
    filter_objects_relationship_endpoint_ids,
    filter_border_objects,
    filter_objects,
    filter_objects_by_size,
    filtered_object_payload,
    filtered_object_payloads,
    keep_one_object,
    object_transform_relationships,
    relabel_overlapping_objects,
    require_enclosing_labels,
)
