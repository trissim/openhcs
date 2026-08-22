"""Relationship backends for CellProfiler-compatible processing."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Annotated
from openhcs.core.alias_property import AliasProperty
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    InputGroupLineageSourceRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.equivalence.relationships import (
    GenericRelationshipAggregateFeatureSemantics,
    RelationshipAggregateFeatureContext,
)
from openhcs.core.pipeline.function_contracts import runtime_bound_parameters
from openhcs.core.runtime_batch_contracts import (
    SliceIndexRuntimeParameter,
    runtime_callable_defaults,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_measurements import (
    MeasurementScalarLiteral,
    MeasurementStatistic,
    RuntimeMeasurementFeatureDeclaration,
    RuntimeMeasurementFeatureSemanticMarker,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    PlaneRuntimeArtifactModule,
    ParentChildLineageArtifactOutputModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    ObjectLabelDrivenPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerObjectCoreMeasurementFeature,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    FormattingMeasurementFeatureTemplate,
    ObjectLocationMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    DirectParentReferenceFeatureMarker,
    RelationshipMeasurementRows,
)

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.core.source_bindings import StepSourceBindingsConfig
    from openhcs.interop.cellprofiler.parser import ModuleBlock


@dataclass(frozen=True)
class RelateObjectsDistanceParentInputRelation(ArtifactSpecRelation):
    """Mark an object input as a repeated distance-parent role."""

    relation_key = "relate_objects_distance_parent_input"
    target_plan_type = ArtifactInputPlan
    target_artifact_type = ObjectLabelsArtifactType


class RelateObjectsDistanceMethod(Enum):
    """CellProfiler RelateObjects child-parent distance calculation mode."""

    NONE = ("none", False, False)
    CENTROID = ("centroid", True, False)
    MINIMUM = ("minimum", False, True)
    BOTH = ("both", True, True)

    def __new__(
        cls,
        label: str,
        calculates_centroid_distance: bool,
        calculates_minimum_distance: bool,
    ) -> "RelateObjectsDistanceMethod":
        obj = object.__new__(cls)
        obj._value_ = label
        return obj

    def __init__(
        self,
        label: str,
        calculates_centroid_distance: bool,
        calculates_minimum_distance: bool,
    ) -> None:
        self._calculates_centroid_distance = calculates_centroid_distance
        self._calculates_minimum_distance = calculates_minimum_distance

    calculates_centroid_distance = AliasProperty[bool]("_calculates_centroid_distance")
    calculates_minimum_distance = AliasProperty[bool]("_calculates_minimum_distance")


DistanceMethod = RelateObjectsDistanceMethod


class RelateObjectsChildMeanFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Semantic marker for a RelateObjects aggregate over child measurements."""


@dataclass(frozen=True, slots=True)
class RelateObjectsChildMeanMeasurementFeature:
    """Nominal identity encoded by a ``Mean_<child>_<feature>`` name."""

    qualified_child_feature_parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.qualified_child_feature_parts) < 2 or any(
            not isinstance(part, str) or not part
            for part in self.qualified_child_feature_parts
        ):
            raise ValueError(
                "RelateObjects child-mean identity requires child and feature parts."
            )


class RelateObjectsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    PlaneRuntimeArtifactModule,
    NoObjectNameMeasurementRecordMixin,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    ParentChildLineageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "RelateObjects"
    function_name = "relate_objects"
    function_variants = ("relate_objects_with_saved_children",)
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("children",), ("parent",))
    parent_relationship_type = "Parent"
    child_relationship_type = "Child"

    @classmethod
    def main_flow_output_specs(
        cls,
        main_flow_candidates: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Publish the saved-child object set returned in the canonical slot."""

        del cls
        object_outputs = tuple(
            spec
            for spec in main_flow_candidates
            if spec.artifact_type is ObjectLabelsArtifactType
        )
        if len(object_outputs) > 1:
            raise ValueError(
                "RelateObjects declares more than one canonical object output: "
                f"{tuple(spec.ref() for spec in object_outputs)!r}."
            )
        return object_outputs

    @classmethod
    def aggregates_child_measurement_feature(cls, feature_name: str) -> bool:
        """Return whether CellProfiler derives a per-parent mean for this feature."""

        if RuntimeMeasurementFeatureDeclaration.feature_has_semantic_marker(
            feature_name,
            DirectParentReferenceFeatureMarker,
        ):
            return False
        return not RuntimeMeasurementFeatureDeclaration.feature_has_semantic_marker(
            feature_name,
            RelateObjectsChildMeanFeatureMarker,
        )

    @staticmethod
    def aggregate_child_measurement_value_is_qualified(value: object) -> bool:
        """Retain explicit missing values so child means propagate them."""

        return not MeasurementScalarLiteral(value).is_absent

    class DistanceMeasurementFeature(FormattingMeasurementFeatureTemplate):
        """Parent-qualified distance features emitted by RelateObjects."""

        DISTANCE_CENTROID = ("Distance_Centroid_{parent_object_name}", float)
        DISTANCE_MINIMUM = ("Distance_Minimum_{parent_object_name}", float)

        @classmethod
        def database_measurement_dtype(cls) -> type[object]:
            """Match CellProfiler's integer SQLite declaration for distances."""

            return int

        @property
        def unqualified_feature_name(self) -> str:
            """Return this declaration's parent-neutral feature identity."""

            suffix = "_{parent_object_name}"
            if not self.value.endswith(suffix):
                raise ValueError(
                    f"{type(self).__name__}.{self.name} must end with {suffix!r}."
                )
            return normalize_runtime_identifier(self.value.removesuffix(suffix))

        @classmethod
        def matching_feature(
            cls,
            feature_name: str,
            *,
            parent_object_name: str,
        ) -> "RelateObjectsModule.DistanceMeasurementFeature | None":
            """Resolve a raw or parent-qualified name to its declaration."""

            normalized_feature_name = normalize_runtime_identifier(feature_name)
            normalized_parent_name = normalize_runtime_identifier(parent_object_name)
            matching = tuple(
                feature
                for feature in cls
                if normalized_feature_name
                in (
                    feature.unqualified_feature_name,
                    normalize_runtime_identifier(
                        feature.feature_name(parent_object_name=normalized_parent_name)
                    ),
                )
            )
            if len(matching) > 1:
                raise ValueError(
                    "RelateObjects distance declarations overlap for feature "
                    f"{feature_name!r}."
                )
            return matching[0] if matching else None

    class AggregateMeasurementFeature(FormattingMeasurementFeatureTemplate):
        """Per-parent aggregate features emitted by RelateObjects."""

        MEAN_CHILD = ("Mean_{child_object_name}_{child_feature_name}", float)

    distance_setting = SettingNameFamily("Calculate child-parent distances?")
    parent_objects_setting = SettingNameFamily(
        "Select the parent objects", aliases=("Parent objects",)
    )
    child_objects_setting = SettingNameFamily(
        "Select the child objects", aliases=("Child objects",)
    )
    per_parent_means_setting = SettingNameFamily(
        "Calculate per-parent means for all child measurements?"
    )
    other_parent_distances_setting = SettingNameFamily(
        "Calculate distances to other parents?"
    )
    other_parent_objects_setting = SettingNameFamily("Parent name")
    save_children_setting = SettingNameFamily(
        "Do you want to save the children with parents as a new object set?"
    )
    output_object_setting = SettingNameFamily("Name the output object")
    parent_objects_binding = SettingToKeywordBinding.input(
        parent_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="parent_labels",
    )
    child_objects_binding = SettingToKeywordBinding.input(
        child_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="child_labels",
    )
    other_parent_objects_binding = SettingToKeywordBinding.input(
        other_parent_objects_setting,
        ObjectLabelsArtifactType,
        repeated=True,
    )
    setting_bindings = (
        parent_objects_binding,
        child_objects_binding,
        other_parent_objects_binding,
        SettingToKeywordBinding.output(output_object_setting, ObjectLabelsArtifactType),
        SettingToKeywordBinding(
            distance_setting,
            "calculate_distances",
            cellprofiler_enum_setting_parser(RelateObjectsDistanceMethod),
        ),
        SettingToKeywordBinding(
            per_parent_means_setting,
            "calculate_per_parent_means",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            other_parent_distances_setting,
            "calculate_distances_to_other_parents",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            save_children_setting, "save_children_with_parents", parse_cellprofiler_bool
        ),
    )

    @classmethod
    def primary_image_domain_input_binding(cls) -> SettingToKeywordBinding:
        """Use the child objects as the relationship invocation domain."""

        return cls.child_objects_binding

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Expose optional distance parents and saved-child output exactly."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        other_parents = cls.other_parent_distances_enabled(module)
        save_children = optional_setting_value(module, cls.save_children_setting)
        save_children = save_children is not None and parse_cellprofiler_bool(
            save_children
        )
        return tuple(
            binding
            for binding in bindings
            if other_parents or binding is not cls.other_parent_objects_binding
            if save_children
            or not (
                binding.require_artifact_plan_type() is ArtifactOutputPlan
                and binding.require_artifact_type() is ObjectLabelsArtifactType
            )
        )

    @classmethod
    def distance_method(
        cls,
        module: "ModuleBlock",
    ) -> RelateObjectsDistanceMethod:
        """Return the nominal distance method declared by one module."""

        return coerce_cellprofiler_enum(
            RelateObjectsDistanceMethod,
            required_setting_value(module, cls.distance_setting),
        )

    @classmethod
    def other_parent_distances_enabled(cls, module: "ModuleBlock") -> bool:
        """Return whether repeated step-parent distance inputs are active."""

        requested = parse_cellprofiler_bool(
            required_setting_value(
                module,
                cls.other_parent_distances_setting,
            )
        )
        method = cls.distance_method(module)
        return requested and (
            method.calculates_centroid_distance or method.calculates_minimum_distance
        )

    @classmethod
    def relationship_measurement_rows(cls, request):
        """Return RelateObjects relationship rows including distance features."""
        return RelateObjectsRelationshipMeasurementRows(request)

    @classmethod
    def relationship_distance_measurements_apply(
        cls,
        callable_contract: "CallableContract",
        relationship_spec: ArtifactSpec,
    ) -> bool:
        """Return whether this is the explicit parent-to-child result."""

        declared_spec = callable_contract.artifact_outputs.by_ref(
            relationship_spec.ref()
        )
        if declared_spec != relationship_spec:
            raise ValueError(
                f"Callable {callable_contract.function_name!r} does not declare "
                f"exact relationship output {relationship_spec.ref()!r}."
            )
        declarations = tuple(
            relation
            for relation in relationship_spec.relations
            if isinstance(relation, ObjectRelationshipDeclaration)
        )
        if len(declarations) != 1:
            raise ValueError(
                f"Callable {callable_contract.function_name!r} relationship output "
                f"{relationship_spec.name!r} requires exactly one "
                "ObjectRelationshipDeclaration."
            )
        return (
            relationship_spec.artifact_type is RelationshipsArtifactType
            and declarations[0].projects_parent_child_measurements()
        )

    ignored_settings = ()

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        """Select the callable whose return ABI matches saved-child topology."""

        del contract, source_bindings
        save_children = optional_setting_value(module, cls.save_children_setting)
        function_name = (
            cls.function_variants[0]
            if save_children is not None and parse_cellprofiler_bool(save_children)
            else str(cls.function_name)
        )
        return cls.require_callable(function_name)

    @classmethod
    def relationship_inputs(
        cls,
        module: "ModuleBlock",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ArtifactSpec]:
        """Return the main relationship endpoints from their setting-owned roles."""

        return (
            artifact_inputs.require_by_name_and_artifact_type(
                required_setting_value(module, cls.parent_objects_setting),
                ObjectLabelsArtifactType,
            ),
            artifact_inputs.require_by_name_and_artifact_type(
                required_setting_value(module, cls.child_objects_setting),
                ObjectLabelsArtifactType,
            ),
        )

    @classmethod
    def other_parent_inputs(
        cls,
        module: "ModuleBlock",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Return active repeated parents in their declared setting order."""

        if not cls.other_parent_distances_enabled(module):
            return ()
        parent_names = setting_values(module, cls.other_parent_objects_setting)
        if not parent_names:
            raise ValueError(
                "RelateObjects enables distances to other parents but declares "
                f"no {cls.other_parent_objects_setting.canonical!r} setting row."
            )
        return tuple(
            artifact_inputs.require_by_name_and_artifact_type(
                name,
                ObjectLabelsArtifactType,
            )
            for name in parent_names
        )

    @classmethod
    def artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
    ):
        """Add exact relationship inputs supporting each repeated parent role."""

        inputs = ArtifactSpecCollection(
            super().artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
            )
        )
        main_parent, _child = cls.relationship_inputs(module, inputs)
        other_parents = cls.other_parent_inputs(module, inputs)
        if not other_parents:
            return inputs.specs

        other_parent_refs = frozenset(parent.ref() for parent in other_parents)
        declared_inputs = tuple(
            (
                replace(
                    spec,
                    relations=(
                        *spec.relations,
                        RelateObjectsDistanceParentInputRelation(
                            source=main_parent.ref()
                        ),
                    ),
                )
                if spec.ref() in other_parent_refs
                else spec
            )
            for spec in inputs.specs
        )

        available = ArtifactSpecCollection(
            ArtifactSpecCollection(
                (
                    *step_context.main_flow_artifacts.specs,
                    *step_context.available_artifacts.specs,
                )
            ).unique(conflict_context="RelateObjects step-parent relationship")
        )
        declarations = available.relation_refs(ObjectRelationshipDeclaration)
        supporting_relationships: list[ArtifactSpec] = []
        main_ref = main_parent.ref().for_plan_type(ArtifactInputPlan)
        for other_parent in other_parents:
            other_ref = other_parent.ref().for_plan_type(ArtifactInputPlan)
            matches = tuple(
                spec
                for spec, declaration in declarations
                if declaration.projects_parent_child_measurements()
                and {
                    declaration.source.for_plan_type(ArtifactInputPlan),
                    declaration.target.for_plan_type(ArtifactInputPlan),
                }
                == {main_ref, other_ref}
            )
            if len(matches) != 1:
                raise ValueError(
                    "RelateObjects repeated parent requires exactly one active "
                    "parent-child relationship with the main parent: "
                    f"{main_parent.name!r} <-> {other_parent.name!r}; got "
                    f"{tuple(spec.name for spec in matches)!r}."
                )
            supporting_relationships.append(matches[0].for_plan_type(ArtifactInputPlan))
        return (*declared_inputs, *supporting_relationships)

    @classmethod
    def prior_child_measurement_artifact_inputs(
        cls,
        module: "ModuleBlock",
        *,
        step_context: "ArtifactDeclarationStepContext",
        child_input: ArtifactSpec,
    ) -> tuple[ArtifactSpec, ...]:
        """Return prior measurement outputs declared against the child objects."""

        enabled = optional_setting_value(module, cls.per_parent_means_setting)
        if enabled is None or not parse_cellprofiler_bool(enabled):
            return ()
        child_ref = child_input.ref()
        return tuple(
            dict.fromkeys(
                producer.spec.for_plan_type(ArtifactInputPlan)
                for producer in step_context.available_artifact_producers
                if producer.spec.plan_type is ArtifactOutputPlan
                and producer.spec.artifact_type is MeasurementsArtifactType
                and any(
                    relation.source == child_ref for relation in producer.spec.relations
                )
            )
        )

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        """Run parent labels in the child object's declared invocation group."""

        inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        parent_input, child_input = cls.relationship_inputs(module, inputs)
        grouped_parent_refs = frozenset(
            parent.ref()
            for parent in (
                parent_input,
                *cls.other_parent_inputs(module, inputs),
            )
        )
        relationship_inputs = tuple(
            (
                spec.with_group_scope_relation(
                    InputGroupLineageSourceRelation(source=child_input.ref()),
                )
                if spec.ref() in grouped_parent_refs and spec.ref() != child_input.ref()
                else spec
            )
            for spec in inputs.specs
        )
        return (
            *relationship_inputs,
            *cls.prior_child_measurement_artifact_inputs(
                module,
                step_context=step_context,
                child_input=child_input,
            ),
        )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        parent_input, child_input = cls.relationship_inputs(module, artifact_inputs)
        parent_relationship = ObjectRelationshipDeclaration(
            source=parent_input.ref(),
            target=child_input.ref(),
            relationship_type=cls.parent_relationship_type,
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=module.module_num,
        )
        child_relationship = ObjectRelationshipDeclaration(
            source=child_input.ref(),
            target=parent_input.ref(),
            relationship_type=cls.child_relationship_type,
            source_role="child",
            target_role="parent",
            source_id_field="child_id",
            target_id_field="parent_id",
            producer_module_number=module.module_num,
        )
        outputs = [
            ArtifactSpec.output(
                parent_relationship.artifact_name(),
                RelationshipsArtifactType,
                relations=(
                    SourceStackLineageSourceRelation(source=child_input.ref()),
                    parent_relationship,
                ),
            ),
            ArtifactSpec.output(
                child_relationship.artifact_name(),
                RelationshipsArtifactType,
                relations=(
                    SourceStackLineageSourceRelation(source=child_input.ref()),
                    child_relationship,
                ),
            ),
            cls.measurement_output_artifact(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
        ]
        save_children = optional_setting_value(module, cls.save_children_setting)
        if save_children is not None and parse_cellprofiler_bool(save_children):
            output_objects = ArtifactSpec.output(
                required_setting_value(module, cls.output_object_setting),
                ObjectLabelsArtifactType,
                relations=(SourceStackLineageSourceRelation(source=child_input.ref()),),
            )
            outputs.insert(0, output_objects)
            outputs.insert(
                3,
                cls.parent_child_relationship_output_artifact(
                    module,
                    step_context=step_context,
                    parent=child_input,
                    child=output_objects,
                    lineage_source=child_input,
                ),
            )
        return tuple(outputs)


class RelateObjectsChildMeanFeatureDeclaration(RuntimeMeasurementFeatureDeclaration):
    """Parse and render RelateObjects child means at their measurement owner."""

    declaration_key = "relate_objects_child_mean"
    semantic_marker_types = (RelateObjectsChildMeanFeatureMarker,)

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> RelateObjectsChildMeanMeasurementFeature | None:
        template = RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD
        aggregate_prefix, separator, _remainder = template.value.partition("_")
        parts = tuple(feature_name.split("_"))
        if (
            not separator
            or len(parts) < 3
            or parts[0] != aggregate_prefix
            or any(not part for part in parts[1:])
        ):
            return None
        return RelateObjectsChildMeanMeasurementFeature(parts[1:])

    @classmethod
    def feature_name(cls, identity: object) -> str:
        if not isinstance(identity, RelateObjectsChildMeanMeasurementFeature):
            raise TypeError(
                f"{cls.__name__}.feature_name requires "
                "RelateObjectsChildMeanMeasurementFeature."
            )
        parts = identity.qualified_child_feature_parts
        return RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD.feature_name(
            child_object_name=parts[0],
            child_feature_name="_".join(parts[1:]),
        )


from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.measurement_feature_queries import (
    MeasurementAxisValueProjection,
    MeasurementFeatureQuery,
    MeasurementFeatureValueIndex,
    MeasurementTableObjectFeatureSemantics,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementSparseColumnarRows,
    measurement_object_label,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_output_matching import (
    RuntimeOutputBundle,
)
from openhcs.core.source_plane_alignment import (
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequenceAlignment,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)


class RelateObjectsDistanceAggregateFeatureSemantics(
    GenericRelationshipAggregateFeatureSemantics
):
    """RelateObjects parent qualification for child-distance aggregates."""

    strategy_key = "relate_objects_parent_qualified_distance"

    @staticmethod
    def _matching_feature(
        context: RelationshipAggregateFeatureContext,
    ) -> RelateObjectsModule.DistanceMeasurementFeature | None:
        if context.dialect is not CELLPROFILER_MEASUREMENT_DIALECT:
            return None
        return RelateObjectsModule.DistanceMeasurementFeature.matching_feature(
            context.feature_name,
            parent_object_name=context.source_name,
        )

    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        return self._matching_feature(context) is not None

    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        feature = self._matching_feature(context)
        assert feature is not None
        return (
            feature.unqualified_feature_name,
            normalize_runtime_identifier(
                feature.feature_name(parent_object_name=context.source_name)
            ),
        )

    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        feature = self._matching_feature(context)
        assert feature is not None
        return self.target_aggregate_feature_name(
            context.target_name,
            feature.unqualified_feature_name,
            aggregate=aggregate,
        )

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        feature = self._matching_feature(context)
        assert feature is not None
        return feature.unqualified_feature_name


from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    ObjectMeasurementTableIndex,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationshipPayloadKernel,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    measurement_row_mapping,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    measurement_axis_integer_domain,
    measurement_axis_integer_value,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

ParentObjectLabelsInput = Annotated[
    ObjectLabelValue,
    "Parent-object labels whose regions receive assigned child objects.",
]
ChildObjectLabelsInput = Annotated[
    ObjectLabelValue,
    "Child-object labels to assign to their containing parent regions.",
]


class ObjectRelationshipBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ObjectRelationshipPayloadKernel,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object relationship operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def relate_children_to_parents(
        self, parent_labels: np.ndarray, child_labels: np.ndarray, child_count: int
    ) -> np.ndarray:
        """Assign each child to its parent object."""

    @abstractmethod
    def centroid_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        """Return child-parent centroid distances."""

    @abstractmethod
    def minimum_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        """Return child-centroid to parent-boundary distances."""

    @abstractmethod
    def label_centers(self, labels: np.ndarray) -> np.ndarray:
        """Return row/column centers indexed by dense positive label id."""

    def parent_child_payload_from_labels(
        self, parent_labels: Any, child_labels: Any
    ) -> DirectedObjectRelationshipPayload:
        """Return parent-child ids using the labels' nominal representation."""
        return object_label_parent_child_payload(
            parent_labels, child_labels, kernel=self
        )

    def parents_of_from_payload(
        self, payload: DirectedObjectRelationshipPayload, child_count: int
    ) -> np.ndarray:
        """Return a dense parents-of-child vector from a relationship payload."""
        parents_of = np.zeros(child_count, dtype=np.int32)
        for parent_id, child_id in zip(
            payload.source_ids, payload.target_ids, strict=True
        ):
            if 0 < child_id <= child_count:
                parents_of[child_id - 1] = int(parent_id)
        return parents_of


class RelateObjectsRelationshipMeasurementRows(RelationshipMeasurementRows):
    """RelateObjects additionally projects configured child-parent distances."""

    def rows(self) -> ColumnarRows:
        row_batches: list[ColumnarRows] = [super().rows()]
        callable_contract = self.request.callable_contract
        module_type = CellProfilerModule.require_callable_contract_owner(
            callable_contract
        )
        declared_artifacts = callable_contract.artifact_specs
        for relationship_spec, declaration, payload in self.output_entries():
            if not module_type.relationship_distance_measurements_apply(
                callable_contract,
                relationship_spec,
            ):
                continue
            parent_spec = declared_artifacts.by_ref(declaration.source)
            child_spec = declared_artifacts.by_ref(declaration.target)
            if parent_spec is None or child_spec is None:
                raise ValueError(
                    f"Callable {callable_contract.function_name!r} relationship "
                    f"output {relationship_spec.name!r} references undeclared "
                    f"endpoints {declaration.source!r}, {declaration.target!r}."
                )
            if self.per_parent_means_enabled():
                row_batches.append(
                    self.parent_mean_upstream_measurement_rows(
                        parent_spec=parent_spec,
                        child_spec=child_spec,
                        payload=payload,
                    )
                )
            row_batches.append(
                self.distance_rows(
                    parent_spec=parent_spec,
                    child_spec=child_spec,
                    payload=payload,
                )
            )
            for (
                other_parent,
                supporting_relationship,
                supporting_declaration,
            ) in self.other_parent_relationship_inputs(
                callable_contract,
                main_parent_spec=parent_spec,
                child_spec=child_spec,
            ):
                other_parent_payload = self.compose_other_parent_payload(
                    main_parent_spec=parent_spec,
                    main_payload=payload,
                    other_parent_spec=other_parent,
                    supporting_relationship_spec=supporting_relationship,
                    supporting_declaration=supporting_declaration,
                )
                row_batches.append(
                    self.distance_rows(
                        parent_spec=other_parent,
                        child_spec=child_spec,
                        payload=other_parent_payload,
                        parent_mean_spec=parent_spec,
                        parent_mean_payload=payload,
                    )
                )
        return ConcatenatedColumnarRows(tuple(row_batches))

    @staticmethod
    def other_parent_relationship_inputs(
        callable_contract: "CallableContract",
        *,
        main_parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
    ) -> tuple[tuple[ArtifactSpec, ArtifactSpec, ObjectRelationshipDeclaration], ...]:
        """Return each repeated parent with its exact supporting relationship."""

        endpoint_refs = {main_parent_spec.ref(), child_spec.ref()}
        other_parents = tuple(
            spec
            for spec in callable_contract.artifact_inputs.of_artifact_type(
                ObjectLabelsArtifactType
            )
            if spec.ref() not in endpoint_refs
        )
        relationship_inputs = callable_contract.artifact_inputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        resolved: list[
            tuple[ArtifactSpec, ArtifactSpec, ObjectRelationshipDeclaration]
        ] = []
        for other_parent in other_parents:
            matches = tuple(
                (spec, declaration)
                for spec, declaration in relationship_inputs
                if declaration.projects_parent_child_measurements()
                and {declaration.source, declaration.target}
                == {main_parent_spec.ref(), other_parent.ref()}
            )
            if len(matches) != 1:
                raise ValueError(
                    "RelateObjects active repeated parent requires one exact "
                    "supporting relationship input: "
                    f"{main_parent_spec.name!r} <-> {other_parent.name!r}; got "
                    f"{tuple(spec.name for spec, _declaration in matches)!r}."
                )
            supporting_spec, supporting_declaration = matches[0]
            resolved.append((other_parent, supporting_spec, supporting_declaration))
        return tuple(resolved)

    def compose_other_parent_payload(
        self,
        *,
        main_parent_spec: ArtifactSpec,
        main_payload: ObjectRelationship,
        other_parent_spec: ArtifactSpec,
        supporting_relationship_spec: ArtifactSpec,
        supporting_declaration: ObjectRelationshipDeclaration,
    ) -> DirectedObjectRelationshipPayload:
        """Compose child-to-main and main-to-step-parent declarations exactly."""

        supporting_value = self.request.adapter.get_relationship(
            supporting_relationship_spec.name,
            artifact_type=supporting_relationship_spec.artifact_type,
        )
        main_slices, main_slice_count = self.relationship_pairs_by_slice(main_payload)
        supporting_slices, supporting_slice_count = self.relationship_pairs_by_slice(
            supporting_value
        )
        if (
            main_slice_count is not None
            and supporting_slice_count is not None
            and main_slice_count != supporting_slice_count
        ):
            raise ValueError(
                "RelateObjects repeated-parent relationship axis does not align "
                f"with the main relationship: {supporting_slice_count} != "
                f"{main_slice_count}."
            )
        if tuple(main_slices) != tuple(supporting_slices):
            raise ValueError(
                "RelateObjects repeated-parent relationship slices do not align "
                f"with the main relationship: {tuple(supporting_slices)!r} != "
                f"{tuple(main_slices)!r}."
            )

        main_ref = main_parent_spec.ref()
        other_ref = other_parent_spec.ref()
        parent_ids: list[int] = []
        child_ids: list[int] = []
        slice_indices: list[int] = []
        for slice_index, main_pairs in main_slices.items():
            step_parent_by_main: dict[int, int] = {}
            for source_id, target_id in supporting_slices[slice_index]:
                if (
                    supporting_declaration.source == other_ref
                    and supporting_declaration.target == main_ref
                ):
                    step_parent_by_main[int(target_id)] = int(source_id)
                elif (
                    supporting_declaration.source == main_ref
                    and supporting_declaration.target == other_ref
                ):
                    step_parent_by_main[int(source_id)] = int(target_id)
                else:
                    raise ValueError(
                        "RelateObjects supporting relationship declaration does not "
                        "match its repeated-parent endpoints."
                    )
            for main_parent_id, child_id in main_pairs:
                other_parent_id = step_parent_by_main.get(int(main_parent_id), 0)
                if other_parent_id <= 0:
                    continue
                parent_ids.append(other_parent_id)
                child_ids.append(int(child_id))
                if slice_index is not None:
                    slice_indices.append(slice_index)
        return DirectedObjectRelationshipPayload(
            source_ids=tuple(parent_ids),
            target_ids=tuple(child_ids),
            slice_indices=tuple(slice_indices),
            slice_count=main_slice_count,
        )

    @staticmethod
    def relationship_pairs_by_slice(
        value: ObjectRelationship | RuntimeSliceAlignedValues[ObjectRelationship],
    ) -> tuple[
        dict[int | None, tuple[tuple[int, int], ...]],
        int | None,
    ]:
        """Project one exact relationship value into ordered per-slice pairs."""

        if isinstance(value, RuntimeSliceAlignedValues):
            return (
                {
                    slice_index: tuple(
                        zip(
                            relationship.payload.source_ids,
                            relationship.payload.target_ids,
                            strict=True,
                        )
                    )
                    for slice_index, relationship in enumerate(value.slices)
                },
                value.slice_count,
            )
        payload = value.payload
        sliced_pairs = payload.runtime_slice_pairs()
        if sliced_pairs is None:
            return (
                {None: tuple(zip(payload.source_ids, payload.target_ids, strict=True))},
                None,
            )
        return (dict(sliced_pairs), payload.slice_count)

    def distance_rows(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        payload: ObjectRelationship | DirectedObjectRelationshipPayload,
        parent_mean_spec: ArtifactSpec | None = None,
        parent_mean_payload: ObjectRelationship | None = None,
    ) -> ColumnarRows:
        if not self.distance_measurements_declared():
            return MeasurementSparseColumnarRows.from_rows((), fields=())
        directed_payload = (
            payload.payload if isinstance(payload, ObjectRelationship) else payload
        )
        sliced_pairs = directed_payload.runtime_slice_pairs()
        parent_mean_slices = (
            None
            if parent_mean_payload is None
            else parent_mean_payload.payload.runtime_slice_pairs()
        )
        if parent_mean_payload is None:
            parent_mean_pairs_by_slice = {}
        elif parent_mean_slices is None:
            parent_mean_pairs_by_slice = {
                None: tuple(
                    zip(
                        parent_mean_payload.payload.source_ids,
                        parent_mean_payload.payload.target_ids,
                        strict=True,
                    )
                )
            }
        else:
            parent_mean_pairs_by_slice = dict(parent_mean_slices)
        if sliced_pairs is not None:
            slice_count = len(sliced_pairs)
            row_batches: list[ColumnarRows] = []
            for slice_index, pairs in sliced_pairs:
                row_batches.append(
                    self.distance_rows_for_pairs(
                        parent_spec=parent_spec,
                        child_spec=child_spec,
                        pairs=pairs,
                        slice_index=slice_index,
                        slice_count=slice_count,
                        parent_mean_spec=parent_mean_spec,
                        parent_mean_pairs=parent_mean_pairs_by_slice.get(slice_index),
                    )
                )
            return ConcatenatedColumnarRows(tuple(row_batches))
        unsliced_parent_mean_pairs = (
            None
            if parent_mean_payload is None
            else parent_mean_pairs_by_slice.get(None)
        )
        return self.distance_rows_for_pairs(
            parent_spec=parent_spec,
            child_spec=child_spec,
            pairs=tuple(
                (
                    (int(parent_id), int(child_id))
                    for parent_id, child_id in zip(
                        directed_payload.source_ids,
                        directed_payload.target_ids,
                        strict=True,
                    )
                )
            ),
            slice_index=None,
            parent_mean_spec=parent_mean_spec,
            parent_mean_pairs=unsliced_parent_mean_pairs,
        )

    def distance_method(self) -> RelateObjectsDistanceMethod:
        callable_contract = self.request.callable_contract
        module_type = CellProfilerModule.require_callable_contract_owner(
            callable_contract
        )
        func = module_type.require_callable(callable_contract.function_name)
        call_kwargs = {
            **runtime_callable_defaults(func),
            **self.request.call_kwargs,
        }
        return call_kwargs["calculate_distances"]

    def distance_measurements_declared(self) -> bool:
        distance_method = self.distance_method()
        return bool(
            distance_method.calculates_centroid_distance
            or distance_method.calculates_minimum_distance
        )

    def per_parent_means_enabled(self) -> bool:
        value = (
            self.request.call_kwargs["calculate_per_parent_means"]
            if "calculate_per_parent_means" in self.request.call_kwargs
            else False
        )
        return bool(value)

    def distance_rows_for_pairs(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        pairs: tuple[tuple[int, int], ...],
        slice_index: int | None,
        slice_count: int | None = None,
        parent_mean_spec: ArtifactSpec | None = None,
        parent_mean_pairs: tuple[tuple[int, int], ...] | None = None,
    ) -> ColumnarRows:
        if not pairs:
            return MeasurementSparseColumnarRows.from_rows((), fields=())
        parent_labels = self.object_labels(
            parent_spec, slice_index=slice_index, slice_count=slice_count
        )
        child_labels = self.object_labels(
            child_spec, slice_index=slice_index, slice_count=slice_count
        )
        (aligned_parent, aligned_child), _adapters = (
            SourceSpatialDomainAdapter.aligned_values((parent_labels, child_labels))
        )
        parent_array = np.asarray(aligned_parent, dtype=np.int32)
        child_array = np.asarray(aligned_child, dtype=np.int32)
        if parent_array.ndim != 2 or child_array.ndim != 2:
            raise ValueError(
                "RelateObjects distance rows require runtime-projected 2-D label planes."
            )
        parent_count = 0
        if child_array.size:
            parent_count = int(child_array.max())
        parents_of = np.zeros(parent_count, dtype=np.int32)
        for parent_id, child_id in pairs:
            if 0 < child_id <= len(parents_of):
                parents_of[child_id - 1] = parent_id
        backend = ObjectRelationshipBackendStrategy.for_memory_type()
        method = self.distance_method()
        centroid_distances = (
            backend.centroid_distances(parent_array, child_array, parents_of)
            if method.calculates_centroid_distance
            else None
        )
        minimum_distances = (
            backend.minimum_distances(parent_array, child_array, parents_of)
            if method.calculates_minimum_distance
            else None
        )
        centroid_feature = (
            RelateObjectsModule.DistanceMeasurementFeature.DISTANCE_CENTROID.feature_name(
                parent_object_name=parent_spec.name
            )
            if centroid_distances is not None
            else None
        )
        minimum_feature = (
            RelateObjectsModule.DistanceMeasurementFeature.DISTANCE_MINIMUM.feature_name(
                parent_object_name=parent_spec.name
            )
            if minimum_distances is not None
            else None
        )
        child_distance_rows = MeasurementSparseColumnarRows.from_rows(
            tuple(
                {
                    MeasurementRowAxisField.OBJECT_NAME.value: child_spec.name,
                    MeasurementRowAxisField.OBJECT_LABEL.value: child_id,
                    **(
                        {}
                        if slice_index is None
                        else {MeasurementRowAxisField.SLICE_INDEX.value: slice_index}
                    ),
                    **(
                        {}
                        if centroid_feature is None or centroid_distances is None
                        else {centroid_feature: float(centroid_distances[child_id - 1])}
                    ),
                    **(
                        {}
                        if minimum_feature is None or minimum_distances is None
                        else {minimum_feature: float(minimum_distances[child_id - 1])}
                    ),
                }
                for _parent_id, child_id in pairs
                if 0 < child_id <= len(parents_of)
            ),
            fields=(
                FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
                FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
                *(
                    ()
                    if slice_index is None
                    else (FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),)
                ),
                *(
                    ()
                    if centroid_feature is None
                    else (
                        RelateObjectsModule.DistanceMeasurementFeature.DISTANCE_CENTROID.field_spec(
                            centroid_feature,
                            required=True,
                        ),
                    )
                ),
                *(
                    ()
                    if minimum_feature is None
                    else (
                        RelateObjectsModule.DistanceMeasurementFeature.DISTANCE_MINIMUM.field_spec(
                            minimum_feature,
                            required=True,
                        ),
                    )
                ),
            ),
        )
        if not self.per_parent_means_enabled():
            return child_distance_rows
        return ConcatenatedColumnarRows(
            (
                child_distance_rows,
                self.parent_mean_distance_rows(
                    parent_object_name=(parent_mean_spec or parent_spec).name,
                    child_object_name=child_spec.name,
                    centroid_child_feature_name=centroid_feature,
                    minimum_child_feature_name=minimum_feature,
                    pairs=(pairs if parent_mean_pairs is None else parent_mean_pairs),
                    centroid_distances=centroid_distances,
                    minimum_distances=minimum_distances,
                    slice_index=slice_index,
                ),
            )
        )

    def parent_mean_distance_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        centroid_child_feature_name: str | None,
        minimum_child_feature_name: str | None,
        pairs: tuple[tuple[int, int], ...],
        centroid_distances: np.ndarray | None,
        minimum_distances: np.ndarray | None,
        slice_index: int | None,
    ) -> MeasurementSparseColumnarRows:
        feature_values = tuple(
            (feature_name, values)
            for feature_name, values in (
                (centroid_child_feature_name, centroid_distances),
                (minimum_child_feature_name, minimum_distances),
            )
            if feature_name is not None and values is not None
        )
        distances_by_parent: dict[int, dict[str, list[float]]] = {}
        for parent_id, child_id in pairs:
            if child_id <= 0:
                continue
            for feature_name, values in feature_values:
                if child_id > len(values):
                    continue
                distances_by_parent.setdefault(parent_id, {}).setdefault(
                    feature_name,
                    [],
                ).append(float(values[child_id - 1]))
        mean_features = tuple(
            (
                feature_name,
                RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD.feature_name(
                    child_object_name=child_object_name,
                    child_feature_name=feature_name,
                ),
            )
            for feature_name, _values in feature_values
        )
        rows = tuple(
            {
                MeasurementRowAxisField.OBJECT_NAME.value: parent_object_name,
                MeasurementRowAxisField.OBJECT_LABEL.value: parent_id,
                **(
                    {}
                    if slice_index is None
                    else {MeasurementRowAxisField.SLICE_INDEX.value: slice_index}
                ),
                **{
                    mean_feature: float(np.mean(distances[child_feature]))
                    for child_feature, mean_feature in mean_features
                    if distances.get(child_feature)
                },
            }
            for parent_id, distances in sorted(distances_by_parent.items())
            if distances
        )
        return MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=(
                FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
                FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
                *(
                    ()
                    if slice_index is None
                    else (FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),)
                ),
                *(
                    RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD.field_spec(
                        mean_feature,
                        required=False,
                    )
                    for _child_feature, mean_feature in mean_features
                ),
            ),
        )

    def parent_mean_upstream_measurement_rows(
        self,
        *,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
        payload: ObjectRelationship,
    ) -> MeasurementSparseColumnarRows:
        """Return CellProfiler per-parent means over prior child measurements."""
        values_by_child = self.upstream_child_measurement_values(child_spec)
        if not values_by_child:
            return MeasurementSparseColumnarRows.from_rows((), fields=())
        sliced_pairs = payload.payload.runtime_slice_pairs()
        relationship_slices = (
            sliced_pairs
            if sliced_pairs is not None
            else (
                (
                    None,
                    tuple(
                        zip(
                            payload.payload.source_ids,
                            payload.payload.target_ids,
                            strict=True,
                        )
                    ),
                ),
            )
        )
        rows: list[dict[str, object]] = []
        fields: list[FieldSpec] = [
            FieldSpec(MeasurementRowAxisField.OBJECT_NAME.value, str),
            FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
        ]
        if sliced_pairs is not None:
            fields.append(FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int))
        for slice_index, pairs in relationship_slices:
            value_slice_index = 0 if slice_index is None else slice_index
            feature_values_by_parent: dict[int, dict[str, list[float]]] = {}
            for parent_id, child_id in pairs:
                if int(parent_id) <= 0:
                    continue
                child_values = values_by_child.get(
                    (value_slice_index, int(child_id)),
                    {},
                )
                if not child_values:
                    continue
                parent_feature_values = feature_values_by_parent.setdefault(
                    int(parent_id),
                    {},
                )
                for feature_name, value in child_values.items():
                    parent_feature_values.setdefault(feature_name, []).append(value)
            for parent_id, feature_values in sorted(feature_values_by_parent.items()):
                means = {
                    RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD.feature_name(
                        child_object_name=child_spec.name,
                        child_feature_name=feature_name,
                    ): float(
                        np.mean(values)
                    )
                    for feature_name, values in sorted(feature_values.items())
                    if values
                }
                if means:
                    rows.append(
                        {
                            MeasurementRowAxisField.OBJECT_NAME.value: parent_spec.name,
                            MeasurementRowAxisField.OBJECT_LABEL.value: parent_id,
                            **(
                                {}
                                if slice_index is None
                                else {
                                    MeasurementRowAxisField.SLICE_INDEX.value: slice_index
                                }
                            ),
                            **means,
                        }
                    )
                    fields.extend(
                        RelateObjectsModule.AggregateMeasurementFeature.MEAN_CHILD.field_spec(
                            feature_name,
                            required=False,
                        )
                        for feature_name in means
                    )
        return MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=FieldSpec.merge_exact(
                (fields,),
                context="RelateObjects upstream mean fields",
            ),
        )

    def upstream_child_measurement_values(
        self,
        child_spec: ArtifactSpec,
    ) -> dict[tuple[int, int], dict[str, float]]:
        """Index upstream child values on the relationship's declared plane axis."""
        declared_tables: list[MeasurementTable] = []
        for spec in self.request.callable_contract.artifact_inputs.specs:
            if spec.artifact_type is not MeasurementsArtifactType:
                continue
            for record in self.request.adapter.artifact_input_records(
                spec.name,
                MeasurementsArtifactType,
            ):
                table = record.value.data
                if not isinstance(table, MeasurementTable):
                    raise TypeError(
                        f"Declared measurement input {spec.name!r} carries "
                        f"{type(table).__name__}, not MeasurementTable."
                    )
                declared_tables.append(table)
        tables = ObjectMeasurementTableIndex.from_tables(
            tuple(declared_tables)
        ).for_object(child_spec.name)
        if tables is None:
            raise RuntimeError(
                "A complete object-measurement table index returned an unknown "
                "selection."
            )
        child_labels = self.unprojected_object_labels(child_spec)
        core_rows = ObjectLocationMeasurementRows(
            child_labels,
            child_spec.name,
        )
        values_by_child: dict[tuple[int, int], dict[str, float]] = {}
        for raw_row in core_rows.rows().iter_row_mappings():
            row = measurement_row_mapping(raw_row)
            slice_index = measurement_axis_integer_value(
                row[MeasurementRowAxisField.SLICE_INDEX.value],
                MeasurementRowAxisField.SLICE_INDEX,
            )
            object_label = measurement_object_label(row)
            if slice_index is None or object_label is None:
                raise ValueError(
                    "CellProfiler core object rows require slice and object-label "
                    "identity."
                )
            values_by_child.setdefault((slice_index, object_label), {})[
                str(row[MeasurementRowAxisField.FEATURE_NAME.value])
            ] = float(row[MeasurementRowValueField.RESULT_VALUE.value])

        plane_domains = core_rows.label_plane_domains()
        payload_scoped = (
            child_labels.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD
        )
        for slice_index, _plane_domain in enumerate(plane_domains):
            object_numbers = self.object_numbers_by_label_id(
                child_spec,
                slice_index=None if payload_scoped else slice_index,
                slice_count=None if payload_scoped else len(plane_domains),
            )
            for object_label, object_number in object_numbers.items():
                values_by_child.setdefault((slice_index, object_label), {})[
                    CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value
                ] = float(object_number)

        if not tables:
            return values_by_child
        identity_policy = (
            self.request.adapter.request.context.source_image_set_identity_policy
        )
        child_axis = SourcePayloadPlaneIdentitySequence(
            child_labels,
            identity_policy,
        ).runtime_axis_identities()
        if not child_axis:
            raise ValueError(
                f"RelateObjects child measurements for {child_spec.name!r} require "
                "source-addressable object-label planes."
            )

        row_axis = MeasurementRowAxisField.SLICE_INDEX
        for table in tables:
            semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
            table_axis = table.source_provenance.image_set_axis(identity_policy)
            if not table_axis:
                raise ValueError(
                    "RelateObjects upstream measurement tables require complete "
                    "source image-set identity."
                )
            aligned_axis = SourcePlaneIdentitySequenceAlignment(
                table_axis,
                child_axis,
            ).target_indexes_for_image_planes()
            if aligned_axis is None:
                raise ValueError(
                    f"RelateObjects measurement table {table.name!r} does not align "
                    f"to child object planes for {child_spec.name!r}."
                )
            aggregate_features = tuple(
                sorted(
                    feature_name
                    for feature_name in semantics.feature_names
                    if RelateObjectsModule.aggregates_child_measurement_feature(
                        feature_name
                    )
                )
            )
            row_axis_values = (
                table.rows.column_values(row_axis.value)
                if row_axis.value in {field.name for field in table.rows.fields}
                else None
            )
            local_slice_indexes = (
                ()
                if row_axis_values is None
                else measurement_axis_integer_domain(row_axis_values, row_axis)
            )
            if row_axis_values is None or not local_slice_indexes:
                if len(aligned_axis) != 1:
                    raise ValueError(
                        f"RelateObjects measurement table {table.name!r} has "
                        "multi-plane source identity but axisless object rows."
                    )
                slice_projections = ((0, aligned_axis[0], None),)
            else:
                if len(aligned_axis) != 1 and any(
                    measurement_axis_integer_value(value, row_axis) is None
                    for value in row_axis_values
                ):
                    raise ValueError(
                        f"RelateObjects measurement table {table.name!r} has "
                        "multi-plane source identity but axisless object rows."
                    )
                invalid_slice_indexes = tuple(
                    slice_index
                    for slice_index in local_slice_indexes
                    if slice_index >= len(aligned_axis)
                )
                if invalid_slice_indexes:
                    raise ValueError(
                        f"RelateObjects measurement table {table.name!r} row axes "
                        f"{invalid_slice_indexes!r} exceed its source axis of "
                        f"{len(aligned_axis)} plane(s)."
                    )
                slice_projections = tuple(
                    (
                        local_slice_index,
                        aligned_axis[local_slice_index],
                        MeasurementAxisValueProjection(
                            row_axis,
                            local_slice_index,
                        ).mask(row_axis_values),
                    )
                    for local_slice_index in local_slice_indexes
                )

            queries = tuple(
                (
                    feature_name,
                    MeasurementFeatureQuery(
                        feature_name,
                        object_name=child_spec.name,
                        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
                    ),
                )
                for feature_name in aggregate_features
            )
            for _local_slice_index, target_slice_index, row_mask in slice_projections:
                for feature_name, query in queries:
                    feature_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
                        table,
                        query,
                        {child_spec.name: query.query_object_name},
                        row_mask=row_mask,
                        measurement_value_qualifier=(
                            RelateObjectsModule.aggregate_child_measurement_value_is_qualified
                        ),
                    )
                    feature_index = feature_indexes.get(child_spec.name)
                    if feature_index is None:
                        continue
                    for (
                        object_label,
                        numeric_value,
                    ) in feature_index.values_by_label.items():
                        child_values = values_by_child.setdefault(
                            (target_slice_index, object_label),
                            {},
                        )
                        previous = child_values.get(feature_name)
                        if previous is not None and not (
                            previous == numeric_value
                            or (np.isnan(previous) and np.isnan(numeric_value))
                        ):
                            raise ValueError(
                                f"RelateObjects child measurement {feature_name!r} "
                                f"has conflicting values for slice "
                                f"{target_slice_index}, object {object_label}: "
                                f"{previous!r} != {numeric_value!r}."
                            )
                        child_values[feature_name] = numeric_value
        return values_by_child


class NumbaNumpyObjectRelationshipBackendStrategy(ObjectRelationshipBackendStrategy):
    """Numba-accelerated NumPy object relationship primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        parent_labels = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
        child_labels = np.array([[1, 0, 0], [0, 2, 2], [0, 0, 3]], dtype=np.int32)
        parents_of = self.relate_children_to_parents(parent_labels, child_labels, 3)
        self.label_centers(parent_labels)
        self.centroid_distances(parent_labels, child_labels, parents_of)
        self.minimum_distances(parent_labels, child_labels, parents_of)

    def relate_children_to_parents(
        self, parent_labels: np.ndarray, child_labels: np.ndarray, child_count: int
    ) -> np.ndarray:
        parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
        parents_of = np.zeros(child_count, dtype=np.int32)
        if child_count == 0 or parent_count == 0:
            return parents_of
        return _relate_children_to_parents_numba(
            np.asarray(parent_labels),
            np.asarray(child_labels),
            child_count,
            parent_count,
        )

    def relate_sparse_ijv_children_to_parents(
        self,
        parent_rows: np.ndarray,
        child_rows: np.ndarray,
        child_count: int,
        parent_count: int,
    ) -> np.ndarray:
        if child_count == 0 or parent_count == 0:
            return np.zeros(child_count, dtype=np.int32)
        return _relate_sparse_ijv_children_to_parents_numba(
            np.asarray(parent_rows, dtype=np.int64),
            np.asarray(child_rows, dtype=np.int64),
            child_count,
            parent_count,
        )

    def centroid_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max())
        return _calculate_centroid_distances_numba(
            np.ascontiguousarray(parent_labels),
            np.ascontiguousarray(child_labels),
            np.asarray(parents_of, dtype=np.int32),
            parent_count,
        )

    def minimum_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max())
        return _calculate_minimum_distances_numba(
            np.ascontiguousarray(parent_labels),
            np.ascontiguousarray(child_labels),
            np.asarray(parents_of, dtype=np.int32),
            parent_count,
        )

    def label_centers(self, labels: np.ndarray) -> np.ndarray:
        if labels.ndim != 2:
            raise ValueError(
                "RelateObjects requires one runtime-projected 2-D label plane."
            )
        label_count = int(labels.max())
        if label_count == 0:
            return np.empty((0, 2), dtype=np.float64)
        centroids = _label_centroids_numba(np.ascontiguousarray(labels), label_count)
        return centroids[1:]


@dataclass(frozen=True, slots=True)
class RelateObjectsResult(RuntimeOutputBundle):
    """Nominal result bundle emitted by RelateObjects."""

    output_labels: ObjectLabelValue
    parent_child_relationship: DirectedObjectRelationshipPayload
    child_parent_relationship: DirectedObjectRelationshipPayload
    relationship_measurements: ColumnarRows
    saved_child_relationship: DirectedObjectRelationshipPayload | None = None

    def as_runtime_tuple(
        self,
    ) -> (
        tuple[
            ObjectLabelValue,
            DirectedObjectRelationshipPayload,
            DirectedObjectRelationshipPayload,
            ColumnarRows,
        ]
        | tuple[
            ObjectLabelValue,
            DirectedObjectRelationshipPayload,
            DirectedObjectRelationshipPayload,
            DirectedObjectRelationshipPayload,
            ColumnarRows,
        ]
    ):
        """Lower to the current positional function-contract ABI."""
        if self.saved_child_relationship is None:
            return (
                self.output_labels,
                self.parent_child_relationship,
                self.child_parent_relationship,
                self.relationship_measurements,
            )
        return (
            self.output_labels,
            self.parent_child_relationship,
            self.child_parent_relationship,
            self.saved_child_relationship,
            self.relationship_measurements,
        )


def _relate_objects_result(
    image: np.ndarray,
    parent_labels: ObjectLabelValue,
    child_labels: ObjectLabelValue,
    calculate_distances: RelateObjectsDistanceMethod = RelateObjectsDistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    save_children_with_parents: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
) -> RelateObjectsResult:
    """Relate CellProfiler child objects to parent objects by spatial overlap."""
    del calculate_distances, calculate_per_parent_means
    if not isinstance(parent_labels, ObjectLabelValue) or not isinstance(
        child_labels, ObjectLabelValue
    ):
        raise TypeError(
            "RelateObjects requires runtime-projected ObjectLabelValue inputs."
        )
    slice_index = 0 if slice_index is None else int(slice_index)
    raw_parent_labels = parent_labels
    raw_child_labels = child_labels
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider
    )
    parent_child_relationship = relationship_backend.parent_child_payload_from_labels(
        raw_parent_labels, raw_child_labels
    )
    (parent_labels, child_labels), _adapters = (
        SourceSpatialDomainAdapter.aligned_values((raw_parent_labels, raw_child_labels))
    )
    parent_labels = np.asarray(parent_labels, dtype=np.int32)
    child_labels = np.asarray(child_labels, dtype=np.int32)
    if parent_labels.ndim != 2 or child_labels.ndim != 2:
        raise ValueError("RelateObjects requires runtime-projected 2-D label planes.")
    child_count = int(child_labels.max()) if child_labels.max() > 0 else 0
    parents_of = relationship_backend.parents_of_from_payload(
        parent_child_relationship, child_count
    )
    saved_child_relationship: DirectedObjectRelationshipPayload | None = None
    if save_children_with_parents:
        retained_child_ids = np.flatnonzero(
            np.concatenate((np.zeros(1, dtype=bool), parents_of > 0))
        ).astype(np.int32, copy=False)
        label_indexes = np.zeros(child_count + 1, dtype=np.int32)
        label_indexes[retained_child_ids] = np.arange(
            1, len(retained_child_ids) + 1, dtype=np.int32
        )
        child_index = np.asarray(child_labels, dtype=np.intp)
        output_labels = label_indexes[child_index]
        saved_child_relationship = (
            relationship_backend.parent_child_payload_from_labels(
                child_labels, output_labels
            )
        )
    else:
        output_labels = child_labels.copy()
    measurements = MeasurementSparseColumnarRows.from_rows((), fields=())
    related_child_ids = tuple(
        (
            child_idx
            for child_idx, parent_idx in enumerate(parents_of, start=1)
            if parent_idx > 0
        )
    )
    related_parent_ids = tuple(
        (int(parent_idx) for parent_idx in parents_of if parent_idx > 0)
    )
    if (
        parent_child_relationship.slice_indices
        or parent_child_relationship.slice_count is not None
    ):
        related_relationship = parent_child_relationship
    else:
        related_relationship = DirectedObjectRelationshipPayload(
            source_ids=related_parent_ids,
            target_ids=related_child_ids,
            slice_indices=tuple((slice_index for _child_id in related_child_ids)),
            slice_count=slice_index + 1,
        )
    output_labels = object_label_value_with_dense_labels(
        raw_child_labels, output_labels.astype(np.float32)
    )
    return RelateObjectsResult(
        output_labels,
        related_relationship,
        DirectedObjectRelationshipPayload(
            source_ids=related_child_ids,
            target_ids=related_parent_ids,
            slice_indices=tuple(slice_index for _child_id in related_child_ids),
            slice_count=slice_index + 1,
        ),
        measurements,
        saved_child_relationship=saved_child_relationship,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(SliceIndexRuntimeParameter)
@special_inputs("parent_labels", "child_labels")
def relate_objects(
    image: np.ndarray,
    parent_labels: ParentObjectLabelsInput,
    child_labels: ChildObjectLabelsInput,
    calculate_distances: RelateObjectsDistanceMethod = RelateObjectsDistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    calculate_distances_to_other_parents: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
) -> tuple[
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
    DirectedObjectRelationshipPayload,
    ColumnarRows,
]:
    """Relate child objects to parents without declaring a saved object set."""

    del calculate_distances_to_other_parents

    result = _relate_objects_result(
        image,
        parent_labels,
        child_labels,
        calculate_distances=calculate_distances,
        calculate_per_parent_means=calculate_per_parent_means,
        save_children_with_parents=False,
        relationship_backend_provider=relationship_backend_provider,
        slice_index=slice_index,
    )
    return (
        result.output_labels,
        result.parent_child_relationship,
        result.child_parent_relationship,
        result.relationship_measurements,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(SliceIndexRuntimeParameter)
@special_inputs("parent_labels", "child_labels")
def relate_objects_with_saved_children(
    image: np.ndarray,
    parent_labels: ParentObjectLabelsInput,
    child_labels: ChildObjectLabelsInput,
    calculate_distances: RelateObjectsDistanceMethod = RelateObjectsDistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    calculate_distances_to_other_parents: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
) -> tuple[
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
    DirectedObjectRelationshipPayload,
    DirectedObjectRelationshipPayload,
    ColumnarRows,
]:
    """Relate child objects to parents and emit the saved-child topology."""

    del calculate_distances_to_other_parents

    result = _relate_objects_result(
        image,
        parent_labels,
        child_labels,
        calculate_distances=calculate_distances,
        calculate_per_parent_means=calculate_per_parent_means,
        save_children_with_parents=True,
        relationship_backend_provider=relationship_backend_provider,
        slice_index=slice_index,
    )
    if result.saved_child_relationship is None:
        raise RuntimeError(
            "RelateObjects saved-child execution omitted its relationship."
        )
    return (
        result.output_labels,
        result.parent_child_relationship,
        result.child_parent_relationship,
        result.saved_child_relationship,
        result.relationship_measurements,
    )


@njit(cache=True)
def _relate_sparse_ijv_children_to_parents_numba(
    parent_ijv: np.ndarray, child_ijv: np.ndarray, child_count: int, parent_count: int
) -> np.ndarray:
    counts = np.zeros((child_count + 1, parent_count + 1), dtype=np.int32)
    parent_linear = _sparse_ijv_linear_coordinates(parent_ijv, child_ijv)
    child_linear = _sparse_ijv_linear_coordinates(child_ijv, parent_ijv)
    parent_order = np.argsort(parent_linear)
    child_order = np.argsort(child_linear)
    parent_position = 0
    child_position = 0
    while parent_position < parent_order.size and child_position < child_order.size:
        parent_index = parent_order[parent_position]
        child_index = child_order[child_position]
        parent_coordinate = parent_linear[parent_index]
        child_coordinate = child_linear[child_index]
        if parent_coordinate < child_coordinate:
            parent_position += 1
            continue
        if child_coordinate < parent_coordinate:
            child_position += 1
            continue
        parent_end = parent_position + 1
        while (
            parent_end < parent_order.size
            and parent_linear[parent_order[parent_end]] == parent_coordinate
        ):
            parent_end += 1
        child_end = child_position + 1
        while (
            child_end < child_order.size
            and child_linear[child_order[child_end]] == child_coordinate
        ):
            child_end += 1
        for grouped_parent_position in range(parent_position, parent_end):
            grouped_parent_index = parent_order[grouped_parent_position]
            parent_id = int(parent_ijv[grouped_parent_index, 2])
            if parent_id <= 0 or parent_id > parent_count:
                continue
            for grouped_child_position in range(child_position, child_end):
                grouped_child_index = child_order[grouped_child_position]
                child_id = int(child_ijv[grouped_child_index, 2])
                if child_id > 0 and child_id <= child_count:
                    counts[child_id, parent_id] += 1
        parent_position = parent_end
        child_position = child_end
    return _parents_of_from_overlap_counts_numba(counts, child_count, parent_count)


@njit(cache=True)
def _sparse_ijv_linear_coordinates(
    rows: np.ndarray, peer_rows: np.ndarray
) -> np.ndarray:
    max_y = 0
    for index in range(rows.shape[0]):
        y = int(rows[index, 0])
        if y > max_y:
            max_y = y
    for index in range(peer_rows.shape[0]):
        y = int(peer_rows[index, 0])
        if y > max_y:
            max_y = y
    dim_y = max_y + 1
    linear = np.empty(rows.shape[0], dtype=np.int64)
    for index in range(rows.shape[0]):
        linear[index] = int(rows[index, 0]) + dim_y * int(rows[index, 1])
    return linear


@njit(cache=True)
def _relate_children_to_parents_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    child_count: int,
    parent_count: int,
) -> np.ndarray:
    counts = np.zeros((child_count + 1, parent_count + 1), dtype=np.int32)
    height, width = child_labels.shape
    for row in range(height):
        for col in range(width):
            child_id = int(child_labels[row, col])
            parent_id = int(parent_labels[row, col])
            if (
                child_id > 0
                and child_id <= child_count
                and (parent_id > 0)
                and (parent_id <= parent_count)
            ):
                counts[child_id, parent_id] += 1
    return _parents_of_from_overlap_counts_numba(counts, child_count, parent_count)


@njit(cache=True)
def _parents_of_from_overlap_counts_numba(
    counts: np.ndarray, child_count: int, parent_count: int
) -> np.ndarray:
    parents_of = np.zeros(child_count, dtype=np.int32)
    for child_id in range(1, child_count + 1):
        best_parent = 0
        best_count = 0
        for parent_id in range(1, parent_count + 1):
            overlap = counts[child_id, parent_id]
            if overlap > best_count:
                best_count = overlap
                best_parent = parent_id
        parents_of[child_id - 1] = best_parent
    return parents_of


@njit(cache=True)
def _label_centroids_numba(labels: np.ndarray, label_count: int) -> np.ndarray:
    sums = np.zeros((label_count + 1, 2), dtype=np.float64)
    counts = np.zeros(label_count + 1, dtype=np.int64)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > 0 and label_id <= label_count:
                sums[label_id, 0] += row
                sums[label_id, 1] += col
                counts[label_id] += 1
    centroids = np.empty((label_count + 1, 2), dtype=np.float64)
    for label_id in range(label_count + 1):
        if counts[label_id] == 0:
            centroids[label_id, 0] = np.nan
            centroids[label_id, 1] = np.nan
        else:
            centroids[label_id, 0] = sums[label_id, 0] / counts[label_id]
            centroids[label_id, 1] = sums[label_id, 1] / counts[label_id]
    return centroids


@njit(cache=True)
def _calculate_centroid_distances_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray,
    parent_count: int,
) -> np.ndarray:
    child_count = len(parents_of)
    distances = np.empty(child_count, dtype=np.float64)
    for child_idx in range(child_count):
        distances[child_idx] = np.nan
    if child_count == 0 or parent_count == 0:
        return distances
    parent_centroids = _label_centroids_numba(parent_labels, parent_count)
    child_centroids = _label_centroids_numba(child_labels, child_count)
    for child_idx in range(child_count):
        parent_id = int(parents_of[child_idx])
        child_id = child_idx + 1
        if parent_id > 0 and parent_id <= parent_count:
            child_row = child_centroids[child_id, 0]
            child_col = child_centroids[child_id, 1]
            parent_row = parent_centroids[parent_id, 0]
            parent_col = parent_centroids[parent_id, 1]
            if not (
                np.isnan(child_row)
                or np.isnan(child_col)
                or np.isnan(parent_row)
                or np.isnan(parent_col)
            ):
                row_delta = child_row - parent_row
                col_delta = child_col - parent_col
                distances[child_idx] = np.sqrt(
                    row_delta * row_delta + col_delta * col_delta
                )
    return distances


@njit(cache=True)
def _is_inner_boundary_pixel(
    labels: np.ndarray, row: int, col: int, label_id: int
) -> bool:
    height, width = labels.shape
    if row > 0 and int(labels[row - 1, col]) != label_id:
        return True
    if row + 1 < height and int(labels[row + 1, col]) != label_id:
        return True
    if col > 0 and int(labels[row, col - 1]) != label_id:
        return True
    if col + 1 < width and int(labels[row, col + 1]) != label_id:
        return True
    return False


@njit(cache=True)
def _calculate_minimum_distances_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray,
    parent_count: int,
) -> np.ndarray:
    child_count = len(parents_of)
    distances = np.empty(child_count, dtype=np.float64)
    for child_idx in range(child_count):
        distances[child_idx] = np.nan
    if child_count == 0 or parent_count == 0:
        return distances
    child_centroids = _label_centroids_numba(child_labels, child_count)
    height, width = parent_labels.shape
    counts = np.zeros(parent_count + 1, dtype=np.int64)
    for row in range(height):
        for col in range(width):
            parent_id = int(parent_labels[row, col])
            if (
                parent_id > 0
                and parent_id <= parent_count
                and _is_inner_boundary_pixel(parent_labels, row, col, parent_id)
            ):
                counts[parent_id] += 1
    offsets = np.zeros(parent_count + 2, dtype=np.int64)
    for parent_id in range(1, parent_count + 1):
        offsets[parent_id + 1] = offsets[parent_id] + counts[parent_id]
    total = offsets[parent_count + 1]
    rows = np.empty(total, dtype=np.float64)
    cols = np.empty(total, dtype=np.float64)
    write_offsets = offsets.copy()
    for row in range(height):
        for col in range(width):
            parent_id = int(parent_labels[row, col])
            if (
                parent_id > 0
                and parent_id <= parent_count
                and _is_inner_boundary_pixel(parent_labels, row, col, parent_id)
            ):
                offset = write_offsets[parent_id]
                rows[offset] = row
                cols[offset] = col
                write_offsets[parent_id] += 1
    for child_idx in range(child_count):
        parent_id = int(parents_of[child_idx])
        child_id = child_idx + 1
        if parent_id <= 0 or parent_id > parent_count:
            continue
        child_row = child_centroids[child_id, 0]
        child_col = child_centroids[child_id, 1]
        if np.isnan(child_row) or np.isnan(child_col):
            continue
        start = offsets[parent_id]
        end = offsets[parent_id + 1]
        if start == end:
            continue
        min_distance_sq = np.inf
        for offset in range(start, end):
            row_delta = rows[offset] - child_row
            col_delta = cols[offset] - child_col
            distance_sq = row_delta * row_delta + col_delta * col_delta
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
        distances[child_idx] = np.sqrt(min_distance_sq)
    return distances


__all__ = public_names_from_objects(
    DistanceMethod,
    NumbaNumpyObjectRelationshipBackendStrategy,
    ObjectRelationshipBackendStrategy,
    RelateObjectsResult,
    relate_objects,
    relate_objects_with_saved_children,
)
