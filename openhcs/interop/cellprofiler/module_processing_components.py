"""CellProfiler module processing-component lowering.

This module owns generic source-axis and runtime-artifact lowering queried by
``pipeline_generator`` when emitting ``FunctionStep`` declarations. Concrete
CellProfiler module rules belong on the module declaration that represents the
module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import re
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import MeasurementsArtifactType
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_set import ComponentSet
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.pipeline_image_schema import (
    PipelineImageSchema,
    SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS,
)
from openhcs.core.source_bindings import (
    SourceBindingMatchMethod,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_matching import source_metadata_component
from openhcs.interop.cellprofiler.symbol_table import (
    ModuleArtifactContracts,
    step_source_bindings_literal,
)
from openhcs.processing.backends.cellprofiler.library import (
    require_function,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    SliceBySliceRuntimeParameter,
)


GeneratedLiteralScalar: TypeAlias = str | int | float | bool | None | Enum
GeneratedLiteralValue: TypeAlias = (
    "GeneratedLiteralScalar | tuple[GeneratedLiteralValue, ...] | "
    "list[GeneratedLiteralValue] | "
    "dict[GeneratedLiteralValue, GeneratedLiteralValue]"
)
GeneratedParameterName = str | list[str] | None
GeneratedGroupByComponent: TypeAlias = (
    "AllComponents | GroupBy | GeneratedGroupByComponentState"
)


class GeneratedGroupByComponentState(Enum):
    """Internal state for group_by before module defaults are applied."""

    UNRESOLVED = "unresolved"


def group_by_is_unresolved(group_by: GeneratedGroupByComponent) -> bool:
    return group_by is GeneratedGroupByComponentState.UNRESOLVED


def variable_component_literal(component: AllComponents) -> str:
    return f"VariableComponents.{component.name}"


def all_component_literal(component: AllComponents) -> str:
    return f"AllComponents.{component.name}"


def coerce_all_component(component: AllComponents | VariableComponents) -> AllComponents:
    if isinstance(component, AllComponents):
        return component
    return AllComponents(component.value)


def all_component_tuple_literal(components: tuple[AllComponents, ...]) -> str:
    if not components:
        return "()"
    literals = tuple(all_component_literal(component) for component in components)
    if len(literals) == 1:
        return f"({literals[0]},)"
    return "(" + ", ".join(literals) + ")"


def group_by_literal(group_by: GeneratedGroupByComponent) -> str | None:
    if group_by_is_unresolved(group_by):
        return None
    return f"GroupBy.{group_by.name}"


def group_by_component_axis(
    group_by: GeneratedGroupByComponent,
) -> AllComponents | None:
    if group_by_is_unresolved(group_by):
        return None
    if isinstance(group_by, AllComponents):
        return group_by
    if group_by.value is None:
        return None
    return AllComponents(group_by.value)


class SourceProcessingAxisRole(Enum):
    """Semantic role for generated OpenHCS source-processing axes."""

    SAMPLE_GROUP = "sample_group"
    IMAGE_SET = "image_set"
    SOURCE_STACK = "source_stack"


@dataclass(frozen=True, slots=True)
class SourceProcessingAxisPlan:
    """Components that lower CellProfiler source-image roles into OpenHCS axes."""

    sample_group_component: AllComponents | None
    image_set_components: tuple[AllComponents, ...] = ()
    source_stack_components: tuple[AllComponents, ...] = ()
    source_alias_components: tuple[AllComponents, ...] = ()
    use_default_axes: bool = False

    @classmethod
    def from_schema(
        cls,
        source_schema: PipelineImageSchema,
        source_bindings: StepSourceBindingsConfig | None = None,
    ) -> "SourceProcessingAxisPlan":
        semantics = SourceProcessingComponentSemantics(source_schema, source_bindings)
        return cls(
            sample_group_component=semantics.sample_group_component(),
            image_set_components=semantics.image_set_components(),
            source_stack_components=source_schema.source_stack_components,
            source_alias_components=semantics.source_alias_components(),
            use_default_axes=source_schema.is_empty,
        )

    def without_source_set_components(
        self,
        components: Iterable[AllComponents],
    ) -> tuple[AllComponents, ...]:
        return (
            ComponentSet.collect(components)
            .excluding(
                ComponentSet.collect((self.sample_group_component,)),
                ComponentSet(self.image_set_components),
            )
            .as_tuple()
        )

    def scalar_source_group_component(
        self,
        components: Iterable[AllComponents],
    ) -> AllComponents | None:
        """Return the grouping axis for scalar source-image invocations."""
        if self.sample_group_component is not None:
            return self.sample_group_component
        source_identity_components = (
            ComponentSet.collect(components)
            .excluding(ComponentSet(self.source_stack_components))
        )
        return source_identity_components.single_or_none(
            "Scalar CellProfiler source processing cannot infer one group-by "
            "component from multiple source identity axes: "
            f"{tuple(component.value for component in source_identity_components)!r}."
        )

    def optional_single_image_set_component(
        self,
        error_message: str,
    ) -> AllComponents | None:
        """Return the image-set component when the schema declares one."""
        image_set_components = ComponentSet(self.image_set_components)
        if not image_set_components and self.use_default_axes:
            image_set_components = ComponentSet.default_group_by()
        if not image_set_components:
            return None
        return image_set_components.single_or_none(error_message)

    def single_component_for_role(
        self,
        role: SourceProcessingAxisRole,
    ) -> AllComponents:
        components = SourceProcessingAxisRolePolicy.for_role(role).components(self)
        error_message = (
            f"Source-processing role {role.value!r} must resolve to exactly "
            f"one component, got {tuple(component.value for component in components)!r}."
        )
        component = ComponentSet(components).single_or_none(error_message)
        if component is None:
            raise ValueError(error_message)
        return component


class SourceProcessingAxisRolePolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal source-axis role lowering policy."""

    __registry_key__ = "role"
    __skip_if_no_key__ = True

    role: ClassVar[SourceProcessingAxisRole | None] = None

    @classmethod
    def for_role(
        cls,
        role: SourceProcessingAxisRole,
    ) -> "SourceProcessingAxisRolePolicy":
        policy_type = cls.__registry__.get(role)
        if policy_type is None:
            raise ValueError(f"Unsupported source-processing axis role: {role!r}.")
        return policy_type()

    @abstractmethod
    def components(self, axis_plan: SourceProcessingAxisPlan) -> tuple[AllComponents, ...]:
        """Return axis components owned by this source-processing role."""


class SampleGroupAxisRolePolicy(SourceProcessingAxisRolePolicy):
    """Lower sample-group role to the schema-declared sample component."""

    role = SourceProcessingAxisRole.SAMPLE_GROUP

    def components(self, axis_plan: SourceProcessingAxisPlan) -> tuple[AllComponents, ...]:
        if axis_plan.sample_group_component is None:
            return ()
        return (axis_plan.sample_group_component,)


class ImageSetAxisRolePolicy(SourceProcessingAxisRolePolicy):
    """Lower image-set role to source-image set components."""

    role = SourceProcessingAxisRole.IMAGE_SET

    def components(self, axis_plan: SourceProcessingAxisPlan) -> tuple[AllComponents, ...]:
        if axis_plan.image_set_components:
            return axis_plan.image_set_components
        if axis_plan.use_default_axes:
            return ComponentSet.default_group_by().as_tuple()
        raise ValueError(
            "CellProfiler source schema does not declare an image-set "
            "component for a category that requires one."
        )


class SourceStackAxisRolePolicy(SourceProcessingAxisRolePolicy):
    """Lower source-stack role to source-image stack components."""

    role = SourceProcessingAxisRole.SOURCE_STACK

    def components(self, axis_plan: SourceProcessingAxisPlan) -> tuple[AllComponents, ...]:
        if axis_plan.source_stack_components:
            return axis_plan.source_stack_components
        if axis_plan.use_default_axes:
            return (AllComponents.Z_INDEX,)
        raise ValueError(
            "CellProfiler source schema does not declare source-stack "
            "components for a category that requires them."
        )


@dataclass(frozen=True, slots=True)
class SourceProcessingComponentSemantics:
    """Derive component roles from source-schema metadata instead of names."""

    source_schema: PipelineImageSchema
    source_bindings: StepSourceBindingsConfig | None = None

    def sample_group_component(self) -> AllComponents | None:
        component = ComponentSet(self.grouping_components()).variable().last()
        if component is not None:
            return component
        if self.uses_ordinal_image_sets_without_declared_identity():
            return SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS[-1]
        if self.source_schema.is_empty:
            return ComponentSet.default_variable().last()
        return None

    def image_set_components(self) -> tuple[AllComponents, ...]:
        components = (
            ComponentSet(self.source_metadata_components())
            .excluding(
                ComponentSet(self.grouping_components()),
                ComponentSet(self.source_schema.source_stack_components),
            )
            .variable()
            .as_tuple()
        )
        if components:
            return components
        if self.uses_implicit_source_alias_axis():
            return (AllComponents.CHANNEL,)
        return components

    def source_alias_components(self) -> tuple[AllComponents, ...]:
        """Return image-set axes that select between named source aliases."""
        selector_components = (
            ComponentSet(self.assignment_selector_components())
            .excluding(
                ComponentSet(self.source_schema.source_stack_components),
                ComponentSet(self.grouping_components()),
            )
            .variable()
            .as_tuple()
        )
        if selector_components:
            return selector_components
        image_set_components = ComponentSet(self.image_set_components())
        explicit_components = (
            ComponentSet(self.assignment_selector_components())
            .intersection(image_set_components)
            .as_tuple()
        )
        if explicit_components:
            return explicit_components
        if self.uses_implicit_source_alias_axis():
            return (AllComponents.CHANNEL,)
        if len(self.source_schema.assignments_by_alias) > 1:
            return image_set_components.as_tuple()
        return ()

    def assignment_selector_components(self) -> tuple[AllComponents, ...]:
        selectors = tuple(
            assignment.selector
            for assignment in self.source_schema.assignments_by_alias.values()
        )
        if self.source_bindings is not None:
            selectors = (
                *selectors,
                *(
                    binding.selector
                    for binding in self.source_bindings.binding_declarations
                ),
            )
        return tuple(
            dict.fromkeys(
                component
                for selector in selectors
                for component in self.selector_components(selector)
            )
        )

    def selector_components(
        self,
        selector: SourceSelector,
    ) -> tuple[AllComponents, ...]:
        return ComponentSet.collect(
            (component_selector.component for component_selector in selector.components),
            (
                source_metadata_component(metadata_selector.field)
                for metadata_selector in selector.metadata
            ),
        ).as_tuple()

    def uses_ordinal_image_sets_without_declared_identity(self) -> bool:
        """Return whether ordered CP image sets need OpenHCS ordinal site identity."""
        match_plan = self.match_plan()
        if match_plan is not None and match_plan.method is not SourceBindingMatchMethod.ORDER:
            return False
        if self.source_metadata_components():
            return False
        if (
            self.source_bindings is not None
            and not self.source_bindings.image_stack_bindings
        ):
            return False
        if match_plan is not None:
            return True
        return len(self.source_schema.loaded_image_aliases) > 1

    def uses_implicit_source_alias_axis(self) -> bool:
        """Return whether CP source aliases lower to OpenHCS channel identity."""
        if not self.uses_ordinal_image_sets_without_declared_identity():
            return False
        loaded_aliases = self.source_schema.loaded_image_aliases
        return len(loaded_aliases) > 1

    def variable_components(self) -> tuple[AllComponents, ...]:
        return ComponentSet.collect(
            (self.sample_group_component(),),
            self.image_set_components(),
        ).as_tuple()

    def source_metadata_components(self) -> tuple[AllComponents, ...]:
        return ComponentSet.collect(
            self.metadata_rule_components(),
            self.match_plan_components(),
            self.grouping_plan_components(),
        ).as_tuple()

    def grouping_components(self) -> tuple[AllComponents, ...]:
        declared_components = ComponentSet.collect(
            self.match_plan_components(),
            self.grouping_plan_components(),
        )
        if declared_components:
            return declared_components.as_tuple()
        return self.configured_default_group_components()

    def configured_default_group_components(self) -> tuple[AllComponents, ...]:
        return (
            ComponentSet.default_variable()
            .intersection(ComponentSet(self.source_metadata_components()))
            .as_tuple()
        )

    def metadata_rule_components(self) -> tuple[AllComponents, ...]:
        return tuple(
            dict.fromkeys(
                component
                for rule in self.metadata_rules()
                for field_name in re.compile(rule.pattern).groupindex
                for component in (source_metadata_component(field_name),)
                if component is not None
            )
        )

    def match_plan_components(self) -> tuple[AllComponents, ...]:
        match_plan = self.match_plan()
        if match_plan is None:
            return ()
        return tuple(
            dict.fromkeys(
                component
                for dimension in match_plan.dimensions
                for field in dimension.fields
                for component in (source_metadata_component(field.metadata_field),)
                if component is not None
            )
        )

    def grouping_plan_components(self) -> tuple[AllComponents, ...]:
        grouping = self.source_schema.grouping
        if grouping is None:
            return ()
        return tuple(
            dict.fromkeys(
                component
                for field in grouping.metadata_fields
                for component in (source_metadata_component(field),)
                if component is not None
            )
        )

    def metadata_rules(self):
        if self.source_bindings is not None and self.source_bindings.metadata_rules:
            return self.source_bindings.metadata_rules
        return self.source_schema.metadata_rules

    def match_plan(self):
        if self.source_bindings is not None and self.source_bindings.match_plan is not None:
            return self.source_bindings.match_plan
        return self.source_schema.match_plan


@dataclass(frozen=True, slots=True)
class ModuleProcessingComponents:
    """Generated OpenHCS processing-component literals for one module."""

    variable_components: tuple[AllComponents, ...]
    group_by_component: GeneratedGroupByComponent = (
        GeneratedGroupByComponentState.UNRESOLVED
    )

    def has_group_by_resolution(self) -> bool:
        return not group_by_is_unresolved(self.group_by_component)

    @property
    def variable_component_literals(self) -> tuple[str, ...]:
        return tuple(
            variable_component_literal(component)
            for component in self.variable_components
        )

    @property
    def group_by_literal(self) -> str | None:
        return group_by_literal(self.group_by_component)

    def execution_components(self) -> ComponentSet:
        """Return every component that can split this module invocation."""
        return ComponentSet.collect(
            self.variable_components,
            (group_by_component_axis(self.group_by_component),),
        )

    def with_variable_components(
        self,
        components: Iterable[AllComponents | VariableComponents],
    ) -> "ModuleProcessingComponents":
        """Return this declaration with explicit variable components."""
        return ModuleProcessingComponents(
            tuple(coerce_all_component(component) for component in components),
            self.group_by_component,
        )

    def with_group_by(
        self,
        group_by: AllComponents | GroupBy,
    ) -> "ModuleProcessingComponents":
        """Return this declaration with generated group_by semantics applied."""
        return ModuleProcessingComponents(
            self.variable_components,
            group_by,
        )

    def with_required_variable_components(
        self,
        components: Iterable[AllComponents | VariableComponents],
        *,
        module_name: str,
    ) -> "ModuleProcessingComponents":
        """Return components with declaration-required variable axes included."""
        required_components = tuple(
            coerce_all_component(component) for component in components
        )
        merged_components = ComponentSet.collect(
            self.variable_components,
            required_components,
        ).as_tuple()
        return ModuleProcessingComponents(
            merged_components,
            self.group_by_component,
        ).validate_required_variable_components(
            required_components,
            module_name=module_name,
        )

    def validate_required_variable_components(
        self,
        components: Iterable[AllComponents | VariableComponents],
        *,
        module_name: str,
    ) -> "ModuleProcessingComponents":
        """Fail when resolved processing config omits declaration-required axes."""
        required_components = tuple(
            coerce_all_component(component) for component in components
        )
        missing_components = tuple(
            component
            for component in required_components
            if component not in self.variable_components
        )
        if missing_components:
            required_literals = ", ".join(
                variable_component_literal(component)
                for component in required_components
            )
            actual_literals = ", ".join(self.variable_component_literals) or "none"
            raise ValueError(
                f"{module_name} requires variable_components "
                f"{required_literals}; resolved {actual_literals}."
            )
        return self


@dataclass(frozen=True, slots=True)
class RuntimeArtifactLineageScope:
    """Source-derived runtime lineage available to a generated module."""

    contract: ModuleArtifactContracts
    variable_components: tuple[AllComponents, ...] = ()
    requires_pairwise_object_domain_scope: bool = False

    def with_variable_components(
        self,
        variable_components: tuple[AllComponents, ...],
    ) -> "RuntimeArtifactLineageScope":
        return RuntimeArtifactLineageScope(
            self.contract,
            variable_components,
            self.requires_pairwise_object_domain_scope,
        )

    def variable_components_without_source_alias(
        self,
        axis_plan: SourceProcessingAxisPlan,
    ) -> tuple[AllComponents, ...]:
        """Named CellProfiler artifacts already carry source-alias identity."""
        return tuple(
            component
            for component in self.variable_components
            if component not in axis_plan.source_alias_components
        )


@dataclass(frozen=True, slots=True)
class GeneratedStepSettings:
    """Generated function kwargs with CellProfiler setting-value literal semantics."""

    entries: tuple[tuple[str, GeneratedLiteralValue], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, GeneratedLiteralValue],
    ) -> "GeneratedStepSettings":
        return cls(tuple(values.items()))

    def __bool__(self) -> bool:
        return bool(self.entries)

    def items(self) -> Iterable[tuple[str, GeneratedLiteralValue]]:
        return self.entries

    def with_defaults(
        self,
        values: Mapping[str, GeneratedLiteralValue],
    ) -> "GeneratedStepSettings":
        """Return settings with source-derived defaults added when absent."""
        existing_names = {name for name, _value in self.entries}
        additions = tuple(
            (name, value)
            for name, value in values.items()
            if name not in existing_names
        )
        if not additions:
            return self
        return GeneratedStepSettings((*self.entries, *additions))

    def value(
        self,
        name: str,
        default: GeneratedLiteralValue,
    ) -> GeneratedLiteralValue:
        for setting_name, value in self.entries:
            if setting_name == name:
                return value
        return default

    def without_dead_output_settings(
        self,
        *,
        dead_settings: Iterable[str],
        param_mapping: Mapping[str, GeneratedParameterName],
    ) -> "GeneratedStepSettings":
        pruned_values = dict(self.entries)
        for setting_name in dead_settings:
            parameter_target = GeneratedParameterTarget.from_setting(
                setting_name,
                param_mapping,
            )
            parameter_target.prune(pruned_values)
        return GeneratedStepSettings.from_mapping(pruned_values)


@dataclass(frozen=True, slots=True)
class GeneratedParameterTarget:
    """Generated Python parameter name(s) controlled by one CP setting row."""

    parameter_names: tuple[str, ...]

    @classmethod
    def from_setting(
        cls,
        setting_name: str,
        param_mapping: Mapping[str, GeneratedParameterName],
    ) -> "GeneratedParameterTarget":
        if setting_name not in param_mapping:
            return cls((setting_name,))
        mapped_parameter = param_mapping[setting_name]
        if mapped_parameter is None:
            return cls(())
        if isinstance(mapped_parameter, list):
            return cls(tuple(mapped_parameter))
        return cls((mapped_parameter,))

    def prune(self, values: dict[str, GeneratedLiteralValue]) -> None:
        for parameter_name in self.parameter_names:
            values.pop(parameter_name, None)


@dataclass(frozen=True, slots=True)
class ModuleProcessingComponentRequest:
    """Typed request for lowering CellProfiler module semantics to OpenHCS axes."""

    module_type: type[object]
    function_name: str
    runtime_lineage: RuntimeArtifactLineageScope
    bound_settings: GeneratedStepSettings
    source_schema: PipelineImageSchema

    def has_direct_source_bindings(self) -> bool:
        """Return whether this module's CPPipe source bindings own axis selection."""
        return not self.runtime_lineage.contract.source_bindings.is_empty

    def object_label_measurement_execution(self) -> ObjectLabelMeasurementExecution:
        return object_label_measurement_execution_from_callable(
            self.resolved_callable()
        )

    def processing_contract(self) -> ProcessingContract | None:
        return CallableContract.from_callable(self.resolved_callable()).processing_contract

    def generated_semantic_control_defaults(self) -> Mapping[str, GeneratedLiteralValue]:
        """Return source-schema defaults for callable semantic-control parameters."""
        if self.processing_contract() is not ProcessingContract.FLEXIBLE:
            return {}
        return {
            SliceBySliceRuntimeParameter.require_parameter_name(): (
                not self.uses_volumetric_source_semantics()
            )
        }

    def uses_volumetric_source_semantics(self) -> bool:
        """Return whether CP setup declares true source-stack semantics."""
        source_bindings = self.runtime_lineage.contract.source_bindings
        if source_bindings.is_empty:
            return bool(self.source_stack_components())
        return bool(self.source_stack_components(source_bindings))

    def runtime_image_execution_mode(self) -> ImagePayloadExecutionMode | None:
        return CallableContract.from_callable(
            self.resolved_callable()
        ).runtime_image_execution_mode

    def requires_full_stack_object_measurement(self) -> bool:
        return (
            self.object_label_measurement_execution()
            is ObjectLabelMeasurementExecution.FULL_STACK
        )

    def requires_source_stack_collapse(self) -> bool:
        if self.requires_full_stack_object_measurement():
            return True
        if self.processing_contract() is ProcessingContract.PURE_3D:
            return True
        return (
            self.runtime_image_execution_mode() is ImagePayloadExecutionMode.FULL_STACK
            and bool(self.runtime_lineage.contract.runtime_artifact_inputs)
        )

    def resolved_callable(self) -> Callable:
        return require_function(
            self.runtime_lineage.contract.module_name,
            function_name=self.function_name,
        )

    def module_default_components(self) -> tuple[AllComponents, ...]:
        return tuple(
            coerce_all_component(component)
            for component in self.module_type.default_variable_components
        )

    def source_stack_components(
        self,
        source_bindings: StepSourceBindingsConfig | None = None,
    ) -> tuple[AllComponents, ...]:
        if (
            source_bindings is not None
            and not source_bindings.image_stack_bindings
        ):
            return ()
        return self.source_schema.source_stack_components

    def axis_plan(
        self,
        source_bindings: StepSourceBindingsConfig | None = None,
    ) -> SourceProcessingAxisPlan:
        return SourceProcessingAxisPlan.from_schema(
            self.source_schema,
            source_bindings,
        )

    def runtime_artifact_scope(
        self,
        *,
        module_requires_pairwise_object_domain_scope: bool,
    ) -> "RuntimeArtifactProcessingScope":
        lineage_components = self.runtime_lineage.variable_components
        source_bindings = self.runtime_lineage.contract.source_bindings
        if not source_bindings.is_empty:
            lineage_components = ComponentSet.collect(
                lineage_components,
                SourceProcessingComponentSemantics(
                    self.source_schema,
                    source_bindings,
                ).variable_components(),
            ).as_tuple()
        return RuntimeArtifactProcessingScope(
            self.runtime_lineage.with_variable_components(lineage_components),
            self.axis_plan(),
            module_requires_pairwise_object_domain_scope,
        )


@dataclass(frozen=True, slots=True)
class RuntimeArtifactProcessingScope:
    """OpenHCS execution scope for modules driven only by runtime artifacts."""

    lineage: RuntimeArtifactLineageScope
    axis_plan: SourceProcessingAxisPlan
    module_requires_pairwise_object_domain_scope: bool = False

    def components(self) -> ModuleProcessingComponents:
        variable_components = self.lineage.variable_components_without_source_alias(
            self.axis_plan
        )
        return self.with_runtime_source_identity(
            ModuleProcessingComponents(variable_components),
        )

    def with_runtime_source_identity(
        self,
        components: ModuleProcessingComponents,
    ) -> ModuleProcessingComponents:
        """Return runtime artifact components without declaring phantom stack axes."""
        return components

    def requires_pairwise_object_domain_scope(self) -> bool:
        object_domain_inputs = tuple(
            spec
            for spec in self.lineage.contract.runtime_artifact_inputs
            if spec.artifact_type.participates_in_pairwise_object_domain_input
        )
        object_domain_outputs = tuple(
            spec
            for spec in self.lineage.contract.outputs
            if spec.artifact_type.participates_in_object_domain_scope
        )
        return len(object_domain_inputs) > 1 and bool(object_domain_outputs)

    def uses_measurement_only_runtime_inputs(self) -> bool:
        return bool(self.lineage.contract.runtime_artifact_inputs) and all(
            spec.artifact_type is MeasurementsArtifactType
            for spec in self.lineage.contract.runtime_artifact_inputs
        )


@dataclass(frozen=True, slots=True)
class SourceBindingProcessingScope:
    """OpenHCS execution scope implied by CellProfiler source bindings."""

    source_bindings: StepSourceBindingsConfig
    source_schema: PipelineImageSchema
    axis_plan: SourceProcessingAxisPlan
    source_stack_components: tuple[AllComponents, ...] = ()

    def components(self) -> ModuleProcessingComponents:
        declared_components = SourceProcessingComponentSemantics(
            self.source_schema,
            self.source_bindings,
        ).variable_components()
        if self.requires_image_set_stack():
            source_alias_components = (
                self.axis_plan.source_alias_components
                or self.axis_plan.image_set_components
            )
            return self.with_source_identity_grouping(
                ModuleProcessingComponents(
                    ComponentSet.collect(
                        source_alias_components,
                        self.source_stack_components,
                    ).as_tuple(),
                )
            )
        if self.source_bindings.requires_pipeline_start_resolution:
            source_stack_scope = self.source_stack_processing_scope(declared_components)
            if source_stack_scope is not None:
                return self.with_source_identity_grouping(source_stack_scope)
            return self.scalar_source_binding_components(declared_components)
        source_stack_scope = self.source_stack_processing_scope(declared_components)
        if source_stack_scope is not None:
            return self.with_source_identity_grouping(source_stack_scope)
        return self.scalar_source_binding_components(declared_components)

    def scalar_source_binding_components(
        self,
        declared_components: tuple[AllComponents, ...],
    ) -> ModuleProcessingComponents:
        """Lower scalar source bindings without discarding image-set variables."""
        variable_components = (
            ComponentSet(declared_components)
            .excluding(
                ComponentSet(self.axis_plan.source_stack_components),
                ComponentSet(self.axis_plan.source_alias_components),
            )
            .as_tuple()
        )
        return self.with_source_identity_grouping(
            ModuleProcessingComponents(variable_components)
        )

    def source_stack_processing_scope(
        self,
        declared_components: tuple[AllComponents, ...],
    ) -> ModuleProcessingComponents | None:
        if not self.source_stack_components:
            return None
        source_stack_components = ComponentSet(
            self.source_stack_components,
        ).excluding(
            ComponentSet(self.axis_plan.source_alias_components),
        ).as_tuple()
        return ModuleProcessingComponents(
            source_stack_components,
        )

    def with_source_identity_grouping(
        self,
        components: ModuleProcessingComponents,
    ) -> ModuleProcessingComponents:
        """Preserve source identity outside the generated stack axis."""
        group_by_component = self.source_identity_group_by_component(
            components.variable_components
        )
        if group_by_component is None:
            return components
        return components.with_group_by(group_by_component)

    def source_identity_group_by_component(
        self,
        variable_components: tuple[AllComponents, ...],
    ) -> AllComponents | None:
        """Return the remaining source axis after bindings choose stack axes."""
        variable_component_set = ComponentSet(variable_components)
        group_by_candidates = (
            ComponentSet.collect(
                self.axis_plan.source_alias_components,
                self.axis_plan.image_set_components,
            )
            .excluding(variable_component_set)
            .as_tuple()
        )
        if not group_by_candidates and self.axis_plan.sample_group_component is not None:
            group_by_candidates = (
                ComponentSet.collect((self.axis_plan.sample_group_component,))
                .excluding(variable_component_set)
                .as_tuple()
            )
        return ComponentSet(group_by_candidates).single_or_none(
            "CellProfiler source bindings cannot infer one group_by component from "
            "multiple remaining source identity axes: "
            f"{tuple(component.value for component in group_by_candidates)!r}."
        )

    def requires_image_set_stack(self) -> bool:
        """Return whether one function call consumes multiple source-image aliases."""
        return (
            self.has_multi_image_binding_set()
            and self.source_bindings.requires_step_input_component_stack(
                self.axis_plan.image_set_components
            )
        ) or self.source_bindings.requires_pipeline_start_image_set_stack

    def has_multi_image_binding_set(self) -> bool:
        return (
            sum(
                1
                for binding in self.source_bindings.bindings
                if binding.participates_in_execution_anchoring
            )
            > 1
        )


def default_module_requires_pairwise_object_domain_scope(
    contract: ModuleArtifactContracts,
) -> bool:
    """Return the generic object-domain scope rule for a module contract."""
    object_domain_inputs = tuple(
        spec
        for spec in contract.runtime_artifact_inputs
        if spec.artifact_type.participates_in_pairwise_object_domain_input
    )
    object_domain_outputs = tuple(
        spec
        for spec in contract.outputs
        if spec.artifact_type.participates_in_object_domain_scope
    )
    return len(object_domain_inputs) > 1 and bool(object_domain_outputs)


def default_module_processing_components(
    request: ModuleProcessingComponentRequest,
    *,
    module_requires_pairwise_object_domain_scope: bool | None = None,
) -> ModuleProcessingComponents:
    """Lower a module contract to the default generated FunctionStep declaration."""
    if module_requires_pairwise_object_domain_scope is None:
        module_requires_pairwise_object_domain_scope = (
            default_module_requires_pairwise_object_domain_scope(
                request.runtime_lineage.contract,
            )
        )
    components = ModuleProcessingScopePolicy.for_request(request).components(
        request,
        module_requires_pairwise_object_domain_scope=(
            module_requires_pairwise_object_domain_scope
        ),
    )
    return components


class ModuleProcessingScopePolicy(ABC, metaclass=AutoRegisterMeta):
    """Auto-registered precedence policy for default module execution scope."""

    __registry_key__ = "policy_name"
    __skip_if_no_key__ = True
    policy_name: ClassVar[str]

    @classmethod
    def for_request(
        cls,
        request: ModuleProcessingComponentRequest,
    ) -> "ModuleProcessingScopePolicy":
        for policy_type in cls.policy_types_by_mro():
            policy = policy_type()
            if policy.matches(request):
                return policy
        raise RuntimeError("No module processing scope policy matched request.")

    @classmethod
    def policy_types_by_mro(cls) -> tuple[type["ModuleProcessingScopePolicy"], ...]:
        registered = set(cls.__registry__.values())
        ordered: list[type[ModuleProcessingScopePolicy]] = []
        seen: set[type[ModuleProcessingScopePolicy]] = set()

        def visit(owner: type[ModuleProcessingScopePolicy]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if owner in registered and owner not in seen:
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @abstractmethod
    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        """Return whether this policy owns the request."""

    @abstractmethod
    def components(
        self,
        request: ModuleProcessingComponentRequest,
        *,
        module_requires_pairwise_object_domain_scope: bool,
    ) -> ModuleProcessingComponents:
        """Return generated processing-component literals for the policy."""


class RuntimeArtifactModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Runtime artifacts determine scope before direct source bindings."""

    policy_name = "runtime_artifact"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        return (
            bool(request.runtime_lineage.contract.runtime_artifact_inputs)
            and request.runtime_lineage.contract.source_bindings.is_empty
        )

    def components(
        self,
        request: ModuleProcessingComponentRequest,
        *,
        module_requires_pairwise_object_domain_scope: bool,
    ) -> ModuleProcessingComponents:
        components = request.runtime_artifact_scope(
            module_requires_pairwise_object_domain_scope=(
                module_requires_pairwise_object_domain_scope
            ),
        ).components()
        if components.variable_components:
            return components
        return components.with_variable_components(request.module_default_components())


class SourceBindingModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Direct CellProfiler source bindings determine scope for source steps."""

    policy_name = "source_binding"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        return not request.runtime_lineage.contract.source_bindings.is_empty

    def components(
        self,
        request: ModuleProcessingComponentRequest,
        *,
        module_requires_pairwise_object_domain_scope: bool,
    ) -> ModuleProcessingComponents:
        del module_requires_pairwise_object_domain_scope
        return SourceBindingProcessingScope(
            request.runtime_lineage.contract.source_bindings,
            request.source_schema,
            request.axis_plan(request.runtime_lineage.contract.source_bindings),
            request.source_stack_components(
                request.runtime_lineage.contract.source_bindings,
            ),
        ).components()


class InputlessArtifactModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Artifact-only aggregate modules execute once per site axis."""

    policy_name = "inputless_artifact"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        return _is_inputless_artifact_only_contract(request.runtime_lineage.contract)

    def components(
        self,
        request: ModuleProcessingComponentRequest,
        *,
        module_requires_pairwise_object_domain_scope: bool,
    ) -> ModuleProcessingComponents:
        del module_requires_pairwise_object_domain_scope
        return ModuleProcessingComponents(
            request.module_default_components(),
        )


class ModuleDefaultProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Module declaration defaults fill the remaining pure image cases."""

    policy_name = "module_default"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        del request
        return True

    def components(
        self,
        request: ModuleProcessingComponentRequest,
        *,
        module_requires_pairwise_object_domain_scope: bool,
    ) -> ModuleProcessingComponents:
        del module_requires_pairwise_object_domain_scope
        return ModuleProcessingComponents(
            tuple(request.module_default_components()),
        )


def source_binding_variable_component_literals(
    source_bindings: StepSourceBindingsConfig,
    source_schema: PipelineImageSchema = PipelineImageSchema.empty(),
) -> tuple[str, ...]:
    """Return generated-code variable-component literals for source bindings."""
    return variable_component_literals(
        SourceProcessingComponentSemantics(
            source_schema,
            source_bindings,
        ).variable_components()
    )


def variable_component_literals(
    components: Iterable[AllComponents],
) -> tuple[str, ...]:
    """Return VariableComponents literals for source-schema component contracts."""
    return tuple(
        variable_component_literal(component)
        for component in components
        if component.name in VariableComponents.__members__
    )


def generated_function_step_semantic_argument_lines(
    *,
    processing_components: ModuleProcessingComponents,
    artifact_contract: ModuleArtifactContracts,
    import_collector: set[tuple[str, str]] | None = None,
) -> tuple[str, ...]:
    """Return generated FunctionStep arguments owned by source semantics."""
    lines: list[str] = []
    if not artifact_contract.source_bindings.is_empty:
        lines.append(
            "        source_bindings="
            f"{step_source_bindings_literal(artifact_contract.source_bindings, import_collector=import_collector)},"
        )
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactSourceLineage:
    """Variable-component scope inherited from source-bound artifact ancestry."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContracts]
    source_schema: PipelineImageSchema = PipelineImageSchema.empty()
    module_requires_pairwise_object_domain_scope: Callable[
        [ModuleArtifactContracts],
        bool,
    ] = default_module_requires_pairwise_object_domain_scope

    def variable_components_for(
        self,
        contract: ModuleArtifactContracts,
    ) -> tuple[AllComponents, ...]:
        components: list[AllComponents] = []
        self._collect(contract, components, set())
        return tuple(components)

    def requires_pairwise_object_domain_scope_for(
        self,
        contract: ModuleArtifactContracts,
    ) -> bool:
        return self._collect_pairwise_object_domain_scope(contract, set())

    def _collect(
        self,
        contract: ModuleArtifactContracts,
        components: list[AllComponents],
        seen_module_nums: set[int],
    ) -> None:
        if contract.module_num in seen_module_nums:
            return
        seen_module_nums.add(contract.module_num)

        if not contract.source_bindings.is_empty:
            source_semantics = SourceProcessingComponentSemantics(
                self.source_schema,
                contract.source_bindings,
            )
            source_components = (
                self.source_schema.source_stack_components
                if (
                    contract.source_bindings.image_stack_bindings
                    and self.source_schema.source_stack_components
                )
                else source_semantics.variable_components()
            )
            self._extend_unique(components, source_components)

        for symbol in contract.input_symbols:
            producer_module_num = symbol.producer_module_num
            if producer_module_num is None:
                continue
            producer_contract = self.contracts_by_module_num.get(producer_module_num)
            if producer_contract is None:
                continue
            self._collect(producer_contract, components, seen_module_nums)

    def _collect_pairwise_object_domain_scope(
        self,
        contract: ModuleArtifactContracts,
        seen_module_nums: set[int],
    ) -> bool:
        if contract.module_num in seen_module_nums:
            return False
        seen_module_nums.add(contract.module_num)

        if self.module_requires_pairwise_object_domain_scope(contract):
            return True

        for symbol in contract.input_symbols:
            producer_module_num = symbol.producer_module_num
            if producer_module_num is None:
                continue
            producer_contract = self.contracts_by_module_num.get(producer_module_num)
            if producer_contract is None:
                continue
            if self._collect_pairwise_object_domain_scope(
                producer_contract,
                seen_module_nums,
            ):
                return True
        return False

    @staticmethod
    def _extend_unique(
        components: list[AllComponents],
        values: Iterable[AllComponents],
    ) -> None:
        components[:] = ComponentSet.collect(components, values).as_tuple()


def _is_inputless_artifact_only_contract(contract: ModuleArtifactContracts) -> bool:
    """Return whether a step should execute once per source sample."""
    return (
        not contract.inputs
        and not contract.runtime_artifact_inputs
        and bool(contract.outputs)
        and all(
            spec.artifact_type.supports_inputless_artifact_only_execution
            for spec in contract.outputs
        )
    )
