"""
PipelineGenerator - Generate complete runnable OpenHCS pipelines.

DETERMINISTIC ONLY:
Uses pre-absorbed cellprofiler_library. No LLM fallback.
Fails loudly if modules are missing from the absorbed library.

Takes parsed .cppipe modules and generates a complete pipeline file with:
- All imports
- Function references from absorbed library
- FunctionStep wrappers with correct variable_components (from LLM-inferred category)
- Pipeline configuration
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, TypeAlias

from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import CallableContract
from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.core.component_set import ComponentSet
from openhcs.core.artifact_materialization_policy import (
    DEFAULT_ARTIFACT_MATERIALIZATION_RULES,
    NO_ARTIFACT_MATERIALIZATION,
)
from openhcs.core.artifact_observability import externally_required_artifact_outputs
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.pipeline_image_schema import (
    PipelineImageSchema,
    SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS,
)
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.source_bindings import (
    SourceBindingMatchMethod,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_matching import source_metadata_component
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.runtime import (
    CellProfilerGridCycleScope,
    CellProfilerInvocationOptions,
)
from openhcs.interop.cellprofiler.module_roles import (
    cellprofiler_infrastructure_import_note,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock

from openhcs.interop.cellprofiler.illumination_settings import (
    IlluminationCalculationScope,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

from openhcs.interop.cellprofiler.artifact_semantics import artifact_setting_symbols
from openhcs.interop.cellprofiler.module_function_resolution import (
    _ModuleFunctionResolutionStrategy,
)
from openhcs.interop.cellprofiler.module_settings_binding import (
    ModuleSettingCoverageRecord,
    _ModuleSettingsBindingStrategy,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.save_images_settings import (
    SAVE_IMAGES_SOURCE_IMAGE_SETTING,
)
from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    require_function,
    validated_contracts,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import (
    MaterializedFilenameIdentity,
    tiff_stack,
)
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    module_contract_literal,
    source_bindings_literal,
)

logger = logging.getLogger(__name__)


RegistryMetadataValue: TypeAlias = str | int | float | bool | None
GeneratedLiteralScalar: TypeAlias = str | int | float | bool | None | Enum
GeneratedLiteralValue: TypeAlias = (
    "GeneratedLiteralScalar | tuple[GeneratedLiteralValue, ...] | "
    "list[GeneratedLiteralValue] | "
    "dict[GeneratedLiteralValue, GeneratedLiteralValue]"
)
AbsorbedRegistryRecord: TypeAlias = Mapping[str, RegistryMetadataValue]
GeneratedParameterName = str | list[str] | None


@dataclass(frozen=True, slots=True)
class AbsorbedRegistryRecordView:
    """Typed reader for absorbed-library registry metadata records."""

    record: AbsorbedRegistryRecord

    def required_string(self, field_name: str) -> str:
        return str(self.record[field_name])

    def optional_string(self, field_name: str, default: str) -> str:
        if field_name not in self.record:
            return default
        return str(self.record[field_name])

    def optional_float(self, field_name: str, default: float) -> float:
        if field_name not in self.record:
            return default
        return float(self.record[field_name])

    def optional_bool(self, field_name: str, default: bool) -> bool:
        if field_name not in self.record:
            return default
        return bool(self.record[field_name])


@dataclass(frozen=True)
class ArtifactSpecKey:
    """Scope-free artifact identity used while pruning generated CP steps."""

    kind: ArtifactKind
    name: str

    @classmethod
    def from_spec(cls, spec: ArtifactSpec) -> ArtifactSpecKey:
        return cls(kind=spec.kind, name=spec.name)


ExternallyMaterializedOutputs = frozenset[ArtifactSpecKey]
ArtifactNameMaterializedOutputs = frozenset[ArtifactSpecKey]


@dataclass(frozen=True)
class AbsorbedModuleMetadata:
    """Validated absorbed-library metadata needed by generated pipelines."""

    function_name: str
    contract: str = "pure_2d"
    category: str = "image_operation"
    confidence: float = 0.5

    @classmethod
    def from_registry_record(
        cls,
        info: AbsorbedRegistryRecord,
    ) -> AbsorbedModuleMetadata:
        record = AbsorbedRegistryRecordView(info)
        return cls(
            function_name=record.required_string("function_name"),
            contract=record.optional_string("contract", "pure_2d"),
            category=record.optional_string("category", "image_operation"),
            confidence=record.optional_float("confidence", 0.5),
        )


_INPUTLESS_ARTIFACT_ONLY_KINDS = frozenset(
    {
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.RELATIONSHIPS,
    }
)
def variable_component_literal(component: AllComponents) -> str:
    return f"VariableComponents.{component.name}"


def all_component_literal(component: AllComponents) -> str:
    return f"AllComponents.{component.name}"


def all_component_tuple_literal(components: tuple[AllComponents, ...]) -> str:
    if not components:
        return "()"
    literals = tuple(all_component_literal(component) for component in components)
    if len(literals) == 1:
        return f"({literals[0]},)"
    return "(" + ", ".join(literals) + ")"


def group_by_literal(component: AllComponents | None) -> str:
    if component is None:
        return "GroupBy.NONE"
    return f"GroupBy.{component.name}"


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

    def source_identity_stack_components(self) -> tuple[AllComponents, ...]:
        """Return axes intentionally stacked into one scalar source identity."""
        return ComponentSet.collect(
            self.image_set_components,
            self.source_stack_components,
        ).as_tuple()

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

    def optional_single_image_set_component(self, error_message: str) -> AllComponents | None:
        """Return the image-set component when the schema declares one."""
        if not self.image_set_components:
            return None
        return ComponentSet(self.image_set_components).single_or_none(error_message)

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
        return tuple(
            dict.fromkeys(
                component
                for assignment in self.source_schema.assignments_by_alias.values()
                for component in self.selector_components(assignment.selector)
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


class CategoryVariableComponentProvider(ABC, metaclass=AutoRegisterMeta):
    """Nominal provider for absorbed-module category execution axes."""

    __registry_key__ = "category"
    __skip_if_no_key__ = True
    category: ClassVar[str | None] = None

    @classmethod
    def components_for_category(
        cls,
        category: str,
        axis_plan: SourceProcessingAxisPlan,
    ) -> tuple[AllComponents, ...]:
        provider_type = cls.__registry__.get(category)
        if provider_type is None:
            return SourceProcessingAxisRolePolicy.for_role(
                SourceProcessingAxisRole.SAMPLE_GROUP
            ).components(axis_plan)
        return provider_type().components(axis_plan)

    @abstractmethod
    def roles(self) -> tuple[SourceProcessingAxisRole, ...]:
        """Return variable components for this absorbed-module category."""

    def components(
        self,
        axis_plan: SourceProcessingAxisPlan,
    ) -> tuple[AllComponents, ...]:
        return ComponentSet.collect(
            *(
                SourceProcessingAxisRolePolicy.for_role(role).components(axis_plan)
                for role in self.roles()
            )
        ).as_tuple()


class ImageOperationCategoryVariableComponentProvider(CategoryVariableComponentProvider):
    """CellProfiler image-operation category executes over sample sites."""

    category = "image_operation"

    def roles(self) -> tuple[SourceProcessingAxisRole, ...]:
        return (SourceProcessingAxisRole.SAMPLE_GROUP,)


class ZProjectionCategoryVariableComponentProvider(CategoryVariableComponentProvider):
    """CellProfiler z-projection category executes over source Z planes."""

    category = "z_projection"

    def roles(self) -> tuple[SourceProcessingAxisRole, ...]:
        return (SourceProcessingAxisRole.SOURCE_STACK,)


class ChannelOperationCategoryVariableComponentProvider(CategoryVariableComponentProvider):
    """CellProfiler channel-operation category executes over image-set channels."""

    category = "channel_operation"

    def roles(self) -> tuple[SourceProcessingAxisRole, ...]:
        return (SourceProcessingAxisRole.IMAGE_SET,)


@dataclass(frozen=True, slots=True)
class ModuleProcessingComponents:
    """Generated OpenHCS processing-component literals for one module."""

    variable_components: tuple[AllComponents, ...]
    group_by_component: AllComponents | None
    source_identity_stack_components: tuple[AllComponents, ...] = ()

    @property
    def variable_component_literals(self) -> tuple[str, ...]:
        return tuple(
            variable_component_literal(component)
            for component in self.variable_components
        )

    @property
    def group_by_literal(self) -> str:
        return group_by_literal(self.group_by_component)

    @property
    def source_identity_stack_axes_literal(self) -> str:
        return all_component_tuple_literal(self.source_identity_stack_components)

    def source_identity_stack_argument_line(self) -> str | None:
        """Return generated FunctionStep source-identity argument when required."""
        if not self.source_identity_stack_components:
            return None
        return (
            "        source_identity_stack_axes="
            f"{self.source_identity_stack_axes_literal},"
        )

    def execution_components(self) -> ComponentSet:
        """Return every component that can split this module invocation."""
        return ComponentSet.collect(
            self.variable_components,
            (self.group_by_component,),
        )

    def without_source_stack_components(
        self,
        axis_plan: SourceProcessingAxisPlan,
    ) -> "ModuleProcessingComponents":
        collapsed_source_stack_components = ComponentSet(
            axis_plan.source_stack_components
        )
        return ModuleProcessingComponents(
            (
                ComponentSet(self.variable_components)
                .excluding(collapsed_source_stack_components)
                .as_tuple()
            ),
            self.group_by_component,
            ComponentSet.collect(
                self.source_identity_stack_components,
                collapsed_source_stack_components,
            ).as_tuple(),
        )

    def with_source_identity_stack_components(
        self,
        components: Iterable[AllComponents],
    ) -> "ModuleProcessingComponents":
        """Mark axes that remain variables but share one source identity."""
        return ModuleProcessingComponents(
            self.variable_components,
            self.group_by_component,
            ComponentSet.collect(
                self.source_identity_stack_components,
                components,
            ).as_tuple(),
        )


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

    category: str
    function_name: str
    runtime_lineage: RuntimeArtifactLineageScope
    bound_settings: GeneratedStepSettings
    source_schema: PipelineImageSchema

    def object_label_measurement_execution(self) -> ObjectLabelMeasurementExecution:
        return object_label_measurement_execution_from_callable(
            self.resolved_callable()
        )

    def processing_contract(self) -> ProcessingContract | None:
        return CallableContract.from_callable(self.resolved_callable()).processing_contract

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

    def category_default_components(self) -> tuple[AllComponents, ...]:
        return CategoryVariableComponentProvider.components_for_category(
            self.category,
            self.axis_plan(),
        )

    def source_stack_components(self) -> tuple[AllComponents, ...]:
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
        scalar_split = self.scalar_runtime_axis_components(variable_components)
        if scalar_split is not None:
            return self.with_runtime_source_identity(scalar_split)
        if self.module_requires_pairwise_object_domain_scope or (
            self.lineage.requires_pairwise_object_domain_scope
            and self.uses_measurement_only_runtime_inputs()
        ):
            return self.with_runtime_source_identity(
                ModuleProcessingComponents(
                    (),
                    self.axis_plan.scalar_source_group_component(
                        variable_components,
                    ),
                ),
            )
        if any(
            spec.kind.participates_in_object_domain_scope
            for spec in self.lineage.contract.outputs
        ):
            if variable_components:
                return self.with_runtime_source_identity(
                    ModuleProcessingComponents(variable_components, None)
                )
            return self.with_runtime_source_identity(
                ModuleProcessingComponents(
                        (),
                        self.axis_plan.scalar_source_group_component(
                            variable_components,
                        ),
                    ),
                )
        if variable_components:
            return self.with_runtime_source_identity(
                ModuleProcessingComponents(variable_components, None)
            )
        return self.with_runtime_source_identity(
            ModuleProcessingComponents(
                (),
                self.axis_plan.scalar_source_group_component(
                    variable_components,
                ),
            ),
        )

    def with_runtime_source_identity(
        self,
        components: ModuleProcessingComponents,
    ) -> ModuleProcessingComponents:
        """Preserve source identity for runtime artifacts split by source axes."""
        source_identity_components = (
            ComponentSet.collect(
                components.execution_components(),
                self.lineage.variable_components,
                self.axis_plan.source_stack_components,
            )
            .intersection(ComponentSet(self.axis_plan.source_identity_stack_components()))
            .as_tuple()
        )
        if not source_identity_components:
            return components
        return components.with_source_identity_stack_components(source_identity_components)

    def scalar_runtime_axis_components(
        self,
        variable_components: tuple[AllComponents, ...],
    ) -> ModuleProcessingComponents | None:
        group_by_component = self.axis_plan.scalar_source_group_component(
            variable_components
        )
        if group_by_component is None:
            return None
        return ModuleProcessingComponents(
            (
                ComponentSet(variable_components)
                .excluding(ComponentSet.collect((group_by_component,)))
                .as_tuple()
            ),
            group_by_component,
        )

    def requires_pairwise_object_domain_scope(self) -> bool:
        object_domain_inputs = tuple(
            spec
            for spec in self.lineage.contract.runtime_artifact_inputs
            if spec.kind.participates_in_pairwise_object_domain_input
        )
        object_domain_outputs = tuple(
            spec
            for spec in self.lineage.contract.outputs
            if spec.kind.participates_in_object_domain_scope
        )
        return len(object_domain_inputs) > 1 and bool(object_domain_outputs)

    def uses_measurement_only_runtime_inputs(self) -> bool:
        return bool(self.lineage.contract.runtime_artifact_inputs) and all(
            spec.kind is ArtifactKind.MEASUREMENTS
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
            return ModuleProcessingComponents(
                ComponentSet.collect(
                    self.axis_plan.image_set_components,
                    self.source_stack_components,
                ).as_tuple(),
                self.axis_plan.sample_group_component,
            )
        if self.source_bindings.requires_pipeline_start_resolution:
            source_stack_scope = self.source_stack_processing_scope(
                declared_components,
                declared_components,
            )
            if source_stack_scope is not None:
                return source_stack_scope
            return self.scalar_source_binding_components(
                declared_components,
                preserve_remaining_source_variables=True,
            )
        source_stack_scope = self.source_stack_processing_scope((), declared_components)
        if source_stack_scope is not None:
            return source_stack_scope
        return self.scalar_source_binding_components(
            declared_components,
            preserve_remaining_source_variables=False,
        )

    def scalar_source_binding_components(
        self,
        declared_components: tuple[AllComponents, ...],
        *,
        preserve_remaining_source_variables: bool,
    ) -> ModuleProcessingComponents:
        """Lower scalar source bindings without discarding image-set variables."""
        group_by_component = self.axis_plan.scalar_source_group_component(
            declared_components,
        )
        variable_components: tuple[AllComponents, ...] = ()
        if preserve_remaining_source_variables:
            variable_components = (
                ComponentSet(declared_components)
                .excluding(
                    ComponentSet.collect((group_by_component,)),
                    ComponentSet(self.axis_plan.source_stack_components),
                    ComponentSet(self.axis_plan.source_alias_components),
                )
                .as_tuple()
            )
        return ModuleProcessingComponents(
            variable_components,
            group_by_component,
        )

    def source_stack_processing_scope(
        self,
        declared_components: tuple[AllComponents, ...],
        scalar_group_components: tuple[AllComponents, ...],
    ) -> ModuleProcessingComponents | None:
        if not self.source_stack_components:
            return None
        source_stack_components = ComponentSet.collect(
            self.axis_plan.without_source_set_components(
                declared_components,
            ),
            self.source_stack_components,
        ).as_tuple()
        return ModuleProcessingComponents(
            source_stack_components,
            self.axis_plan.scalar_source_group_component(
                scalar_group_components,
            ),
            source_stack_components,
        )

    def requires_image_set_stack(self) -> bool:
        """Return whether one function call consumes multiple source-image aliases."""
        return (
            self.has_multi_image_binding_group()
            and self.source_bindings.requires_step_input_component_stack(
                self.axis_plan.image_set_components
            )
        ) or self.source_bindings.requires_pipeline_start_image_set_stack

    def has_multi_image_binding_group(self) -> bool:
        return any(
            sum(
                1
                for binding in group.bindings
                if binding.participates_in_execution_anchoring
            )
            > 1
            for group in self.source_bindings.groups
        )


class ModuleProcessingComponentStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for lowering module runtime-scope semantics."""

    __registry_key__ = "module_name"
    module_name: ClassVar[str]

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleProcessingComponentStrategy":
        canonical_name = canonical_module_name(module_name)
        strategy_type = cls.__registry__.get(canonical_name)
        if strategy_type is None:
            return DefaultModuleProcessingComponentStrategy()
        return strategy_type()

    @abstractmethod
    def components(
        self,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        """Return generated processing-component literals for this module."""

    def module_requires_pairwise_object_domain_scope(
        self,
        contract: ModuleArtifactContracts,
    ) -> bool:
        object_domain_inputs = tuple(
            spec
            for spec in contract.runtime_artifact_inputs
            if spec.kind.participates_in_pairwise_object_domain_input
        )
        object_domain_outputs = tuple(
            spec
            for spec in contract.outputs
            if spec.kind.participates_in_object_domain_scope
        )
        return len(object_domain_inputs) > 1 and bool(object_domain_outputs)


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
        strategy: ModuleProcessingComponentStrategy,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        """Return generated processing-component literals for the policy."""


class RuntimeArtifactModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Runtime artifacts determine scope before direct source bindings."""

    policy_name = "runtime_artifact"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        return bool(request.runtime_lineage.contract.runtime_artifact_inputs)

    def components(
        self,
        strategy: ModuleProcessingComponentStrategy,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        return request.runtime_artifact_scope(
            module_requires_pairwise_object_domain_scope=(
                strategy.module_requires_pairwise_object_domain_scope(
                    request.runtime_lineage.contract
                )
            ),
        ).components()


class SourceBindingModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Direct CellProfiler source bindings determine scope for source steps."""

    policy_name = "source_binding"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        return not request.runtime_lineage.contract.source_bindings.is_empty

    def components(
        self,
        strategy: ModuleProcessingComponentStrategy,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        del strategy
        return SourceBindingProcessingScope(
            request.runtime_lineage.contract.source_bindings,
            request.source_schema,
            request.axis_plan(request.runtime_lineage.contract.source_bindings),
            request.source_stack_components(),
        ).components()


class InputlessArtifactModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Artifact-only aggregate modules execute once per site axis."""

    policy_name = "inputless_artifact"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        return _is_inputless_artifact_only_contract(request.runtime_lineage.contract)

    def components(
        self,
        strategy: ModuleProcessingComponentStrategy,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        del strategy
        return ModuleProcessingComponents(
            request.category_default_components(),
            None,
        )


class CategoryDefaultModuleProcessingScopePolicy(ModuleProcessingScopePolicy):
    """Absorbed module category defaults fill the remaining pure image cases."""

    policy_name = "category_default"

    def matches(self, request: ModuleProcessingComponentRequest) -> bool:
        del request
        return True

    def components(
        self,
        strategy: ModuleProcessingComponentStrategy,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        del strategy
        return ModuleProcessingComponents(
            tuple(request.category_default_components()),
            None,
        )


class DefaultModuleProcessingComponentStrategy(ModuleProcessingComponentStrategy):
    """Default conversion from source bindings/contracts to OpenHCS runtime scope."""

    module_name = "__default__"

    def components(
        self,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        components = ModuleProcessingScopePolicy.for_request(request).components(
            self,
            request,
        )
        if request.requires_source_stack_collapse():
            return components.without_source_stack_components(request.axis_plan())
        return components


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
) -> tuple[str, ...]:
    """Return generated FunctionStep arguments owned by source semantics."""
    lines: list[str] = []
    source_identity_line = processing_components.source_identity_stack_argument_line()
    if source_identity_line is not None:
        lines.append(source_identity_line)
    if not artifact_contract.source_bindings.is_empty:
        lines.append(
            "        source_bindings="
            f"{source_bindings_literal(artifact_contract.source_bindings)},"
        )
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactSourceLineage:
    """Variable-component scope inherited from source-bound artifact ancestry."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContracts]
    source_schema: PipelineImageSchema = PipelineImageSchema.empty()

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
            self._extend_unique(
                components,
                ComponentSet.collect(
                    SourceProcessingComponentSemantics(
                        self.source_schema,
                        contract.source_bindings,
                    ).variable_components(),
                    self.source_schema.source_stack_components,
                ).as_tuple(),
            )

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

        if ModuleProcessingComponentStrategy.for_module(
            contract.module_name
        ).module_requires_pairwise_object_domain_scope(contract):
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


class TrackObjectsProcessingComponentStrategy(DefaultModuleProcessingComponentStrategy):
    """Track labels across source-frame timepoint order without stacking channels."""

    module_name = "TrackObjects"

    def components(
        self,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        del request
        return ModuleProcessingComponents(
            (AllComponents.TIMEPOINT,),
            None,
        )


class StraightenWormsProcessingComponentStrategy(
    DefaultModuleProcessingComponentStrategy
):
    """Preserve per-source-image identity for straightened image artifacts."""

    module_name = "StraightenWorms"

    def components(
        self,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        base_components = super().components(request)
        if (
            SourceBindingProcessingScope(
                request.runtime_lineage.contract.source_bindings,
                request.source_schema,
                request.axis_plan(request.runtime_lineage.contract.source_bindings),
            ).has_multi_image_binding_group()
        ):
            return ModuleProcessingComponents(
                base_components.variable_components,
                base_components.group_by_component,
                (AllComponents.CHANNEL,),
            )
        return base_components


class GrayToColorProcessingComponentStrategy(DefaultModuleProcessingComponentStrategy):
    """Composite color outputs intentionally stack channel source identities."""

    module_name = "GrayToColor"

    def components(
        self,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        base_components = super().components(request)
        return ModuleProcessingComponents(
            base_components.variable_components,
            base_components.group_by_component,
            ComponentSet.collect(
                base_components.source_identity_stack_components,
                (AllComponents.CHANNEL,),
            ).as_tuple(),
        )


class MeasureImageAreaOccupiedProcessingComponentStrategy(
    DefaultModuleProcessingComponentStrategy
):
    """Scope object-area aggregate rows per image set."""

    module_name = "MeasureImageAreaOccupiedBinary"

    def module_requires_pairwise_object_domain_scope(
        self,
        contract: ModuleArtifactContracts,
    ) -> bool:
        object_domain_inputs = tuple(
            spec
            for spec in contract.runtime_artifact_inputs
            if spec.kind.participates_in_pairwise_object_domain_input
        )
        return len(object_domain_inputs) > 1


class CorrectIlluminationCalculateProcessingComponentStrategy(
    DefaultModuleProcessingComponentStrategy
):
    """Lower CellProfiler all-image illumination scope to a site stack per channel."""

    module_name = "CorrectIlluminationCalculate"

    def components(
        self,
        request: ModuleProcessingComponentRequest,
    ) -> ModuleProcessingComponents:
        raw_scope = request.bound_settings.value(
            "calculation_scope",
            IlluminationCalculationScope.EACH,
        )
        scope = coerce_cellprofiler_enum(IlluminationCalculationScope, raw_scope)
        if scope.requires_channel_grouping:
            axis_plan = request.axis_plan()
            sample_group_components = SourceProcessingAxisRolePolicy.for_role(
                SourceProcessingAxisRole.SAMPLE_GROUP
            ).components(axis_plan)
            return ModuleProcessingComponents(
                sample_group_components,
                axis_plan.optional_single_image_set_component(
                    "CorrectIlluminationCalculate all-image scope cannot infer one "
                    "group-by component from multiple image-set axes: "
                    f"{tuple(component.value for component in axis_plan.image_set_components)!r}."
                ),
                sample_group_components,
            )
        return super().components(request)


def _is_inputless_artifact_only_contract(contract: ModuleArtifactContracts) -> bool:
    """Return whether a step should execute once per source sample."""
    return (
        not contract.inputs
        and not contract.runtime_artifact_inputs
        and bool(contract.outputs)
        and all(spec.kind in _INPUTLESS_ARTIFACT_ONLY_KINDS for spec in contract.outputs)
    )


def _has_materialized_output(contract: ModuleArtifactContracts) -> bool:
    """Return whether any output is externally observable by artifact policy."""
    return any(
        spec.materialization is not None
        or spec.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
        for spec in contract.outputs
    )


def _save_images_required_artifacts(
    skipped_modules: Iterable[ModuleBlock],
) -> frozenset[ArtifactSpecKey]:
    """Return image artifacts required by skipped CellProfiler SaveImages modules."""
    return frozenset(
        ArtifactSpecKey(ArtifactKind.IMAGE, image_name)
        for module in skipped_modules
        if module.name == "SaveImages"
        for value in setting_values(module, SAVE_IMAGES_SOURCE_IMAGE_SETTING)
        for image_name in split_symbol_names(value)
    )


@dataclass
class GeneratedPipeline:
    """Complete generated OpenHCS pipeline."""
    
    name: str
    code: str
    source_cppipe: str
    converted_modules: List[str]
    failed_modules: List[str]
    artifact_contracts: tuple[ModuleArtifactContracts, ...] = ()
    runtime_module_contracts: tuple[tuple[int, ModuleArtifactContract], ...] = ()
    source_schema: PipelineImageSchema = PipelineImageSchema.empty()
    setting_coverage: tuple[ModuleSettingCoverageRecord, ...] = ()

    @property
    def runtime_module_contracts_by_module_num(
        self,
    ) -> dict[int, ModuleArtifactContract]:
        """Runtime artifact contracts keyed by original CellProfiler module number."""
        return dict(self.runtime_module_contracts)
    
    def save(
        self,
        output_path: Path,
        *,
        filemanager: FileManagerLike | None = None,
        backend: Backend = Backend.DISK,
    ) -> None:
        """Save pipeline to file."""
        if not isinstance(backend, Backend):
            raise TypeError(
                "GeneratedPipeline.save backend must be Backend, "
                f"got {type(backend).__name__}."
            )
        if filemanager is None:
            output_path.write_text(self.code)
        else:
            filemanager.ensure_directory(str(output_path.parent), backend.value)
            filemanager.save(self.code, str(output_path), backend.value)
        logger.info(f"Saved pipeline to {output_path}")


@dataclass(frozen=True, slots=True)
class SkippedModuleSelection:
    """Public optional skipped-module argument normalized for generation."""

    modules: tuple[ModuleBlock, ...] = ()

    @classmethod
    def from_optional(
        cls,
        modules: Optional[List[ModuleBlock]],
    ) -> "SkippedModuleSelection":
        if modules is None:
            return cls(())
        return cls(tuple(modules))


@dataclass(frozen=True)
class GeneratedPipelineRequest:
    """Nominal request for one registry-backed CellProfiler pipeline generation."""

    pipeline_name: str
    source_cppipe: Path
    skipped_modules: tuple[ModuleBlock, ...] = ()
    prune_dead_unmaterialized_artifact_steps: bool = False
    materialize_skipped_save_images: bool = True
    materialize_terminal_images: bool = True

    @classmethod
    def from_public_args(
        cls,
        *,
        pipeline_name: str,
        source_cppipe: Path,
        skipped_modules: Optional[List[ModuleBlock]],
        prune_dead_unmaterialized_artifact_steps: bool,
        materialize_skipped_save_images: bool,
        materialize_terminal_images: bool,
    ) -> "GeneratedPipelineRequest":
        """Build a generation request from the stable public API arguments."""
        return cls(
            pipeline_name=pipeline_name,
            source_cppipe=source_cppipe,
            skipped_modules=SkippedModuleSelection.from_optional(
                skipped_modules
            ).modules,
            prune_dead_unmaterialized_artifact_steps=(
                prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=materialize_skipped_save_images,
            materialize_terminal_images=materialize_terminal_images,
        )


@dataclass(frozen=True)
class PipelineGeneratorRegistryStage:
    """Absorbed-library registry loading and module metadata lookup."""

    generator: "PipelineGenerator"

    def load_registry(self) -> dict[str, AbsorbedModuleMetadata]:
        """Load full module metadata from the OpenHCS-owned absorbed catalog."""
        if self.generator._explicit_library_root:
            return self.load_legacy_registry(self.generator.library_root)

        try:
            registry = {
                module_name: AbsorbedModuleMetadata.from_registry_record(info)
                for module_name, info in validated_contracts().items()
            }
            logger.info(f"Loaded {len(registry)} absorbed functions from registry")
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def load_legacy_registry(
        self,
        library_root: Path,
    ) -> dict[str, AbsorbedModuleMetadata]:
        """Load metadata from an explicit maintenance-time absorbed-library root."""
        contracts_file = library_root / "contracts.json"
        if not contracts_file.exists():
            raise FileNotFoundError(
                f"No absorbed library found at {contracts_file}. "
                "Run 'python -m benchmark.converter.absorb' first."
            )

        try:
            data = json.loads(contracts_file.read_text())
            registry: dict[str, AbsorbedModuleMetadata] = {}
            for module_name, info in data.items():
                record = AbsorbedRegistryRecordView(info)
                if not record.optional_bool("validated", False):
                    continue
                registry[module_name] = AbsorbedModuleMetadata.from_registry_record(
                    info
                )
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def has_module(self, module_name: str) -> bool:
        """Check if module exists in absorbed library."""
        return canonical_module_name(module_name) in self.generator._registry

    def module_metadata(self, module_name: str) -> AbsorbedModuleMetadata:
        """Return absorbed metadata for a module after canonical name resolution."""
        return self.generator._registry[canonical_module_name(module_name)]


@dataclass(slots=True)
class OutputSymbolsBySetting:
    """Output artifact symbols grouped by normalized CellProfiler setting name."""

    values: dict[str, set[ArtifactSpecKey]]

    @classmethod
    def empty(cls) -> "OutputSymbolsBySetting":
        return cls({})

    def add(self, setting_name: str, artifact: ArtifactSpecKey) -> None:
        if setting_name not in self.values:
            self.values[setting_name] = set()
        self.values[setting_name].add(artifact)

    def dead_setting_names(
        self,
        retained_outputs: frozenset[ArtifactSpecKey],
    ) -> frozenset[str]:
        return frozenset(
            setting_name
            for setting_name, output_symbols in self.values.items()
            if output_symbols and not output_symbols & retained_outputs
        )


@dataclass(frozen=True)
class PipelineGeneratorArtifactPruner:
    """Dead-artifact pruning and setting-pruning authority."""

    generator: "PipelineGenerator"

    def prune_dead_unmaterialized_artifact_steps(
        self,
        modules: list[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_required_artifacts: set[ArtifactSpecKey] | None = None,
    ) -> list[ModuleBlock]:
        """Remove artifact-producing steps whose outputs are neither consumed nor materialized."""
        live_artifacts = {
            ArtifactSpecKey.from_spec(output)
            for contract in artifact_contracts.values()
            for output in contract.outputs
            if (
                output.materialization is not None
                or output.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
            )
        } | {
            ArtifactSpecKey.from_spec(output)
            for contract in artifact_contracts.values()
            for output in externally_required_artifact_outputs(
                contract.declared_outputs
            )
        }
        if externally_required_artifacts:
            live_artifacts.update(externally_required_artifacts)
        live_module_nums: set[int] = set()

        for module in reversed(modules):
            contract = artifact_contracts[module.module_num]
            output_keys = {
                ArtifactSpecKey.from_spec(output) for output in contract.outputs
            }
            keep = (
                not contract.outputs
                or _has_materialized_output(contract)
                or bool(output_keys & live_artifacts)
            )
            if not keep:
                continue
            live_module_nums.add(module.module_num)
            retained_outputs = tuple(
                output
                for output in contract.outputs
                if (
                    output.materialization is not None
                    or output.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
                    or ArtifactSpecKey.from_spec(output) in live_artifacts
                )
            )
            artifact_contracts[module.module_num] = replace(
                contract,
                output_symbols=tuple(
                    symbol
                    for symbol in contract.output_symbols
                    if symbol.artifact_spec() in retained_outputs
                ),
            )
            live_artifacts.update(
                ArtifactSpecKey.from_spec(input_spec)
                for input_spec in contract.runtime_artifact_inputs
            )

        pruned = [module for module in modules if module.module_num in live_module_nums]
        skipped = [module for module in modules if module.module_num not in live_module_nums]
        if skipped:
            logger.info(
                "Pruned %d dead unmaterialized artifact step(s): %s",
                len(skipped),
                [module.name for module in skipped],
            )
        return pruned

    def prune_dead_output_setting_kwargs(
        self,
        *,
        module: ModuleBlock,
        translated_kwargs: GeneratedStepSettings,
        param_mapping: Mapping[str, GeneratedParameterName],
        artifact_contract: ModuleArtifactContracts,
    ) -> GeneratedStepSettings:
        """Drop function kwargs for output-name settings pruned from artifacts."""
        dead_settings = self.dead_output_setting_names(
            module=module,
            artifact_contract=artifact_contract,
        )
        return translated_kwargs.without_dead_output_settings(
            dead_settings=dead_settings,
            param_mapping=param_mapping,
        )

    @staticmethod
    def dead_output_setting_names(
        *,
        module: ModuleBlock,
        artifact_contract: ModuleArtifactContracts,
    ) -> frozenset[str]:
        """Return CellProfiler output-name settings whose artifacts were pruned."""
        retained_outputs = frozenset(
            ArtifactSpecKey(symbol.kind.artifact_kind, symbol.name)
            for symbol in artifact_contract.output_symbols
        )
        output_symbols_by_setting = OutputSymbolsBySetting.empty()
        for symbol in artifact_setting_symbols(module):
            if symbol.role.is_input:
                continue
            normalized_setting = normalize_cellprofiler_setting_name(
                symbol.setting_name
            )
            output_symbols_by_setting.add(
                normalized_setting,
                ArtifactSpecKey(symbol.role.artifact_kind, symbol.name)
            )
        return output_symbols_by_setting.dead_setting_names(retained_outputs)

    @staticmethod
    def terminal_image_artifacts(
        modules: list[ModuleBlock],
        artifact_contracts: Mapping[int, ModuleArtifactContracts],
        *,
        external_consumers: Iterable[ArtifactSpec] = (),
    ) -> set[ArtifactSpecKey]:
        """Return final image outputs that remain observable pipeline products."""

        consumed: set[ArtifactSpecKey] = {
            ArtifactSpecKey.from_spec(spec) for spec in external_consumers
        }
        terminal: set[ArtifactSpecKey] = set()
        for module in reversed(modules):
            contract = artifact_contracts[module.module_num]
            module_outputs = {
                ArtifactSpecKey.from_spec(spec)
                for spec in contract.outputs
                if spec.kind is ArtifactKind.IMAGE
            }
            terminal.update(output for output in module_outputs if output not in consumed)
            consumed.update(ArtifactSpecKey.from_spec(spec) for spec in contract.inputs)
            consumed.update(
                ArtifactSpecKey.from_spec(spec)
                for spec in contract.runtime_artifact_inputs
            )
        return terminal


@dataclass(frozen=True)
class PipelineGeneratorRuntimeContractProjector:
    """Projection from symbol-table contracts to runtime artifact contracts."""

    generator: "PipelineGenerator"

    def by_module_num(
        self,
        modules: List[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs = frozenset(),
        artifact_name_materialized_outputs: ArtifactNameMaterializedOutputs = (
            frozenset()
        ),
    ) -> dict[int, ModuleArtifactContract]:
        """Build product-runtime contracts without serializing them into generated code."""
        return {
            module.module_num: self.runtime_module_contract(
                artifact_contracts[module.module_num],
                externally_materialized_outputs=externally_materialized_outputs,
                artifact_name_materialized_outputs=artifact_name_materialized_outputs,
            )
            for module in modules
            if (
                artifact_contracts[module.module_num].inputs
                or artifact_contracts[module.module_num].outputs
            )
        }

    def runtime_module_contract(
        self,
        contract: ModuleArtifactContracts,
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs,
        artifact_name_materialized_outputs: ArtifactNameMaterializedOutputs,
    ) -> ModuleArtifactContract:
        """Project symbol-table contracts into runtime module contracts."""
        outputs = tuple(
            self.runtime_output_spec(
                spec,
                externally_materialized_outputs=externally_materialized_outputs,
                artifact_name_materialized_outputs=artifact_name_materialized_outputs,
            )
            for spec in contract.outputs
        )
        declared_outputs = tuple(
            self.runtime_output_spec(
                spec,
                externally_materialized_outputs=externally_materialized_outputs,
                artifact_name_materialized_outputs=artifact_name_materialized_outputs,
            )
            for spec in contract.declared_outputs
        )
        return ModuleArtifactContract(
            module_name=contract.module_name,
            inputs=contract.inputs,
            runtime_artifact_inputs=contract.runtime_artifact_inputs,
            outputs=outputs,
            declared_outputs=declared_outputs,
        )

    @staticmethod
    def runtime_output_spec(
        spec: ArtifactSpec,
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs,
        artifact_name_materialized_outputs: ArtifactNameMaterializedOutputs,
    ) -> ArtifactSpec:
        """Apply runtime-only materialization required by skipped SaveImages modules."""
        if ArtifactSpecKey.from_spec(spec) not in externally_materialized_outputs:
            if (
                spec.materialization is None
                and spec.kind is not ArtifactKind.SPECIAL
                and spec.kind not in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
            ):
                return replace(spec, materialization=NO_ARTIFACT_MATERIALIZATION)
            return spec
        filename_identity = (
            MaterializedFilenameIdentity.ARTIFACT_NAME
            if ArtifactSpecKey.from_spec(spec) in artifact_name_materialized_outputs
            else MaterializedFilenameIdentity.SOURCE_IDENTITY
        )
        return replace(
            spec,
            materialization=tiff_stack(
                normalize_uint8=True,
                filename_identity=filename_identity,
            ),
        )


class ArtifactContractCommentSection(ABC, metaclass=AutoRegisterMeta):
    """Auto-registered generated-comment section for artifact contracts."""

    __registry_key__ = "section_name"
    __skip_if_no_key__ = True
    section_name: ClassVar[str]
    order: ClassVar[int]

    @classmethod
    def lines_for(cls, contract: ModuleArtifactContracts) -> list[str]:
        lines: list[str] = []
        section_types = sorted(
            cls.__registry__.values(),
            key=lambda section_type: section_type.order,
        )
        for section_type in section_types:
            section = section_type()
            if section.matches(contract):
                lines.append(section.line(contract))
        return lines

    @staticmethod
    def format_artifact_specs(specs: tuple[ArtifactSpec, ...]) -> str:
        """Format artifact specs for deterministic generated-code comments."""
        return ", ".join(f"{spec.kind.value}:{spec.name}" for spec in specs)

    @abstractmethod
    def matches(self, contract: ModuleArtifactContracts) -> bool:
        """Return whether this comment section has content."""

    @abstractmethod
    def line(self, contract: ModuleArtifactContracts) -> str:
        """Return the generated comment line for this section."""


class InputArtifactCommentSection(ArtifactContractCommentSection):
    """Comment section for declared CellProfiler artifact inputs."""

    section_name = "inputs"
    order = 10

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.inputs)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return (
            "        # CellProfiler artifact inputs: "
            + self.format_artifact_specs(contract.inputs)
        )


class SourceBindingCommentSection(ArtifactContractCommentSection):
    """Comment section for external source-image bindings."""

    section_name = "source_bindings"
    order = 20

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.external_source_symbols)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return "        # Source bindings: " + ", ".join(
            symbol.name for symbol in contract.external_source_symbols
        )


class RuntimeArtifactCommentSection(ArtifactContractCommentSection):
    """Comment section for runtime artifact dependencies."""

    section_name = "runtime_artifact_inputs"
    order = 30

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.runtime_artifact_inputs)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return (
            "        # Runtime artifact inputs: "
            + self.format_artifact_specs(contract.runtime_artifact_inputs)
        )


class OutputArtifactCommentSection(ArtifactContractCommentSection):
    """Comment section for declared CellProfiler artifact outputs."""

    section_name = "outputs"
    order = 40

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.outputs)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return (
            "        # CellProfiler artifact outputs: "
            + self.format_artifact_specs(contract.outputs)
        )


@dataclass(frozen=True, slots=True)
class StepInputSourceLiteral:
    """Generated LazyProcessingConfig input_source fragment for one step."""

    value: str | None = None

    @classmethod
    def from_contract(
        cls,
        contract: ModuleArtifactContracts,
    ) -> "StepInputSourceLiteral":
        if contract.external_source_symbols:
            return cls("InputSource.PIPELINE_START")
        return cls()

    def append_to(self, lines: list[str]) -> None:
        if self.value is None:
            return
        lines.append(f"            input_source={self.value},")


@dataclass(frozen=True)
class PipelineGeneratorCodeEmitter:
    """Generated-code emission for imports, FunctionStep declarations, and comments."""

    generator: "PipelineGenerator"

    def generate_steps_from_registry(
        self,
        modules: List[ModuleBlock],
        function_names_by_module: Mapping[int, str],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        source_schema: PipelineImageSchema,
    ) -> tuple[str, tuple[ModuleSettingCoverageRecord, ...]]:
        """Generate pipeline_steps using registry functions with bound settings."""
        lines = [
            "# Pipeline Steps",
            "# Settings from .cppipe are bound as default parameters",
            "# variable_components derived from LLM-inferred category",
            "pipeline_steps = [",
        ]
        setting_coverage: list[ModuleSettingCoverageRecord] = []
        source_lineage = RuntimeArtifactSourceLineage(
            artifact_contracts,
            source_schema,
        )

        for module in modules:
            meta = self.generator.registry.module_metadata(module.name)
            category = meta.category
            step_name = module.name
            artifact_contract = artifact_contracts[module.module_num]
            func_name = function_names_by_module[module.module_num]

            input_source_literal = StepInputSourceLiteral.from_contract(
                artifact_contract
            )

            param_mapping: dict[str, GeneratedParameterName] = {}
            dead_output_settings = self.generator.pruner.dead_output_setting_names(
                module=module,
                artifact_contract=artifact_contract,
            )
            bound_settings = _ModuleSettingsBindingStrategy.for_module(
                module.name
            ).bind(
                module,
                binder=self.generator.settings_binder,
                param_mapping=param_mapping,
                ignored_unmapped_settings=dead_output_settings,
            )
            setting_coverage.extend(bound_settings.setting_coverage)
            translated_kwargs = GeneratedStepSettings.from_mapping(
                bound_settings.kwargs
            )
            translated_kwargs = self.generator.pruner.prune_dead_output_setting_kwargs(
                module=module,
                translated_kwargs=translated_kwargs,
                param_mapping=param_mapping,
                artifact_contract=artifact_contract,
            )
            invocation_options_literal = self.invocation_options_literal(
                bound_settings.invocation_options
            )
            processing_components = ModuleProcessingComponentStrategy.for_module(
                module.name
            ).components(
                ModuleProcessingComponentRequest(
                    category=category,
                    function_name=func_name,
                    runtime_lineage=RuntimeArtifactLineageScope(
                        artifact_contract,
                        source_lineage.variable_components_for(artifact_contract),
                        source_lineage.requires_pairwise_object_domain_scope_for(
                            artifact_contract
                        ),
                    ),
                    bound_settings=translated_kwargs,
                    source_schema=source_schema,
                )
            )

            lines.append("    FunctionStep(")
            lines.extend(self.artifact_contract_comments(artifact_contract))
            if translated_kwargs:
                kwargs_lines = ["{"]
                for k, v in translated_kwargs.items():
                    kwargs_lines.append(
                        f"            {repr(k)}: {python_literal(v)},"
                    )
                kwargs_lines.append("        }")
                kwargs_str = "\n".join(kwargs_lines)

                if invocation_options_literal is None:
                    lines.append(f"        func=({func_name}, {kwargs_str}),")
                else:
                    lines.append(
                        f"        func=({func_name}, {kwargs_str}, "
                        f"{invocation_options_literal}),"
                    )
            else:
                if invocation_options_literal is None:
                    lines.append(f"        func={func_name},")
                else:
                    lines.append(
                        f"        func=({func_name}, {{}}, "
                        f"{invocation_options_literal}),"
                    )

            lines.append(f'        name="{step_name}",')
            lines.extend(
                generated_function_step_semantic_argument_lines(
                    processing_components=processing_components,
                    artifact_contract=artifact_contract,
                )
            )
            lines.append("        processing_config=LazyProcessingConfig(")
            lines.append(
                "            variable_components=["
                + ", ".join(processing_components.variable_component_literals)
                + "],"
            )
            lines.append(
                f"            group_by={processing_components.group_by_literal},"
            )
            input_source_literal.append_to(lines)
            lines.append("        ),")

            lines.append("    ),")

        lines.append("]")
        return "\n".join(lines), tuple(setting_coverage)

    @staticmethod
    def runtime_contract_binding_block(
        runtime_contracts_by_module: Mapping[int, ModuleArtifactContract],
    ) -> str:
        """Return source that rebinds generated CP steps to runtime contracts."""
        if not runtime_contracts_by_module:
            return ""
        literal_imports: set[tuple[str, str]] = set()
        contract_lines: list[str] = []
        for module_num, contract in sorted(runtime_contracts_by_module.items()):
            contract_lines.append(
                "    "
                f"{module_num}: "
                f"{module_contract_literal(contract, import_collector=literal_imports)},"
            )
        lines = [
            "",
            "",
            "# CellProfiler runtime artifact contracts",
        ]
        lines.extend(
            f"from {module_name} import {symbol_name}"
            for module_name, symbol_name in sorted(literal_imports)
        )
        if literal_imports:
            lines.append("")
        lines.append("_CELLPROFILER_RUNTIME_CONTRACTS_BY_MODULE_NUM = {")
        lines.extend(contract_lines)
        lines.extend(
            (
                "}",
                "",
                "from types import ModuleType as _OpenHCSGeneratedModuleType",
                "from openhcs.interop.cellprofiler.runtime.generated_pipeline import (",
                "    bind_generated_pipeline_runtime as _openhcs_bind_generated_pipeline_runtime,",
                ")",
                "_openhcs_generated_module = _OpenHCSGeneratedModuleType(",
                "    globals().get('__name__', 'openhcs_generated_pipeline')",
                ")",
                "_openhcs_generated_module.pipeline_steps = pipeline_steps",
                "_openhcs_bind_generated_pipeline_runtime(",
                "    _openhcs_generated_module,",
                "    _CELLPROFILER_RUNTIME_CONTRACTS_BY_MODULE_NUM,",
                ")",
                "del _openhcs_generated_module",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def invocation_options_literal(
        options: RuntimeInvocationOptions | None,
    ) -> str | None:
        """Return generated-code literal for typed invocation options."""
        if options is None:
            return None
        if isinstance(options, CellProfilerInvocationOptions):
            scope = options.grid_cycle_scope
            if not isinstance(scope, CellProfilerGridCycleScope):
                raise TypeError(
                    "CellProfilerInvocationOptions.grid_cycle_scope must be "
                    "CellProfilerGridCycleScope."
                )
            return (
                "CellProfilerInvocationOptions("
                f"grid_cycle_scope=CellProfilerGridCycleScope.{scope.name})"
            )
        raise TypeError(
            "Unsupported RuntimeInvocationOptions for generated pipeline: "
            f"{type(options).__name__}."
        )

    @staticmethod
    def backend_function_import_block(function_names: Iterable[str]) -> str:
        """Return imports for the absorbed backend functions used by the pipeline."""
        unique_function_names = tuple(dict.fromkeys(sorted(function_names)))
        if not unique_function_names:
            return ""
        lines = [
            "from openhcs.processing.backends.cellprofiler import "
            "get_cellprofiler_function as _get_cellprofiler_function",
        ]
        lines.extend(
            f"{function_name} = _get_cellprofiler_function({function_name!r})"
            for function_name in unique_function_names
        )
        lines.append("")
        return "\n".join(lines)

    def artifact_contract_comments(
        self,
        contract: ModuleArtifactContracts,
    ) -> list[str]:
        """Return generated comments summarizing artifact contract semantics."""
        return ArtifactContractCommentSection.lines_for(contract)


@dataclass(frozen=True)
class PipelineGeneratorBuildStage:
    """Top-level CellProfiler module partitioning and generated-pipeline assembly."""

    generator: "PipelineGenerator"

    def generate(
        self,
        request: GeneratedPipelineRequest,
        modules: List[ModuleBlock],
    ) -> GeneratedPipeline:
        """Generate pipeline using absorbed library (instant, no LLM)."""
        skipped_modules = list(request.skipped_modules)
        registry_modules = []
        missing_modules = []

        for module in modules:
            if self.generator.registry.has_module(module.name):
                registry_modules.append(module)
            else:
                missing_modules.append(module)
                logger.warning(f"Module {module.name} not in absorbed library")

        imports = self.generator.IMPORTS_BASE.format(
            source_file=request.source_cppipe.name
        )

        if skipped_modules:
            skip_note = "\n# Skipped infrastructure modules (handled by OpenHCS):\n"
            for module in skipped_modules:
                skip_note += (
                    f"#   - {cellprofiler_infrastructure_import_note(module.name)}\n"
                )
            imports += skip_note + "\n"

        if missing_modules:
            raise ValueError(
                f"Missing {len(missing_modules)} modules from absorbed library: "
                f"{[m.name for m in missing_modules]}. "
                "Re-run absorption with --force to regenerate."
            )

        ordered_modules = [*skipped_modules, *registry_modules]
        symbol_table = CellProfilerSymbolTable.compile(ordered_modules)
        contracts_by_module = {
            module.module_num: symbol_table.contract_for(module)
            for module in registry_modules
        }
        infrastructure_contracts = tuple(
            symbol_table.contract_for(module)
            for module in skipped_modules
        )
        save_images_required_artifacts = (
            _save_images_required_artifacts(skipped_modules)
            if request.materialize_skipped_save_images
            else frozenset()
        )
        infrastructure_input_artifacts = {
            ArtifactSpecKey.from_spec(input_spec)
            for contract in infrastructure_contracts
            for input_spec in (
                *contract.inputs,
                *contract.runtime_artifact_inputs,
            )
        }
        terminal_image_artifacts = (
            self.generator.pruner.terminal_image_artifacts(
                registry_modules,
                contracts_by_module,
                external_consumers=(
                    input_spec
                    for contract in infrastructure_contracts
                    for input_spec in (
                        *contract.inputs,
                        *contract.runtime_artifact_inputs,
                    )
                ),
            )
            if request.materialize_terminal_images
            else frozenset()
        )
        externally_materialized_outputs = (
            save_images_required_artifacts | terminal_image_artifacts
        )
        artifact_name_materialized_outputs = save_images_required_artifacts
        executable_modules = (
            self.generator.pruner.prune_dead_unmaterialized_artifact_steps(
                registry_modules,
                contracts_by_module,
                externally_required_artifacts=(
                    infrastructure_input_artifacts
                    | externally_materialized_outputs
                ),
            )
            if request.prune_dead_unmaterialized_artifact_steps
            else registry_modules
        )
        runtime_module_contracts_by_module = self.generator.runtime_contracts.by_module_num(
            executable_modules,
            contracts_by_module,
            externally_materialized_outputs=externally_materialized_outputs,
            artifact_name_materialized_outputs=artifact_name_materialized_outputs,
        )

        function_names_by_module: dict[int, str] = {}
        for module in executable_modules:
            meta = self.generator.registry.module_metadata(module.name)
            resolved_function = _ModuleFunctionResolutionStrategy.for_module(
                module.name
            ).resolve(
                module,
                default_function_name=meta.function_name,
            )
            function_names_by_module[module.module_num] = (
                resolved_function.function_name
            )

        if executable_modules:
            imports += "# Absorbed CellProfiler functions\n"
            imports += self.generator.emitter.backend_function_import_block(
                function_names_by_module.values()
            )
            imports += (
                "from openhcs.interop.cellprofiler.runtime import (\n"
                "    CellProfilerGridCycleScope,\n"
                "    CellProfilerInvocationOptions,\n"
                ")\n"
                "\n"
            )

        steps, setting_coverage = self.generator.emitter.generate_steps_from_registry(
            executable_modules,
            function_names_by_module,
            contracts_by_module,
            symbol_table.source_schema,
        )
        code = (
            imports
            + steps
            + self.generator.emitter.runtime_contract_binding_block(
                runtime_module_contracts_by_module
            )
        )

        return GeneratedPipeline(
            name=request.pipeline_name,
            code=code,
            source_cppipe=str(request.source_cppipe),
            converted_modules=[m.name for m in executable_modules],
            failed_modules=[m.name for m in missing_modules],
            artifact_contracts=tuple(
                contracts_by_module[module.module_num]
                for module in executable_modules
            ),
            runtime_module_contracts=tuple(
                (
                    module.module_num,
                    runtime_module_contracts_by_module[module.module_num],
                )
                for module in executable_modules
                if module.module_num in runtime_module_contracts_by_module
            ),
            source_schema=symbol_table.source_schema,
            setting_coverage=setting_coverage,
        )


class PipelineGenerator:
    """
    Generate complete OpenHCS pipeline from converted functions.

    TWO MODES:
    1. Registry-based: Uses pre-absorbed cellprofiler_library (instant, no LLM)
    2. LLM-based: Inline function definitions (fallback for unabsorbed modules)

    Creates a runnable pipeline file with:
    1. Standard imports (+ registry imports if using absorbed library)
    2. Converted function definitions (only for non-registry functions)
    3. FunctionStep wrappers for each function
    4. pipeline_steps list
    """

    # Standard imports for generated pipelines
    IMPORTS_BASE = '''"""
OpenHCS Pipeline - Converted from CellProfiler
Source: {source_file}

Auto-generated by CellProfiler to OpenHCS converter.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# OpenHCS imports
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifacts import ArtifactKind, ArtifactSidecarRole, ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.source_bindings import (
    ComponentSelector,
    EMPTY_SOURCE_BINDINGS,
    GroupedSourceBindings,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.config import LazyProcessingConfig
from openhcs.constants.constants import VariableComponents, GroupBy
from openhcs.constants.constants import AllComponents
from openhcs.constants.input_source import InputSource
from openhcs.processing.materialization import MaterializedFilenameIdentity, tiff_stack

'''

    def __init__(self, library_root: Optional[Path] = None):
        """
        Initialize generator.

        Args:
            library_root: Path to absorbed cellprofiler_library
        """
        self._explicit_library_root = library_root is not None
        self.library_root = library_root or Path(__file__).parent.parent / "cellprofiler_library"
        self.settings_binder = SettingsBinder()
        self.registry = PipelineGeneratorRegistryStage(self)
        self.pruner = PipelineGeneratorArtifactPruner(self)
        self.runtime_contracts = PipelineGeneratorRuntimeContractProjector(self)
        self.emitter = PipelineGeneratorCodeEmitter(self)
        self.builder = PipelineGeneratorBuildStage(self)
        self._registry = self.registry.load_registry()

    def has_module(self, module_name: str) -> bool:
        """Check if module exists in absorbed library."""
        return self.registry.has_module(module_name)

    def generate_from_registry(
        self,
        pipeline_name: str,
        source_cppipe: Path,
        modules: List[ModuleBlock],
        skipped_modules: Optional[List[ModuleBlock]] = None,
        prune_dead_unmaterialized_artifact_steps: bool = False,
        materialize_skipped_save_images: bool = True,
        materialize_terminal_images: bool = True,
    ) -> GeneratedPipeline:
        """
        Generate pipeline using absorbed library (instant, no LLM).

        Args:
            pipeline_name: Name for the generated pipeline
            source_cppipe: Path to source .cppipe file
            modules: ModuleBlocks from .cppipe parser (processing modules only)
            skipped_modules: Infrastructure modules that were skipped

        Returns:
            GeneratedPipeline using registry functions
        """
        return self.builder.generate(
            GeneratedPipelineRequest.from_public_args(
                pipeline_name=pipeline_name,
                source_cppipe=source_cppipe,
                skipped_modules=skipped_modules,
                prune_dead_unmaterialized_artifact_steps=(
                    prune_dead_unmaterialized_artifact_steps
                ),
                materialize_skipped_save_images=materialize_skipped_save_images,
                materialize_terminal_images=materialize_terminal_images,
            ),
            modules=modules,
        )

def python_literal(value: GeneratedLiteralValue) -> str:
    """Render a deterministic generated-code literal for bound setting values."""
    if isinstance(value, Enum):
        return repr(value.value)
    if isinstance(value, tuple):
        trailing_comma = ""
        if len(value) == 1:
            trailing_comma = ","
        return (
            "("
            + ", ".join(python_literal(item) for item in value)
            + trailing_comma
            + ")"
        )
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{python_literal(key)}: {python_literal(item)}"
            for key, item in value.items()
        ) + "}"
    return repr(value)
