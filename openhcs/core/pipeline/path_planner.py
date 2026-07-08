"""
Pipeline path planning - actually reduced duplication.

This version ACTUALLY eliminates duplication instead of adding abstraction theater.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set

from metaclass_registry import AutoRegisterMeta
from openhcs.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.axis_filter import StepAxisFilterSet
from openhcs.core.artifacts import (
    ArtifactGroupScopeSourceRelation,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpecRef,
    ArtifactSpecRelation,
    ArtifactSpec,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    CompiledFunctionPattern,
    FunctionPatternSyntax,
    compile_function_pattern,
    inject_artifact_input_values,
    strip_disabled_functions,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractProviderLike,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
    public_callable_invocation_contract,
)
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    InputConversionPlan,
    MaterializedOutputPlan,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.pipeline.artifact_planning import (
    ArtifactGraph,
    ArtifactProducer,
    extract_artifact_declarations,
)
from openhcs.core.pipeline.compilation_session import (
    CompilationPlateScope,
    CompilationSession,
)
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
)
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    StepSourceBindingsConfig,
    resolve_effective_step_source_bindings,
)
from openhcs.core.step_dependencies import (
    StepInputDependency,
    StepInputDependencyKind,
)

logger = logging.getLogger(__name__)


PlannerGroupKey = str | None
PathPlannerStepDisplayName = str | int


class MissingArtifactInputError(ValueError):
    """Raised when a step consumes an artifact that no prior producer provides."""

    def __init__(
        self,
        *,
        step_id: int,
        artifact_key: str,
        step_name: str | None,
    ) -> None:
        self.step_id = step_id
        self.artifact_key = artifact_key
        self.step_name = step_name
        step_label = step_name or str(step_id)
        super().__init__(
            f"Step {step_id} ({step_label}) needs artifact input "
            f"{artifact_key!r}, but no previous step, source binding, or "
            "metadata resolver provides it."
        )


@dataclass(frozen=True)
class PathPlannerGroupScope:
    """Nominal scope for artifact execution-group planning."""

    keys: tuple[PlannerGroupKey, ...]
    component: AllComponents | None = None

    def __post_init__(self) -> None:
        if self.component is not None:
            object.__setattr__(
                self,
                "component",
                ComponentSet.coerce_component(self.component),
            )

    @classmethod
    def ungrouped(cls) -> "PathPlannerGroupScope":
        return cls((None,))

    @classmethod
    def from_raw(
        cls,
        group_keys: Iterable[Hashable | None],
        *,
        component: AllComponents | None = None,
    ) -> "PathPlannerGroupScope":
        return cls(
            tuple(cls.normalize_key(group_key) for group_key in group_keys),
            component=component,
        )

    @classmethod
    def from_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
    ) -> "PathPlannerGroupScope":
        if output_plan.group_keys:
            return cls.from_raw(
                output_plan.group_keys,
                component=output_plan.group_component,
            )
        return cls.ungrouped()

    @classmethod
    def from_plan(
        cls,
        plan: ArtifactInputPlan | ArtifactOutputPlan,
    ) -> "PathPlannerGroupScope":
        if plan.group_keys:
            return cls.from_raw(
                plan.group_keys,
                component=plan.group_component,
            )
        return cls.ungrouped()

    @staticmethod
    def normalize_key(key: Hashable | None) -> PlannerGroupKey:
        if key is None:
            return None
        return str(key)

    @property
    def is_ungrouped(self) -> bool:
        return self.keys == (None,)

    def output_groups_for(
        self,
        output_names: Iterable[str],
    ) -> dict[str, tuple[PlannerGroupKey, ...]]:
        return {
            output_name: self.keys
            for output_name in output_names
        }

    def missing_from(
        self,
        producer_scope: "PathPlannerGroupScope",
    ) -> list[PlannerGroupKey]:
        return [
            group
            for group in self.keys
            if group not in producer_scope.keys
        ]

    def single_group_key(self) -> PlannerGroupKey | None:
        if len(self.keys) != 1:
            return None
        return self.keys[0]


@dataclass(frozen=True)
class PathPlannerComponentScopes:
    """Component value scopes carried by the main-flow image branch."""

    scopes: Mapping[VariableComponents, PathPlannerGroupScope]

    @classmethod
    def empty(cls) -> "PathPlannerComponentScopes":
        return cls({})

    def scope_for_group_by(
        self,
        group_by: GroupBy | None,
    ) -> PathPlannerGroupScope:
        group_by_component = self.component_from_group_by(group_by)
        if group_by_component is None:
            return PathPlannerGroupScope.ungrouped()
        if group_by_component in self.scopes:
            return self.scopes[group_by_component]
        return PathPlannerGroupScope.ungrouped()

    def output_after(
        self,
        snapshot: StepSnapshot,
        execution_scope: PathPlannerGroupScope,
    ) -> "PathPlannerComponentScopes":
        if not snapshot.is_function_step:
            return self

        scopes = dict(self.scopes)
        variable_components = tuple(snapshot.variable_components or ())
        for component in variable_components:
            scopes[component] = PathPlannerGroupScope.ungrouped()

        group_by = PathPlannerExecutionGroups.normalized_group_by(snapshot)
        group_by_component = self.component_from_group_by(group_by)
        if (
            group_by_component is not None
            and group_by_component not in variable_components
            and not execution_scope.is_ungrouped
        ):
            scopes[group_by_component] = execution_scope

        return PathPlannerComponentScopes(scopes)

    @staticmethod
    def component_from_group_by(group_by: GroupBy | None) -> VariableComponents | None:
        if group_by is None or group_by is GroupBy.NONE:
            return None
        return VariableComponents(group_by.value)

@dataclass(frozen=True)
class ArtifactPlanMaps:
    """Compiled artifact I/O maps for one step."""

    declarations: ArtifactGraph
    group_scope: PathPlannerGroupScope
    inputs: dict[str, ArtifactInputPlan]
    outputs: dict[str, ArtifactOutputPlan]
    inputs_by_group: dict[Optional[str], OrderedDict]
    outputs_by_group: dict[Optional[str], OrderedDict]


class PathPlannerArtifactRelationEffect(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """MRO-selected path-planner behavior for declared artifact relations."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @abstractmethod
    def apply_output_groups(
        self,
        *,
        relation: ArtifactSpecRelation,
        producer: ArtifactProducer,
        plans_by_ref: Mapping[ArtifactSpecRef, ArtifactPlan],
        output_groups: dict[str, PathPlannerGroupScope],
        step_index: int,
        step_name: str | None,
    ) -> None:
        """Apply this relation's output-group effect in place."""


class GroupScopeSourceRelationPathPlannerEffect(PathPlannerArtifactRelationEffect):
    """Propagate output group scope from the relation source artifact."""

    value_type = ArtifactGroupScopeSourceRelation

    def apply_output_groups(
        self,
        *,
        relation: ArtifactSpecRelation,
        producer: ArtifactProducer,
        plans_by_ref: Mapping[ArtifactSpecRef, ArtifactPlan],
        output_groups: dict[str, PathPlannerGroupScope],
        step_index: int,
        step_name: str | None,
    ) -> None:
        if not isinstance(relation, ArtifactGroupScopeSourceRelation):
            raise TypeError(
                "GroupScopeSourceRelationPathPlannerEffect requires "
                "ArtifactGroupScopeSourceRelation."
            )
        source_plan = plans_by_ref.get(relation.source)
        if source_plan is None:
            raise MissingArtifactInputError(
                step_id=step_index,
                artifact_key=relation.source.name,
                step_name=step_name,
            )
        if source_plan.artifact_type is not relation.source.artifact_type:
            raise ValueError(
                f"Artifact output '{producer.name}' declares group lineage "
                f"from {relation.source.artifact_type.value}:{relation.source.name}, "
                f"but the resolved input is "
                f"{source_plan.artifact_type.value}:{source_plan.name}."
            )
        output_groups[producer.name] = PathPlannerGroupScope.from_plan(source_plan)


@dataclass(frozen=True)
class PathPlannerExecutionGroups:
    """Execution-group discovery stage for path planning."""

    planner: PathPlanner

    @staticmethod
    def normalize_group_key(key: Hashable | None) -> PlannerGroupKey:
        return PathPlannerGroupScope.normalize_key(key)

    def get_execution_groups(
        self,
        snapshot: StepSnapshot,
        input_component_scopes: PathPlannerComponentScopes | None = None,
    ) -> PathPlannerGroupScope:
        """Determine which component groups this step will execute for."""
        if not snapshot.is_function_step:
            return PathPlannerGroupScope.ungrouped()

        func_pattern = snapshot.func
        group_by = self.normalized_group_by(snapshot)
        if isinstance(func_pattern, dict):
            scope = PathPlannerGroupScope.from_raw(
                func_pattern.keys(),
                component=self.execution_component_for_dict_pattern(
                    group_by,
                    snapshot.name,
                ),
            )
            logger.debug("Dict function pattern groups: %s", scope.keys)
            return scope

        component_scopes = (
            PathPlannerComponentScopes.empty()
            if input_component_scopes is None
            else input_component_scopes
        )
        scope = component_scopes.scope_for_group_by(
            group_by,
        )
        if scope.is_ungrouped:
            source_scope = self.source_binding_scope_for_group_by(snapshot, group_by)
            if not source_scope.is_ungrouped:
                scope = source_scope
        logger.debug("FunctionStep groups for %s: %s", snapshot.name, scope.keys)
        return scope

    def source_binding_scope_for_group_by(
        self,
        snapshot: StepSnapshot,
        group_by: GroupBy | None,
    ) -> PathPlannerGroupScope:
        """Derive execution groups declared by source-binding component identity."""
        group_by_component = PathPlannerComponentScopes.component_from_group_by(
            group_by
        )
        if group_by_component is None:
            return PathPlannerGroupScope.ungrouped()

        source_bindings = self.planner.source_bindings_for_snapshot(snapshot)
        if not source_bindings.enabled:
            return PathPlannerGroupScope.ungrouped()

        component = ComponentSet.coerce_component(group_by_component)
        group_keys = tuple(
            dict.fromkeys(
                selector.value
                for binding in source_bindings.image_stack_bindings
                for selector in binding.component_identity
                if selector.component is component
            )
        )
        if not group_keys:
            return PathPlannerGroupScope.ungrouped()
        return PathPlannerGroupScope.from_raw(group_keys, component=component)

    @staticmethod
    def execution_component_for_dict_pattern(
        group_by: GroupBy | None,
        step_name: str | None,
    ) -> AllComponents:
        """Return the declared component for dict-pattern runtime dispatch."""
        if group_by is None or group_by is GroupBy.NONE or group_by.value is None:
            raise ValueError(
                f"Step '{step_name}' uses a dict function pattern without a "
                "concrete group_by component. Dict keys are dispatch groups; "
                "GroupBy.NONE is only valid for non-dict function patterns."
            )
        return AllComponents.from_value(group_by.value)

    @staticmethod
    def normalized_group_by(snapshot: StepSnapshot) -> GroupBy:
        """Use the same group_by normalization as compiled execution plans."""
        from openhcs.core.pipeline.funcstep_contract_validator import (
            FuncStepContractValidator,
        )

        return FuncStepContractValidator.normalized_group_by(
            snapshot.group_by,
            snapshot.variable_components,
            snapshot.name,
        )


@dataclass(frozen=True)
class PathPlannerArtifactStage:
    """Artifact declaration, I/O-plan, and FunctionStep injection stage."""

    planner: PathPlanner

    def prepare_step_declarations(
        self,
        snapshot: StepSnapshot,
        input_component_scopes: PathPlannerComponentScopes | None = None,
    ) -> tuple[ArtifactGraph, PathPlannerGroupScope, FunctionPatternSyntax | None]:
        """Normalize a step's function pattern and collect artifact declarations."""
        if not snapshot.is_function_step:
            return ArtifactGraph.empty(), PathPlannerGroupScope.ungrouped(), None

        func_pattern = strip_disabled_functions(snapshot.func)
        source_bindings = self.planner.source_bindings_for_snapshot(snapshot)

        declarations = extract_artifact_declarations(
            self.declaration_pattern(func_pattern),
            declaration_provider=self.planner.declaration_provider,
            invocation_contract_provider=self.planner.invocation_contract_provider,
            step_context=self.artifact_declaration_context(
                snapshot,
                source_bindings=source_bindings,
            ),
        )
        group_scope = self.planner.execution_groups.get_execution_groups(
            snapshot,
            input_component_scopes,
        )
        declarations = self.namespace_grouped_outputs_for_runtime_consumers(
            func_pattern,
            declarations,
            group_scope,
        )
        return declarations, group_scope, func_pattern

    @staticmethod
    def declaration_pattern(
        func_pattern: FunctionPatternSyntax | None,
    ) -> FunctionPatternSyntax:
        """Return the declaration-time pattern, with disabled-only steps empty."""
        if func_pattern is None:
            return []
        return func_pattern

    @classmethod
    def stripped_declaration_pattern(
        cls,
        func_pattern: FunctionPatternSyntax | None,
    ) -> FunctionPatternSyntax:
        """Return declaration pattern after disabled functions are removed."""
        if func_pattern is None:
            return []
        return cls.declaration_pattern(strip_disabled_functions(func_pattern))

    def namespace_grouped_outputs_for_runtime_consumers(
        self,
        func_pattern: FunctionPatternSyntax | None,
        declarations: ArtifactGraph,
        group_scope: PathPlannerGroupScope,
    ) -> ArtifactGraph:
        """Namespace grouped outputs by the step execution groups."""
        output_names = tuple(declarations.outputs)
        if (
            isinstance(func_pattern, dict)
            or group_scope.is_ungrouped
            or not output_names
        ):
            return declarations

        return declarations.with_output_groups(
            group_scope.output_groups_for(output_names)
        )

    def compile_plan_maps(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        declarations: ArtifactGraph,
        group_scope: PathPlannerGroupScope,
    ) -> ArtifactPlanMaps:
        """Compile artifact declarations into runtime I/O maps."""
        step_name = snapshot.name
        group_by = PathPlannerExecutionGroups.normalized_group_by(snapshot)
        source_bindings = self.planner.source_bindings_for_snapshot(snapshot)
        artifact_inputs = self.process_artifact_inputs(
            declarations.inputs,
            declarations.outputs,
            step_index,
            consumer_scope=group_scope,
            step_name=step_name,
        )
        output_groups = self.output_groups_from_declared_relations(
            declarations,
            artifact_inputs,
            group_scope=group_scope,
            source_bindings=source_bindings,
            group_by=group_by,
            step_index=step_index,
            step_name=step_name,
        )
        effective_group_scope = self.execution_scope_from_declared_output_relations(
            declarations,
            group_scope,
            output_groups,
        )
        if effective_group_scope != group_scope:
            group_scope = effective_group_scope
            artifact_inputs = self.process_artifact_inputs(
                declarations.inputs,
                declarations.outputs,
                step_index,
                consumer_scope=group_scope,
                step_name=step_name,
            )
            output_groups = self.output_groups_from_declared_relations(
                declarations,
                artifact_inputs,
                group_scope=group_scope,
                source_bindings=source_bindings,
                group_by=group_by,
                step_index=step_index,
                step_name=step_name,
            )
        artifact_outputs = self.process_artifact_outputs(
            declarations.outputs,
            step_index,
            output_groups,
            step_name=step_name,
        )

        return ArtifactPlanMaps(
            declarations=declarations,
            group_scope=group_scope,
            inputs=artifact_inputs,
            outputs=artifact_outputs,
            inputs_by_group=self.planner.paths.artifact_inputs_by_group(
                artifact_inputs,
                group_scope,
            ),
            outputs_by_group=self.planner.paths.artifact_outputs_by_group(
                artifact_outputs
            ),
        )

    def execution_scope_from_declared_output_relations(
        self,
        declarations: ArtifactGraph,
        group_scope: PathPlannerGroupScope,
        output_groups: Mapping[str, PathPlannerGroupScope],
    ) -> PathPlannerGroupScope:
        """Narrow scalar step execution when output lineage proves a smaller scope."""
        if group_scope.is_ungrouped or not declarations.producers:
            return group_scope
        if not self.outputs_are_namespaced_by_group_scope(declarations, group_scope):
            return group_scope

        relation_scopes: list[PathPlannerGroupScope] = []
        for name, groups in declarations.output_groups.items():
            default_scope = PathPlannerGroupScope.from_raw(
                groups,
                component=group_scope.component,
            )
            resolved_scope = output_groups.get(name, default_scope)
            if resolved_scope != default_scope:
                relation_scopes.append(resolved_scope)

        if not relation_scopes:
            return group_scope
        narrowed = self.union_group_scopes(relation_scopes)
        if narrowed is None:
            return group_scope
        if (
            group_scope.component is not None
            and narrowed.component != group_scope.component
        ):
            return group_scope
        return narrowed

    @staticmethod
    def outputs_are_namespaced_by_group_scope(
        declarations: ArtifactGraph,
        group_scope: PathPlannerGroupScope,
    ) -> bool:
        """Return whether outputs were broadened to the scalar step scope."""
        expected = set(group_scope.keys)
        return all(groups == expected for groups in declarations.output_groups.values())

    @staticmethod
    def union_group_scopes(
        scopes: Sequence[PathPlannerGroupScope],
    ) -> PathPlannerGroupScope | None:
        """Return a single scope containing all keys when components agree."""
        if not scopes:
            return None
        component = scopes[0].component
        keys: list[PlannerGroupKey] = []
        for scope in scopes:
            if scope.component != component:
                return None
            for key in scope.keys:
                if key not in keys:
                    keys.append(key)
        return PathPlannerGroupScope.from_raw(keys, component=component)

    def output_groups_from_declared_relations(
        self,
        declarations: ArtifactGraph,
        artifact_inputs: Mapping[str, ArtifactInputPlan],
        *,
        group_scope: PathPlannerGroupScope,
        source_bindings: StepSourceBindingsConfig,
        group_by: GroupBy | None,
        step_index: int,
        step_name: str | None,
    ) -> Mapping[str, PathPlannerGroupScope]:
        """Return output groups after applying declared artifact relations."""
        output_groups: dict[str, PathPlannerGroupScope] = {
            name: PathPlannerGroupScope.from_raw(
                groups,
                component=group_scope.component,
            )
            for name, groups in declarations.output_groups.items()
        }
        plans_by_ref = self.relation_source_plans_by_ref(
            declarations,
            artifact_inputs,
            group_scope=group_scope,
            source_bindings=source_bindings,
            group_by=group_by,
        )
        for producer in declarations.producers:
            for relation in producer.spec.relations:
                effect = PathPlannerArtifactRelationEffect.for_nominal_value(
                    relation
                )
                if effect is not None:
                    effect.apply_output_groups(
                        relation=relation,
                        producer=producer,
                        plans_by_ref=plans_by_ref,
                        output_groups=output_groups,
                        step_index=step_index,
                        step_name=step_name,
                    )
        return output_groups

    def relation_source_plans_by_ref(
        self,
        declarations: ArtifactGraph,
        artifact_inputs: Mapping[str, ArtifactInputPlan],
        *,
        group_scope: PathPlannerGroupScope,
        source_bindings: StepSourceBindingsConfig,
        group_by: GroupBy | None,
    ) -> dict[ArtifactSpecRef, ArtifactPlan]:
        """Return compiler plans addressable by full artifact spec reference."""
        plans_by_ref: dict[ArtifactSpecRef, ArtifactPlan] = {}
        for name, spec in declarations.inputs.items():
            input_plan = artifact_inputs.get(name)
            if input_plan is not None:
                plans_by_ref[spec.ref()] = input_plan
        plans_by_ref.update(
            self.source_binding_relation_plans_by_ref(
                source_bindings,
                group_scope=group_scope,
                group_by=group_by,
            )
        )
        for output_plan in self.planner.declared.values():
            plans_by_ref[
                ArtifactSpecRef.output(
                    output_plan.name,
                    output_plan.artifact_type,
                )
            ] = output_plan
        return plans_by_ref

    @staticmethod
    def source_binding_relation_plans_by_ref(
        source_bindings: StepSourceBindingsConfig,
        *,
        group_scope: PathPlannerGroupScope,
        group_by: GroupBy | None,
    ) -> dict[ArtifactSpecRef, ArtifactInputPlan]:
        """Return source-bound aliases addressable as relation source plans."""
        plans_by_ref: dict[ArtifactSpecRef, ArtifactInputPlan] = {}
        for binding in source_bindings.binding_declarations:
            binding_scope = PathPlannerArtifactStage.source_binding_group_scope(
                binding,
                group_scope,
                group_by,
            )
            plan = ArtifactInputPlan(
                name=binding.alias,
                path=f"source-binding:{binding.alias}",
                artifact_type=binding.artifact_kind,
                group_keys=binding_scope.keys,
                group_component=binding_scope.component,
            )
            plans_by_ref[
                ArtifactSpecRef.input(
                    plan.name,
                    plan.artifact_type,
                )
            ] = plan
        return plans_by_ref

    @staticmethod
    def source_binding_group_scope(
        binding: NamedSourceBinding,
        group_scope: PathPlannerGroupScope,
        group_by: GroupBy | None,
    ) -> PathPlannerGroupScope:
        """Return relation scope declared by a source binding identity."""

        component = group_scope.component
        if component is None:
            group_by_component = PathPlannerComponentScopes.component_from_group_by(
                group_by
            )
            if group_by_component is not None:
                component = ComponentSet.coerce_component(group_by_component)
        if component is None:
            return group_scope
        identity_values = tuple(
            selector.value
            for selector in binding.component_identity
            if selector.component is component
        )
        if not identity_values:
            return group_scope
        return PathPlannerGroupScope.from_raw(
            identity_values,
            component=component,
        )

    def build_step_compiled_function_pattern(
        self,
        snapshot: StepSnapshot,
        is_function_step: bool,
        func_pattern: FunctionPatternSyntax | None,
        artifact_inputs: Mapping[str, ArtifactInputPlan],
        artifact_outputs: Mapping[str, ArtifactOutputPlan],
    ) -> CompiledFunctionPattern | None:
        """Build the executable function-pattern graph for a FunctionStep."""
        if not is_function_step or not func_pattern:
            return None

        return compile_function_pattern(
            func_pattern,
            artifact_inputs,
            artifact_outputs,
            declaration_provider=self.planner.declaration_provider,
            invocation_contract_provider=self.planner.invocation_contract_provider,
            step_context=self.artifact_declaration_context(snapshot),
            runtime_parameter_bindings=snapshot.callable_runtime_config_bindings,
        )

    def artifact_declaration_context(
        self,
        snapshot: StepSnapshot,
        *,
        source_bindings: StepSourceBindingsConfig | None = None,
    ) -> ArtifactDeclarationStepContext:
        """Return compile-time context for invocation artifact providers."""
        if source_bindings is None:
            source_bindings = self.planner.source_bindings_for_snapshot(snapshot)
        return ArtifactDeclarationStepContext(
            step_name=snapshot.name,
            step_index=snapshot.index,
            source_bindings=source_bindings,
            processing_config=snapshot.processing_config,
        )

    def process_artifact_outputs(
        self,
        outputs: Mapping[str, ArtifactSpec],
        sid: int,
        output_groups: Mapping[str, PathPlannerGroupScope] | None = None,
        step_name: Optional[str] = None,
    ) -> dict[str, ArtifactOutputPlan]:
        """Compile storage plans for artifacts produced by this step."""
        result: dict[str, ArtifactOutputPlan] = {}
        if not outputs:
            return result

        results_path = self.planner.paths.results_path()
        for key, spec in sorted(outputs.items()):
            filename = PipelinePathPlanner._build_axis_filename(
                self.planner.ctx.axis_id,
                key,
                step_index=sid,
            )
            path = results_path / filename
            group_scope = (
                output_groups[key]
                if output_groups is not None and key in output_groups
                else PathPlannerGroupScope.ungrouped()
            )
            normalized_groups = list(group_scope.keys)
            paths_by_group = self.planner.paths.paths_by_group(
                str(path),
                normalized_groups,
            )
            result[key] = ArtifactOutputPlan(
                name=key,
                path=str(path),
                artifact_type=spec.artifact_type,
                materialization=spec.materialization,
                sidecar_role=spec.sidecar_role,
                group_keys=tuple(normalized_groups),
                group_component=group_scope.component,
                paths_by_group=paths_by_group,
                producer_step_index=sid,
                producer_step_scope_id=self.planner.plans[sid].step_scope_id,
                producer_step_name=step_name,
            )
            self.planner.declared[key] = result[key]

        return result

    def process_artifact_inputs(
        self,
        inputs: Mapping[str, ArtifactSpec],
        step_outputs: Mapping[str, ArtifactSpec],
        sid: int,
        consumer_scope: PathPlannerGroupScope,
        step_name: Optional[str] = None,
    ) -> dict[str, ArtifactInputPlan]:
        """Compile storage plans for artifacts consumed by this step."""
        result: dict[str, ArtifactInputPlan] = {}
        if not inputs:
            return result

        for key, input_spec in sorted(inputs.items()):
            if key in self.planner.declared:
                result[key] = self._producer_artifact_input_plan(
                    key,
                    input_spec,
                    consumer_scope,
                    sid,
                    step_name,
                )
            elif key in step_outputs:
                output_spec = step_outputs[key]
                if output_spec.artifact_type != input_spec.artifact_type:
                    raise ValueError(
                        f"Artifact '{key}' is produced as {output_spec.artifact_type.value} "
                        f"but consumed as {input_spec.artifact_type.value} in step '{step_name or sid}'."
                    )
                if output_spec.sidecar_role is not input_spec.sidecar_role:
                    raise ValueError(
                        f"Artifact '{key}' is produced with sidecar role "
                        f"{output_spec.sidecar_role!r} but consumed with "
                        f"{input_spec.sidecar_role!r} in step '{step_name or sid}'."
                    )
                result[key] = ArtifactInputPlan(
                    name=key,
                    path="self",
                    artifact_type=input_spec.artifact_type,
                    sidecar_role=input_spec.sidecar_role,
                    group_component=consumer_scope.component,
                    source_step_id=sid,
                    source_step_scope_id=self.planner.plans[sid].step_scope_id,
                )
            elif not self.planner.ctx.microscope_handler.can_resolve_metadata_artifact(
                key
            ):
                raise MissingArtifactInputError(
                    step_id=sid,
                    artifact_key=key,
                    step_name=step_name,
                )

        return result

    def _producer_artifact_input_plan(
        self,
        key: str,
        input_spec: ArtifactSpec,
        consumer_scope: PathPlannerGroupScope,
        sid: int,
        step_name: Optional[str],
    ) -> ArtifactInputPlan:
        producer = self.planner.declared[key]
        if producer.artifact_type != input_spec.artifact_type:
            producer_name = self._producer_artifact_display_name(producer)
            consumer_name = self._consumer_step_display_name(step_name, sid)
            raise ValueError(
                f"Artifact input '{key}' in step '{consumer_name}' expects "
                f"{input_spec.artifact_type.value}, but producer step '{producer_name}' "
                f"provides {producer.artifact_type.value}."
            )
        if producer.sidecar_role is not input_spec.sidecar_role:
            producer_name = self._producer_artifact_display_name(producer)
            consumer_name = self._consumer_step_display_name(step_name, sid)
            raise ValueError(
                f"Artifact input '{key}' in step '{consumer_name}' expects "
                f"sidecar role {input_spec.sidecar_role!r}, but producer step "
                f"'{producer_name}' provides {producer.sidecar_role!r}."
            )
        producer_scope = PathPlannerGroupScope.from_output_plan(producer)
        producer_paths_by_group = self._producer_artifact_paths_by_group(producer)

        if self._preserve_producer_scope(producer_scope, consumer_scope):
            paths_by_group = producer_paths_by_group.copy()
            return ArtifactInputPlan(
                name=key,
                path=self._producer_artifact_input_path(
                    producer,
                    producer_scope,
                    paths_by_group,
                ),
                artifact_type=producer.artifact_type,
                sidecar_role=input_spec.sidecar_role,
                paths_by_group=paths_by_group,
                group_keys=producer_scope.keys,
                group_component=producer_scope.component,
                source_step_id=producer.producer_step_index,
                source_step_scope_id=producer.producer_step_scope_id,
            )
        if not producer_scope.is_ungrouped and consumer_scope.is_ungrouped:
            paths_by_group = producer_paths_by_group.copy()
        elif not producer_scope.is_ungrouped:
            missing = consumer_scope.missing_from(producer_scope)
            if missing:
                if not producer_paths_by_group:
                    producer_name = self._producer_artifact_display_name(producer)
                    consumer_name = self._consumer_step_display_name(step_name, sid)
                    raise ValueError(
                        f"Artifact input '{key}' in step '{consumer_name}' cannot be resolved: "
                        f"producer step '{producer_name}' provides groups {producer_scope.keys}, "
                        f"but consumer needs {missing}."
                    )
                logger.debug(
                    "Artifact input %r in step %r preserves producer groups %s "
                    "for wider consumer groups %s.",
                    key,
                    step_name or sid,
                    producer_scope.keys,
                    consumer_scope.keys,
                )
                paths_by_group = producer_paths_by_group.copy()
                return ArtifactInputPlan(
                    name=key,
                    path=self._producer_artifact_input_path(
                        producer,
                        producer_scope,
                        paths_by_group,
                    ),
                    artifact_type=producer.artifact_type,
                    sidecar_role=input_spec.sidecar_role,
                    paths_by_group=paths_by_group,
                    group_keys=producer_scope.keys,
                    group_component=producer_scope.component,
                    source_step_id=producer.producer_step_index,
                    source_step_scope_id=producer.producer_step_scope_id,
                )
            paths_by_group = {
                group: producer_paths_by_group[group]
                for group in consumer_scope.keys
                if group in producer_paths_by_group
            }
        else:
            paths_by_group = {
                group: producer.path for group in consumer_scope.keys
            }
            producer_scope = consumer_scope

        return ArtifactInputPlan(
            name=key,
            path=self._producer_artifact_input_path(
                producer,
                producer_scope,
                paths_by_group,
            ),
            artifact_type=producer.artifact_type,
            sidecar_role=input_spec.sidecar_role,
            paths_by_group=paths_by_group,
            group_keys=tuple(paths_by_group.keys()),
            group_component=producer_scope.component,
            source_step_id=producer.producer_step_index,
            source_step_scope_id=producer.producer_step_scope_id,
        )

    @staticmethod
    def _preserve_producer_scope(
        producer_scope: PathPlannerGroupScope,
        consumer_scope: PathPlannerGroupScope,
    ) -> bool:
        """Return whether producer groups are typed by a different component."""
        return (
            not producer_scope.is_ungrouped
            and not consumer_scope.is_ungrouped
            and producer_scope.component is not None
            and consumer_scope.component is not None
            and producer_scope.component != consumer_scope.component
        )

    @staticmethod
    def _producer_artifact_display_name(
        producer: ArtifactOutputPlan,
    ) -> PathPlannerStepDisplayName:
        if producer.producer_step_name is not None:
            return producer.producer_step_name
        if producer.producer_step_index is not None:
            return producer.producer_step_index
        return "unknown"

    @staticmethod
    def _consumer_step_display_name(
        step_name: Optional[str],
        sid: int,
    ) -> PathPlannerStepDisplayName:
        if step_name is not None:
            return step_name
        return sid

    @staticmethod
    def _producer_artifact_paths_by_group(
        producer: ArtifactOutputPlan,
    ) -> dict[Optional[str], str]:
        if producer.paths_by_group is None:
            return {}
        return dict(producer.paths_by_group)

    @staticmethod
    def _producer_artifact_input_path(
        producer: ArtifactOutputPlan,
        producer_scope: PathPlannerGroupScope,
        paths_by_group: Mapping[str | None, str],
    ) -> str:
        producer_group = producer_scope.single_group_key()
        if producer_group is not None:
            return paths_by_group.get(producer_group, producer.path)
        return producer.path

    def inject_metadata(
        self,
        pattern: FunctionPatternSyntax,
        inputs: Mapping[str, ArtifactSpec],
    ) -> FunctionPatternSyntax:
        """Inject metadata for artifact inputs."""
        for key in inputs:
            if (
                key not in self.planner.declared
                and self.planner.ctx.microscope_handler.can_resolve_metadata_artifact(
                    key
                )
            ):
                value = self.planner.ctx.microscope_handler.resolve_metadata_artifact(
                    key,
                    self.planner.ctx.plate_path,
                )
                pattern = inject_artifact_input_values(pattern, {key: value})
        return pattern

@dataclass(frozen=True)
class PathPlannerMaterializationStage:
    """Input conversion and materialized-output planning stage."""

    planner: PathPlanner

    def materialized_output_dir_for_step(
        self,
        snapshot: StepSnapshot,
    ) -> Optional[Path]:
        """Resolve optional per-step materialization output directory."""
        materialization_config = snapshot.materialization_config
        if not materialization_config or not materialization_config.enabled:
            return None

        step_axis_filters = self.planner.ctx.step_axis_filters.get(
            snapshot.index,
            StepAxisFilterSet.empty(),
        )
        if not step_axis_filters.allows(materialization_config, self.planner.ctx.axis_id):
            logger.debug(
                "Skipping materialization for step %s, axis %s (filtered out)",
                snapshot.name,
                self.planner.ctx.axis_id,
            )
            return None

        return self.planner.paths.build_output_path(materialization_config)

    def input_conversion_plan_for_step(
        self,
        step_index: int,
        input_dir: Path,
    ) -> Optional[InputConversionPlan]:
        """Resolve optional compiler-provided or config-provided input conversion."""
        existing_plan = self.planner.plans[step_index].input_conversion
        if existing_plan is not None:
            return existing_plan

        output_dir = self.planner.paths.input_conversion_output_path(step_index)
        if output_dir is None:
            return None

        return InputConversionPlan(
            output_dir=output_dir,
            backend=self.planner.vfs.materialization_backend.value,
            uses_virtual_workspace=False,
            original_subdir=input_dir.name,
        )

    def apply_materialization_plan(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        materialized_output_dir: Optional[Path],
    ) -> None:
        """Attach optional materialization path fields to a step plan."""
        if not materialized_output_dir:
            return

        materialization_config = snapshot.materialization_config
        materialized_plate_root = self.planner.paths.build_output_plate_root(
            self.planner.plate_path,
            materialization_config,
            is_per_step_materialization=False,
        )
        self.planner.plans[step_index].materialized_output = MaterializedOutputPlan(
            output_dir=materialized_output_dir,
            backend=self.planner.vfs.materialization_backend.value,
            plate_root=str(materialized_plate_root),
            sub_dir=materialization_config.sub_dir,
            analysis_results_dir=str(
                self.planner.paths.analysis_results_dir_for(materialized_output_dir)
            ),
        )
        self.planner.plans[step_index].materialization_config = materialization_config

    def apply_input_conversion_plan(
        self,
        step_index: int,
        input_conversion_plan: Optional[InputConversionPlan],
    ) -> None:
        """Attach optional input conversion path fields to a step plan."""
        if input_conversion_plan is None:
            return

        self.planner.plans[step_index].input_conversion = input_conversion_plan


@dataclass(frozen=True)
class PathPlannerValidationStage:
    """Connectivity and materialization path validation stage."""

    planner: PathPlanner

    def validate(self) -> None:
        """Validate connectivity and materialization paths."""
        for i in range(1, self.planner.session.step_count):
            curr = self.planner.session.snapshot(i)
            dependency = self.planner.plans[i].main_input_dependency
            if dependency.kind is StepInputDependencyKind.PIPELINE_START:
                continue
            if dependency.kind is not StepInputDependencyKind.STEP_OUTPUT:
                raise ValueError(
                    f"Step {curr.name} has unresolved main input dependency."
                )
            source_step_index = dependency.source_step_index
            if source_step_index is None:
                raise ValueError(
                    f"Step {curr.name} main input dependency is missing source_step_index."
                )
            curr_in = self.planner.plans[i].input_dir
            source_out = self.planner.plans[source_step_index].output_dir
            if curr_in != source_out:
                has_artifact_bridge = any(
                    inp.source_step_id in [source_step_index, "prev"]
                    or inp.source_step_scope_id == dependency.source_step_scope_id
                    for inp in self.planner.plans[i].artifact_inputs.values()
                )
                if not has_artifact_bridge:
                    producer_name = self.planner.session.snapshot(source_step_index).name
                    raise ValueError(f"Disconnect: {producer_name} -> {curr.name}")

        self.validate_materialization_paths()

    def validate_materialization_paths(self) -> None:
        """Validate and resolve materialization path collisions."""
        global_path = self.planner.paths.build_output_path(self.planner.cfg)

        mat_steps = [
            (
                snapshot,
                self.planner.plans[i].pipeline_position or i,
                self.planner.paths.build_output_path(snapshot.materialization_config),
            )
            for i, snapshot in self.planner.session.indexed_snapshots()
            if snapshot.materialization_config
            and snapshot.materialization_config.enabled
        ]

        path_groups = defaultdict(list)
        for snapshot, pos, path in mat_steps:
            if path == global_path:
                self.resolve_and_update_paths(snapshot, pos, path, "main flow")
            else:
                path_groups[str(path)].append((snapshot, pos, path))

        for path_key, step_list in path_groups.items():
            if len(step_list) > 1:
                for snapshot, pos, path in step_list:
                    self.resolve_and_update_paths(snapshot, pos, path, f"pos {pos}")

    def resolve_and_update_paths(
        self,
        snapshot: StepSnapshot,
        position: int,
        original_path: Path,
        conflict_type: str,
    ) -> None:
        """Resolve path conflict by updating the compiled plan only."""
        del original_path, conflict_type
        materialization_config = snapshot.materialization_config

        original_sub_dir = materialization_config.sub_dir
        new_sub_dir = f"{original_sub_dir}_step{position}"

        from dataclasses import replace
        updated_config = replace(materialization_config, sub_dir=new_sub_dir)

        resolved_path = self.planner.paths.build_output_path(updated_config)
        resolved_analysis_results_dir = self.planner.paths.analysis_results_dir_for(
            resolved_path
        )

        if step_plan := self.planner.plans.get(position):
            if step_plan.materialized_output is not None:
                step_plan.materialized_output = MaterializedOutputPlan(
                    output_dir=resolved_path,
                    backend=step_plan.materialized_output.backend,
                    plate_root=step_plan.materialized_output.plate_root,
                    sub_dir=new_sub_dir,
                    analysis_results_dir=str(resolved_analysis_results_dir),
                )
                step_plan.materialization_config = updated_config


@dataclass(frozen=True)
class PathPlannerPathAuthority:
    """Path and grouped-artifact expansion authority for compiled step plans."""

    planner: PathPlanner

    @staticmethod
    def build_output_plate_root(
        plate_path: Path,
        path_config,
        is_per_step_materialization: bool = False,
    ) -> Path:
        """Build output plate root directory directly from configuration components.

        Results always use the output plate path so metadata remains colocated with
        processed images instead of the original input images.
        """
        del is_per_step_materialization

        if not path_config.output_dir_suffix:
            raise ValueError(
                f"output_dir_suffix cannot be None or empty. "
                f"Results must always use output plate path, not input plate path. "
                f"Config: {path_config}"
            )

        return Path(
            _cached_output_plate_root(
                str(plate_path),
                (
                    None
                    if path_config.global_output_folder is None
                    else str(path_config.global_output_folder)
                ),
                path_config.output_dir_suffix,
            )
        )

    @staticmethod
    def paths_by_group(
        base_path: str,
        group_keys: List[Optional[str]],
    ) -> Dict[Optional[str], str]:
        """Expand one artifact path into per-execution-group artifact paths."""
        return dict(_cached_paths_by_group(base_path, tuple(group_keys)))

    @staticmethod
    def artifact_outputs_by_group(
        artifact_outputs: Dict[str, ArtifactOutputPlan],
    ) -> Dict[Optional[str], OrderedDict]:
        """Expand artifact outputs into per-group plans with finalized paths."""
        if not artifact_outputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = defaultdict(OrderedDict)
        for output_key, output_plan in artifact_outputs.items():
            paths_by_group = output_plan.paths_by_group or {None: output_plan.path}
            for group_key in paths_by_group:
                grouped[group_key][output_key] = output_plan.for_group(group_key)
        return dict(grouped)

    @staticmethod
    def artifact_inputs_by_group(
        artifact_inputs: Dict[str, ArtifactInputPlan],
        consumer_scope: PathPlannerGroupScope,
    ) -> Dict[Optional[str], OrderedDict]:
        """Expand artifact inputs into per-group plans with finalized paths."""
        if not artifact_inputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = {}
        for group_key in consumer_scope.keys:
            per_group = OrderedDict()
            for input_key, input_plan in artifact_inputs.items():
                group_plan = PathPlannerPathAuthority.artifact_input_for_consumer_group(
                    input_plan,
                    group_key,
                    consumer_scope.component,
                )
                if group_plan is not None:
                    per_group[input_key] = group_plan
            if per_group:
                grouped[group_key] = per_group
        return grouped

    @staticmethod
    def artifact_input_for_consumer_group(
        input_plan: ArtifactInputPlan,
        consumer_group_key: str | None,
        consumer_group_component: AllComponents | None,
    ) -> ArtifactInputPlan | None:
        """Return an input plan for one grouped stack without crossing components."""
        if (
            input_plan.group_component is None
            or consumer_group_component is None
            or input_plan.group_component == consumer_group_component
        ):
            group_plan = input_plan.for_group(consumer_group_key)
            if group_plan is not None:
                return group_plan

        input_group_key = input_plan.single_group_key
        if input_group_key is not None:
            return input_plan.for_group(input_group_key)
        return None

    @staticmethod
    def analysis_results_dir_for(image_dir: Path) -> Path:
        """Return the analysis-results sibling directory for an image directory."""
        return Path(_cached_analysis_results_dir_for(str(image_dir)))

    def build_output_path(self, path_config=None) -> Path:
        """Build complete output path: plate_root + sub_dir."""
        config = path_config or self.planner.cfg
        if not config.output_dir_suffix:
            raise ValueError(
                f"output_dir_suffix cannot be None or empty. "
                f"Results must always use output plate path, not input plate path. "
                f"Config: {config}"
            )
        return Path(
            _cached_output_path(
                str(self.planner.plate_path),
                (
                    None
                    if config.global_output_folder is None
                    else str(config.global_output_folder)
                ),
                config.output_dir_suffix,
                config.sub_dir,
            )
        )

    def input_conversion_output_path(self, step_index: int) -> Optional[Path]:
        """Get input conversion output path if config exists."""
        config = self.planner.plans[step_index].input_conversion_config
        if config is not None:
            return self.build_output_path(config)
        return None

    def results_path(self) -> Path:
        """Get analysis results path from global pipeline configuration."""
        path = self.planner.session.global_config.materialization_results_path
        if not self.planner.cfg.output_dir_suffix:
            raise ValueError(
                f"output_dir_suffix cannot be None or empty. "
                f"Results must always use output plate path, not input plate path. "
                f"Config: {self.planner.cfg}"
            )
        return Path(
            _cached_results_path(
                str(self.planner.plate_path),
                (
                    None
                    if self.planner.cfg.global_output_folder is None
                    else str(self.planner.cfg.global_output_folder)
                ),
                self.planner.cfg.output_dir_suffix,
                str(path),
            )
        )


@dataclass(frozen=True)
class PathPlannerStepAssemblyStage:
    """Per-step dependency, directory, and compiled-plan assembly stage."""

    planner: PathPlanner

    def prime_future_artifact_inputs(self) -> None:
        """Precompute artifact input keys used by later steps for each step index."""
        future_inputs: Set[str] = set()
        self.planner.future_artifact_inputs = [
            set() for _ in range(self.planner.session.step_count)
        ]

        for i in self.planner.session.reverse_snapshot_indices():
            self.planner.future_artifact_inputs[i] = set(future_inputs)

            snapshot = self.planner.session.snapshot(i)
            if snapshot.is_function_step:
                pattern = self.planner.artifacts.stripped_declaration_pattern(
                    snapshot.func
                )
                declarations = extract_artifact_declarations(
                    pattern,
                    declaration_provider=self.planner.declaration_provider,
                    invocation_contract_provider=(
                        self.planner.invocation_contract_provider
                    ),
                    step_context=self.planner.artifacts.artifact_declaration_context(
                        snapshot
                    ),
                )
                step_inputs = set(declarations.inputs.keys())
            else:
                step_inputs = set()

            future_inputs.update(step_inputs)

    def plan_step(self, snapshot: StepSnapshot, step_index: int) -> None:
        """Plan one step's directories, artifacts, and executable pattern."""
        self.planner.plans[step_index].step_scope_id = snapshot.scope_id
        main_input_dependency = self.main_input_dependency(snapshot, step_index)
        input_component_scopes = self.input_component_scopes(main_input_dependency)
        input_dir, output_dir = self.step_io_dirs(main_input_dependency, step_index)

        declarations, group_scope, func_pattern = (
            self.planner.artifacts.prepare_step_declarations(
                snapshot,
                input_component_scopes,
            )
        )
        artifact_maps = self.planner.artifacts.compile_plan_maps(
            snapshot,
            step_index,
            declarations,
            group_scope,
        )

        if snapshot.is_function_step and any(
            self.planner.ctx.microscope_handler.can_resolve_metadata_artifact(k)
            for k in declarations.inputs
        ):
            func_pattern = self.planner.artifacts.inject_metadata(
                func_pattern,
                declarations.inputs,
            )

        self.planner.plans[step_index].func = func_pattern
        self.update_core_step_plan(
            snapshot,
            step_index,
            main_input_dependency,
            input_dir,
            output_dir,
            artifact_maps,
            self.planner.artifacts.build_step_compiled_function_pattern(
                snapshot,
                snapshot.is_function_step,
                func_pattern,
                artifact_maps.inputs,
                artifact_maps.outputs,
            ),
        )
        self.planner.materialization.apply_materialization_plan(
            snapshot,
            step_index,
            self.planner.materialization.materialized_output_dir_for_step(snapshot),
        )
        self.planner.materialization.apply_input_conversion_plan(
            step_index,
            self.planner.materialization.input_conversion_plan_for_step(
                step_index,
                input_dir,
            ),
        )
        self.planner.main_flow_component_scopes[step_index] = (
            input_component_scopes.output_after(snapshot, artifact_maps.group_scope)
        )

    def input_component_scopes(
        self,
        main_input_dependency: StepInputDependency,
    ) -> PathPlannerComponentScopes:
        """Return component scopes visible on a step's main-flow input branch."""
        if main_input_dependency.kind is StepInputDependencyKind.PIPELINE_START:
            return PathPlannerComponentScopes.empty()
        source_step_index = main_input_dependency.source_step_index
        if source_step_index is None:
            return PathPlannerComponentScopes.empty()
        return self.planner.main_flow_component_scopes.get(
            source_step_index,
            PathPlannerComponentScopes.empty(),
        )

    def update_core_step_plan(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        main_input_dependency: StepInputDependency,
        input_dir: Path,
        output_dir: Path,
        artifact_maps: ArtifactPlanMaps,
        compiled_function_pattern: CompiledFunctionPattern | None,
    ) -> None:
        """Write the always-present path and artifact planning fields."""
        main_plate_root = self.planner.paths.build_output_plate_root(
            self.planner.plate_path,
            self.planner.cfg,
            is_per_step_materialization=False,
        )
        step_plan = self.planner.plans[step_index]
        step_plan.step_scope_id = snapshot.scope_id
        step_plan.input_dir = input_dir
        step_plan.output_dir = output_dir
        step_plan.output_plate_root = str(main_plate_root)
        step_plan.sub_dir = self.planner.cfg.sub_dir
        step_plan.analysis_results_dir = str(
            self.planner.paths.analysis_results_dir_for(Path(output_dir))
        )
        step_plan.pipeline_position = step_index
        step_plan.input_source = self.input_source(snapshot)
        step_plan.group_by = PathPlannerExecutionGroups.normalized_group_by(snapshot)
        step_plan.main_input_dependency = main_input_dependency
        step_plan.artifact_inputs = artifact_maps.inputs
        step_plan.artifact_outputs = artifact_maps.outputs
        step_plan.artifact_inputs_by_group = artifact_maps.inputs_by_group
        step_plan.artifact_outputs_by_group = artifact_maps.outputs_by_group
        step_plan.execution_groups = list(artifact_maps.group_scope.keys)
        step_plan.execution_group_component = artifact_maps.group_scope.component
        step_plan.compiled_function_pattern = compiled_function_pattern

    def main_input_dependency(
        self,
        snapshot: StepSnapshot,
        step_index: int,
    ) -> StepInputDependency:
        """Resolve the explicit main-input edge for one step."""
        existing_plan = self.planner.plans.get(step_index)
        if (
            existing_plan is not None
            and existing_plan.main_input_dependency.is_resolved
        ):
            return existing_plan.main_input_dependency

        if step_index == 0 or snapshot.input_source == InputSource.PIPELINE_START:
            return StepInputDependency.pipeline_start()

        producer_index = step_index - 1
        producer_scope_id = self.planner.session.snapshot(producer_index).scope_id
        return StepInputDependency.step_output(
            source_step_index=producer_index,
            source_step_scope_id=producer_scope_id,
        )

    def step_io_dirs(
        self,
        main_input_dependency: StepInputDependency,
        step_index: int,
    ) -> tuple[Path, Path]:
        """Resolve read/write directories for one step."""
        plan = self.planner.plans.get(step_index)
        reads_from_pipeline_start = (
            main_input_dependency.kind is StepInputDependencyKind.PIPELINE_START
        )

        if plan is not None and plan.input_dir is not None:
            input_dir = Path(plan.input_dir)
        elif reads_from_pipeline_start:
            input_dir = self.planner.initial_input
        else:
            source_step_index = main_input_dependency.source_step_index
            if source_step_index is None:
                raise ValueError(
                    f"Step {step_index} main input dependency is missing source_step_index."
                )
            input_dir = Path(self.planner.plans[source_step_index].output_dir)

        if plan is not None and plan.output_dir is not None:
            output_dir = Path(plan.output_dir)
        elif reads_from_pipeline_start:
            output_dir = self.planner.paths.build_output_path()
        else:
            output_dir = input_dir

        return input_dir, output_dir

    @staticmethod
    def input_source(snapshot: StepSnapshot) -> str:
        """Get input source string."""
        if snapshot.input_source == InputSource.PIPELINE_START:
            return "PIPELINE_START"
        return "PREVIOUS_STEP"


# ===== PATH PLANNING (NO duplication) =====

class PathPlanner:
    """Minimal path planner with zero duplication."""

    def __init__(
        self,
        session: CompilationSession,
        declaration_provider: InvocationArtifactDeclarationProviderLike = (
            callable_contract_artifact_declarations
        ),
        invocation_contract_provider: InvocationContractProviderLike = (
            public_callable_invocation_contract
        ),
    ):
        self.session = session
        self.ctx = session.context
        self.cfg = session.global_config.path_planning_config
        self.vfs = session.global_config.vfs_config
        self.plans: dict[int, CompiledStepPlan] = session.plans
        self.declared = {}  # Tracks artifact outputs
        self.orchestrator = session.orchestrator
        self.source_bindings_defaults = (
            session.orchestrator.pipeline_config.source_bindings_config
        )
        self.step_source_bindings_defaults = (
            session.orchestrator.pipeline_config.step_source_bindings_config
        )
        self.declaration_provider = declaration_provider
        self.invocation_contract_provider = invocation_contract_provider
        self.future_artifact_inputs: List[Set[str]] = [
            set() for _ in range(session.step_count)
        ]
        self.main_flow_component_scopes: dict[int, PathPlannerComponentScopes] = {}
        self.execution_groups = PathPlannerExecutionGroups(self)
        self.paths = PathPlannerPathAuthority(self)
        self.artifacts = PathPlannerArtifactStage(self)
        self.materialization = PathPlannerMaterializationStage(self)
        self.validation = PathPlannerValidationStage(self)
        self.steps = PathPlannerStepAssemblyStage(self)

        # Initial input determination (once)
        self.initial_input = Path(self.ctx.input_dir)
        self.plate_scope = CompilationPlateScope.from_context(self.ctx)
        self.plate_path = self.plate_scope.path

    def source_bindings_for_snapshot(
        self,
        snapshot: StepSnapshot,
    ) -> StepSourceBindingsConfig:
        """Return step source bindings resolved through pipeline defaults."""
        return resolve_effective_step_source_bindings(
            snapshot.source_bindings,
            source_bindings_defaults=self.source_bindings_defaults,
            step_source_bindings_defaults=self.step_source_bindings_defaults,
            activate_source_bindings=(
                snapshot.input_source == InputSource.PIPELINE_START
            ),
        )

    def plan(self) -> dict[int, CompiledStepPlan]:
        """Plan all paths with zero duplication."""
        self.steps.prime_future_artifact_inputs()
        for i, snapshot in self.session.indexed_snapshots():
            self.steps.plan_step(snapshot, i)

        self.validation.validate()

        # Set output_plate_root and sub_dir for metadata writing
        if self.session.step_count:
            self.ctx.output_plate_root = self.paths.build_output_plate_root(
                self.plate_path,
                self.cfg,
                is_per_step_materialization=False,
            )
            self.ctx.sub_dir = self.cfg.sub_dir

        return self.plans

# ===== PUBLIC API =====

class PipelinePathPlanner:
    """Public API matching original interface."""

    @staticmethod
    def prepare_pipeline_paths(
        session: CompilationSession,
        declaration_provider: InvocationArtifactDeclarationProviderLike = (
            callable_contract_artifact_declarations
        ),
        invocation_contract_provider: InvocationContractProviderLike = (
            public_callable_invocation_contract
        ),
    ) -> Dict:
        """Prepare path plans for an already resolved compilation session."""
        return PathPlanner(
            session,
            declaration_provider=declaration_provider,
            invocation_contract_provider=invocation_contract_provider,
        ).plan()

    @staticmethod
    def build_output_plate_root(
        plate_path: Path,
        path_config,
        is_per_step_materialization: bool = False,
    ) -> Path:
        """Build output plate root from configuration components."""
        return PathPlannerPathAuthority.build_output_plate_root(
            plate_path,
            path_config,
            is_per_step_materialization=is_per_step_materialization,
        )

    @staticmethod
    def _build_axis_filename(axis_id: str, key: str, extension: str = "pkl", step_index: Optional[int] = None) -> str:
        """Build standardized axis-based filename with optional step index.

        Args:
            axis_id: Well/axis identifier (e.g., "R02C02")
            key: Artifact output key (e.g., "match_results")
            extension: File extension (default: "pkl")
            step_index: Optional step index to prevent collisions when multiple steps
                       produce the same artifact output

        Returns:
            Filename string (e.g., "R02C02_match_results_step3.pkl")
        """
        return _cached_axis_filename(axis_id, key, extension, step_index)

    @staticmethod
    def build_dict_pattern_path(base_path: str, dict_key: str) -> str:
        """Build channel-specific path for dict patterns.

        Inserts _w{dict_key} after well ID in the filename.
        Example: "dir/A01_rois_step7.pkl" + "1" -> "dir/A01_w1_rois_step7.pkl"

        Args:
            base_path: Base path without channel component
            dict_key: Dict pattern key (e.g., "1" for channel 1)

        Returns:
            Channel-specific path
        """
        return _cached_dict_pattern_path(base_path, dict_key)


@lru_cache(maxsize=65536)
def _cached_output_plate_root(
    plate_path: str,
    global_output_folder: str | None,
    output_dir_suffix: str,
) -> str:
    """Return the output plate root for one normalized path config."""
    path = Path(plate_path)
    if plate_path.startswith("/omero/"):
        base = path.parent
    elif global_output_folder:
        base = Path(global_output_folder)
    else:
        base = path.parent
    return str(base / f"{path.name}{output_dir_suffix}")


@lru_cache(maxsize=65536)
def _cached_output_path(
    plate_path: str,
    global_output_folder: str | None,
    output_dir_suffix: str,
    sub_dir: str,
) -> str:
    """Return the configured image output path for one normalized path config."""
    return str(
        Path(
            _cached_output_plate_root(
                plate_path,
                global_output_folder,
                output_dir_suffix,
            )
        )
        / sub_dir
    )


@lru_cache(maxsize=65536)
def _cached_results_path(
    plate_path: str,
    global_output_folder: str | None,
    output_dir_suffix: str,
    materialization_results_path: str,
) -> str:
    """Return the artifact results path for one normalized path config."""
    results_path = Path(materialization_results_path)
    if results_path.is_absolute():
        return str(results_path)
    output_plate_root = Path(
        _cached_output_plate_root(
            plate_path,
            global_output_folder,
            output_dir_suffix,
        )
    )
    return str(output_plate_root / results_path)


@lru_cache(maxsize=131072)
def _cached_axis_filename(
    axis_id: str,
    key: str,
    extension: str,
    step_index: int | None,
) -> str:
    """Return the standard artifact filename for one axis/output/step."""
    if step_index is not None:
        return f"{axis_id}_{key}_step{step_index}.{extension}"
    return f"{axis_id}_{key}.{extension}"


@lru_cache(maxsize=65536)
def _cached_dict_pattern_path(base_path: str, dict_key: str) -> str:
    """Return the grouped artifact path for one base path and group key."""
    path = Path(base_path)
    dir_part = path.parent
    filename = path.name
    well_id, rest = filename.split("_", 1)
    return str(dir_part / f"{well_id}_w{dict_key}_{rest}")


@lru_cache(maxsize=32768)
def _cached_paths_by_group(
    base_path: str,
    group_keys: tuple[str | None, ...],
) -> tuple[tuple[str | None, str], ...]:
    """Return immutable grouped-path items for cache-safe reuse."""
    return tuple(
        (
            group_key,
            (
                base_path
                if group_key is None
                else _cached_dict_pattern_path(base_path, group_key)
            ),
        )
        for group_key in group_keys
    )


@lru_cache(maxsize=8192)
def _cached_analysis_results_dir_for(image_dir: str) -> str:
    """Return cached analysis-results sibling path for one image directory."""
    path = Path(image_dir)
    return str(path.parent / f"{path.name}_results")
