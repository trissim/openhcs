"""Compiler-side contract derivation for generated CellProfiler steps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
)
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    NormalizedFunctionGroup,
    NormalizedFunctionItem,
    NormalizedFunctionPattern,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    CompositeInvocationContractProvider,
    InvocationContractPlan,
    InvocationContractProvider,
    InvocationContractProviderFactory,
    unnamed_main_flow_artifact_name,
)
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.artifact_planning import (
    extract_artifact_declarations,
)
from openhcs.core.steps.function_step import FunctionStep


@dataclass(frozen=True, slots=True)
class CellProfilerInvocationContractProvider(InvocationContractProvider):
    """Session-scoped exact CellProfiler invocation-contract provider."""

    plans: Mapping[
        tuple[int, FunctionInvocationKey],
        InvocationContractPlan,
    ]

    def __post_init__(self) -> None:
        normalized: dict[tuple[int, FunctionInvocationKey], InvocationContractPlan] = {}
        for key, plan in self.plans.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], int)
                or not isinstance(key[1], FunctionInvocationKey)
            ):
                raise TypeError(
                    "CellProfilerInvocationContractProvider keys must be "
                    "(int, FunctionInvocationKey) tuples."
                )
            if not isinstance(plan, InvocationContractPlan):
                raise TypeError(
                    "CellProfilerInvocationContractProvider values must be "
                    f"InvocationContractPlan, got {type(plan).__name__}."
                )
            if key in normalized:
                raise ValueError(
                    f"Duplicate CellProfiler invocation contract key {key!r}."
                )
            normalized[key] = plan
        if not normalized:
            raise ValueError(
                "CellProfilerInvocationContractProvider requires at least one plan."
            )
        object.__setattr__(self, "plans", MappingProxyType(normalized))

    def __call__(
        self,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationContractPlan | None:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
        )

        try:
            module_type = CellProfilerModule.for_callable_contract(
                invocation.contract
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"CellProfiler contract lookup failed for step "
                f"{step_context.step_index!r} ({step_context.step_name!r}), "
                f"invocation {invocation.key!r}: {exc}"
            ) from exc
        if module_type is None:
            return None
        step_index = step_context.step_index
        if not isinstance(step_index, int):
            raise TypeError(
                "CellProfiler invocation contract lookup requires an integer "
                "step index."
            )
        key = (step_index, invocation.key)
        try:
            return self.plans[key]
        except KeyError as exc:
            raise ValueError(
                f"CellProfiler contract lookup failed for step {step_index} "
                f"({step_context.step_name!r}), invocation {invocation.key!r}, "
                f"module {module_type.__name__}: no exact contract was compiled "
                f"for key {key!r}."
            ) from exc


class CellProfilerInvocationContractProviderFactory(InvocationContractProviderFactory):
    """Compile exact CellProfiler invocation contracts from public snapshots."""

    @classmethod
    def provider_for_session(
        cls,
        session: CompilationSession,
    ) -> InvocationContractProvider | None:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
        )
        from openhcs.interop.cellprofiler.runtime.adapter import (
            CellProfilerRuntimeAdapter,
        )

        plans: dict[
            tuple[int, FunctionInvocationKey],
            InvocationContractPlan,
        ] = {}
        forward_context = ArtifactDeclarationStepContext.empty()
        next_module_num = 1
        for snapshot in session.snapshots:
            if (
                not snapshot.step.enabled
                or not isinstance(snapshot.step, FunctionStep)
                or snapshot.step.func is None
            ):
                continue
            source_bindings = snapshot.step.source_bindings
            effective_source_bindings = source_bindings.for_input_source(
                snapshot.step.processing_config.input_source
            )
            forward_context = replace(
                forward_context,
                step_name=snapshot.step.name,
                step_index=snapshot.index,
            ).with_source_binding_scope(
                source_bindings=effective_source_bindings,
                group_by=snapshot.step.processing_config.group_by,
                input_source=snapshot.step.processing_config.input_source,
            )
            step_context = forward_context

            normalized_pattern = normalize_function_pattern(snapshot.step.func)
            first_step_module_num = next_module_num
            step_invocations = []
            group_contexts: list[ArtifactDeclarationStepContext] = []
            for group in normalized_pattern.groups:
                group_context = step_context
                for invocation in group.items:
                    module_type = CellProfilerModule.for_callable_contract(
                        invocation.contract
                    )
                    if module_type is None:
                        native_graph = extract_artifact_declarations(
                            NormalizedFunctionPattern(
                                groups=(
                                    NormalizedFunctionGroup(
                                        source_group_key=group.source_group_key,
                                        items=(invocation,),
                                    ),
                                ),
                                is_grouped=normalized_pattern.is_grouped,
                            ),
                            step_context=group_context,
                        )
                        declared_outputs = tuple(native_graph.outputs.values())
                        native_images = tuple(
                            spec
                            for spec in declared_outputs
                            if spec.artifact_type is ImageArtifactType
                            and spec.participates_in_main_flow
                        )
                        next_main_flow = group_context.main_flow_artifacts
                        if native_images:
                            next_main_flow = ArtifactSpecCollection(
                                spec.for_plan_type(ArtifactInputPlan)
                                for spec in native_images
                            )
                        elif invocation.contract.processing_contract is not None:
                            next_main_flow = ArtifactSpecCollection(
                                (
                                    ArtifactSpec.input(
                                        unnamed_main_flow_artifact_name(
                                            snapshot.index,
                                            invocation.key,
                                        ),
                                        ImageArtifactType,
                                    ),
                                )
                            )
                        group_context = group_context.advance_artifact_graph(
                            native_graph,
                            main_flow_artifacts=next_main_flow,
                        )
                        continue
                    scope = invocation.contract.execution_scope
                    try:
                        invocation_blocks, consumed_kwarg_names = (
                            module_type.module_blocks_for_invocation(
                                invocation=invocation,
                                step_context=group_context,
                            )
                        )
                        if not invocation_blocks:
                            raise ValueError(
                                f"CellProfiler module {module_type.require_module_name()!r} "
                                "cannot reconstruct an exact module block from the "
                                "public invocation and scoped artifact context."
                            )
                        step_invocations.append((invocation, invocation_blocks))
                        numbered_invocations, next_module_num = (
                            CellProfilerModule.number_public_step_invocation_blocks(
                                tuple(step_invocations),
                                first_module_num=first_step_module_num,
                            )
                        )
                        numbered_module_blocks = numbered_invocations[-1]
                        compiled_contract, consumed_kwarg_names = (
                            module_type.invocation_callable_contract(
                                invocation=invocation,
                                numbered_module_blocks=numbered_module_blocks,
                                consumed_kwarg_names=consumed_kwarg_names,
                                step_context=group_context,
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise type(exc)(
                            "CellProfiler contract compilation failed for step "
                            f"{snapshot.index} ({snapshot.step.name!r}), invocation "
                            f"{invocation.key!r}, module {module_type.__name__}: {exc}"
                        ) from exc
                    compiled_contract = replace(
                        compiled_contract,
                        metadata=replace(
                            compiled_contract.metadata,
                            runtime_adapter=(
                                None
                                if (
                                    scope is FunctionStepExecutionScope.PLATE
                                    or not module_type.uses_cellprofiler_runtime_adapter()
                                )
                                else CellProfilerRuntimeAdapter.runtime_adapter_spec()
                            ),
                        ),
                    )
                    key = (snapshot.index, invocation.key)
                    if key in plans:
                        raise ValueError(
                            f"Duplicate CellProfiler invocation contract for step "
                            f"{snapshot.index} ({snapshot.step.name!r}), invocation "
                            f"{invocation.key!r}, module {module_type.__name__}: "
                            f"{key!r}."
                        )
                    plans[key] = InvocationContractPlan(
                        contract=compiled_contract,
                        consumed_kwarg_names=consumed_kwarg_names,
                    )
                    group_context = module_type.advance_artifact_context(
                        group_context,
                        contract=compiled_contract,
                        invocation_key=invocation.key,
                    )
                group_contexts.append(group_context)
            next_main_flow = ArtifactSpecCollection(
                ArtifactSpecCollection(
                    spec
                    for group_context in group_contexts
                    for spec in group_context.main_flow_artifacts.specs
                ).unique(conflict_context="function-pattern group main flow")
            )
            compiled_step_graph = extract_artifact_declarations(
                normalized_pattern,
                invocation_contract_provider=(
                    CellProfilerInvocationContractProvider(plans)
                    if plans
                    else CompositeInvocationContractProvider(())
                ),
                step_context=step_context,
            )
            forward_context = step_context.advance_artifact_graph(
                compiled_step_graph,
                main_flow_artifacts=next_main_flow,
            )
        return CellProfilerInvocationContractProvider(plans) if plans else None
