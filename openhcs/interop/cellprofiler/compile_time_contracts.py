"""Compiler-side contract derivation for generated CellProfiler steps."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from openhcs.core.artifacts import ImageArtifactType
from openhcs.constants.input_source import InputSource
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import (
    CompositeInvocationContractProvider,
    InvocationContractProviderFactory,
    InvocationContractProviderLike,
)
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.pipeline_image_schema import (
    PipelineImageSchema,
    SourceArtifactAssignment,
)
from openhcs.core.source_bindings import (
    SourceBindingOrigin,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
    resolve_effective_step_source_bindings,
    source_bindings_defaults_to_base,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock


def cellprofiler_module_settings_invocation_contract_provider_for_session(
    session: CompilationSession,
) -> InvocationContractProviderLike | None:
    """Derive CellProfiler runtime contracts from compile-only module settings."""
    if not isinstance(session, CompilationSession):
        raise TypeError(
            "CellProfiler compile-time contract provider requires "
            f"CompilationSession, got {type(session).__name__}."
    )
    providers: list[InvocationContractProviderLike] = []
    source_bindings_config = _source_bindings_config_for_session(session)
    runtime_contracts = _runtime_contracts_from_selected_cppipe(session)
    if runtime_contracts:
        from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
            CellProfilerGeneratedInvocationContractProvider,
        )

        providers.append(
            CellProfilerGeneratedInvocationContractProvider.for_snapshots(
                runtime_contracts,
                session.snapshots,
                source_bindings_config=source_bindings_config,
                step_source_bindings_config=_step_source_bindings_config_for_session(
                    session
                ),
            )
        )
    else:
        module_items = _module_items_from_session(session)
        if module_items:
            from openhcs.interop.cellprofiler.pipeline_generator import (
                PipelineGenerator,
            )
            from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
                CellProfilerGeneratedInvocationContractProvider,
            )
            from openhcs.interop.cellprofiler.symbol_table import (
                CellProfilerSymbolTable,
            )

            modules = [module for module, _kwargs, _step, _bindings in module_items]
            symbol_table = CellProfilerSymbolTable.compile(
                modules,
                source_schema=_source_schema_for_session(module_items),
            )
            contracts_by_module = {
                module.module_num: symbol_table.contract_for(module)
                for module in modules
            }
            runtime_contracts = PipelineGenerator().runtime_contracts.by_module_num(
                modules,
                contracts_by_module,
            )
            providers.append(
                CellProfilerGeneratedInvocationContractProvider.for_snapshots(
                    runtime_contracts,
                    session.snapshots,
                    source_bindings_config=source_bindings_config,
                    step_source_bindings_config=(
                        _step_source_bindings_config_for_session(session)
                    ),
                )
            )
    step_contract_provider = _step_invocation_contract_provider_for_session(session)
    if step_contract_provider is not None:
        providers.append(step_contract_provider)
    if not providers:
        return None
    if len(providers) == 1:
        return providers[0]
    return CompositeInvocationContractProvider(tuple(providers))


def _step_invocation_contract_provider_for_session(
    session: CompilationSession,
) -> InvocationContractProviderLike | None:
    from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
        CellProfilerStepInvocationContractProvider,
    )

    return CellProfilerStepInvocationContractProvider.for_snapshots(
        session.snapshots,
        source_bindings_config=_source_bindings_config_for_session(session),
        step_source_bindings_config=_step_source_bindings_config_for_session(session),
    )


def _source_bindings_config_for_session(
    session: CompilationSession,
) -> SourceBindingsConfig:
    """Return pipeline source-binding defaults visible to compile-time providers."""
    return source_bindings_defaults_to_base(
        session.orchestrator.pipeline_config.source_bindings_config
    )


def _step_source_bindings_config_for_session(
    session: CompilationSession,
) -> StepSourceBindingsConfig:
    """Return pipeline step-source-binding defaults visible to providers."""
    return session.orchestrator.pipeline_config.step_source_bindings_config


class CellProfilerInvocationContractProviderFactory(InvocationContractProviderFactory):
    """Registered provider for generated CellProfiler invocation contracts."""

    @classmethod
    def provider_for_session(
        cls,
        session: Any,
    ) -> InvocationContractProviderLike | None:
        return cellprofiler_module_settings_invocation_contract_provider_for_session(
            session
        )


def _runtime_contracts_from_selected_cppipe(
    session: CompilationSession,
) -> Mapping[int, ModuleArtifactContract] | None:
    """Return regenerated CellProfiler runtime contracts for the selected .cppipe."""
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.interop.cellprofiler.runtime_pipeline import PreparedGeneratedPipeline

    orchestrator = session.orchestrator
    if not isinstance(orchestrator, PipelineOrchestrator):
        return None

    result = orchestrator.input_workspace_preparation_result
    if result is not None and isinstance(
        result.prepared_pipeline,
        PreparedGeneratedPipeline,
    ):
        return (
            result.prepared_pipeline
            .generated_pipeline
            .runtime_module_contracts_by_module_num
        )

    cppipe_path = orchestrator.selected_pipeline_path
    if cppipe_path is None:
        request = orchestrator.input_workspace_preparation
        if request is not None:
            cppipe_path = request.selected_pipeline_path
    if cppipe_path is None or cppipe_path.suffix != ".cppipe":
        return None
    return _runtime_contracts_from_cppipe_path(cppipe_path)


@lru_cache(maxsize=32)
def _runtime_contracts_from_cppipe_path(
    cppipe_path: Path,
) -> Mapping[int, ModuleArtifactContract]:
    """Regenerate runtime module contracts from a CellProfiler pipeline file."""
    from openhcs.interop.cellprofiler.runtime_pipeline import (
        CPPipePipelineGenerationRequest,
    )

    generated = CPPipePipelineGenerationRequest(cppipe_path=cppipe_path).generate()
    return generated.generated_pipeline.runtime_module_contracts_by_module_num


def _effective_source_bindings_for_snapshot(
    snapshot: StepSnapshot,
    session: CompilationSession,
) -> StepSourceBindingsConfig:
    """Return source bindings with pipeline defaults applied for compile-time CP use."""
    return resolve_effective_step_source_bindings(
        snapshot.source_bindings,
        source_bindings_defaults=_source_bindings_config_for_session(session),
        step_source_bindings_defaults=_step_source_bindings_config_for_session(session),
        activate_source_bindings=(
            snapshot.input_source == InputSource.PIPELINE_START
        ),
    )


def _module_items_from_session(
    session: CompilationSession,
) -> tuple[tuple[ModuleBlock, dict, StepSnapshot, StepSourceBindingsConfig], ...]:
    modules: list[tuple[ModuleBlock, dict, StepSnapshot, StepSourceBindingsConfig]] = []
    module_num = 0
    for snapshot in session.snapshots:
        if not snapshot.is_function_step:
            continue
        function_spec = snapshot.func
        if function_spec is None:
            continue
        source_bindings = _effective_source_bindings_for_snapshot(snapshot, session)
        for item in normalize_function_pattern(function_spec).iter_items():
            if snapshot.invocation_contracts.contract_for(item.key) is not None:
                continue
            module = _module_from_item(
                item,
                module_num=module_num + 1,
                snapshot=snapshot,
                source_bindings=source_bindings,
            )
            if module is not None:
                module_num += 1
                modules.append((module, item.kwargs_dict, snapshot, source_bindings))
    return tuple(modules)


def _module_from_item(
    item: Any,
    *,
    module_num: int,
    snapshot: StepSnapshot,
    source_bindings: StepSourceBindingsConfig,
) -> ModuleBlock | None:
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerRuntimeCallable,
    )
    from openhcs.interop.cellprofiler.module_declarations import (
        CellProfilerCompileTimeSettingsRequest,
        CellProfilerModule,
    )
    from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

    raw_callable = item.contract.resolve_runtime_callable()
    if isinstance(raw_callable, CellProfilerRuntimeCallable):
        return None
    metadata = CellProfilerFunctionCatalog.runtime_metadata(raw_callable)
    if metadata is None:
        return None
    module_type = CellProfilerModule.for_module(metadata.module_name)
    records = []
    if module_type is not None:
        records.extend(
            module_type.compile_time_setting_records_for_invocation(
                CellProfilerCompileTimeSettingsRequest(
                    module_name=metadata.module_name,
                    module_num=module_num,
                    kwargs=item.kwargs_dict,
                    invocation_options=item.invocation_options,
                    source_bindings=source_bindings,
                    group_key=item.key.group_key,
                )
            )
        )
    settings = {record.name: record.value for record in records}
    return ModuleBlock(
        name=metadata.module_name,
        module_num=module_num,
        enabled=bool(snapshot.enabled),
        settings=settings,
        setting_records=list(records),
    )


def _source_schema_for_session(
    module_items: tuple[tuple[ModuleBlock, dict, StepSnapshot, StepSourceBindingsConfig], ...],
) -> PipelineImageSchema:
    source_artifacts: dict[str, SourceArtifactAssignment] = {}
    for _module, _kwargs, _snapshot, source_bindings in module_items:
        if not source_bindings.enabled:
            continue
        for binding in source_bindings.binding_declarations:
            if binding.origin is not SourceBindingOrigin.PIPELINE_START:
                continue
            if binding.artifact_kind is ImageArtifactType:
                continue
            source_artifacts[binding.alias] = SourceArtifactAssignment(
                alias=binding.alias,
                artifact_kind=binding.artifact_kind,
                selector=binding.selector,
                origin=binding.origin,
                component_identity=binding.component_identity,
            )
    if not source_artifacts:
        return PipelineImageSchema.empty()
    return PipelineImageSchema(source_artifacts_by_alias=source_artifacts)
