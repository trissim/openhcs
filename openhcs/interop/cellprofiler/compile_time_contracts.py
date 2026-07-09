"""Compiler-side contract derivation for generated CellProfiler steps."""

from __future__ import annotations

from typing import Any

from openhcs.core.artifacts import ImageArtifactType
from openhcs.constants.input_source import InputSource
from openhcs.core.function_patterns import (
    NormalizedFunctionItem,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    InvocationContractProviderFactory,
    InvocationContractProviderLike,
)
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
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerCompileTimeArtifactFlow,
)


def cellprofiler_module_settings_invocation_contract_provider_for_session(
    session: CompilationSession,
) -> InvocationContractProviderLike | None:
    """Derive CellProfiler runtime contracts from compile-only module settings."""
    if not isinstance(session, CompilationSession):
        raise TypeError(
            "CellProfiler compile-time contract provider requires "
            f"CompilationSession, got {type(session).__name__}."
    )
    source_bindings_config = _source_bindings_config_for_session(session)
    module_items = _module_items_from_session(session)
    if not module_items:
        return None

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
    return CellProfilerGeneratedInvocationContractProvider.for_snapshots(
        runtime_contracts,
        session.snapshots,
        source_bindings_config=source_bindings_config,
        step_source_bindings_config=(
            _step_source_bindings_config_for_session(session)
        ),
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
    artifact_flow = CellProfilerCompileTimeArtifactFlow.empty()
    for snapshot in session.snapshots:
        if not snapshot.is_function_step:
            continue
        function_spec = snapshot.func
        if function_spec is None:
            continue
        source_bindings = _effective_source_bindings_for_snapshot(snapshot, session)
        invocation_flow = (
            _artifact_flow_from_source_bindings(source_bindings, artifact_flow)
            if source_bindings.enabled
            else artifact_flow
        )
        for item in normalize_function_pattern(function_spec).iter_items():
            for module_group_key in _module_group_keys_for_item(
                item,
                snapshot=snapshot,
                source_bindings=source_bindings,
                artifact_flow=invocation_flow,
            ):
                module = _module_from_item(
                    item,
                    module_num=module_num + 1,
                    snapshot=snapshot,
                    source_bindings=source_bindings,
                    group_key=module_group_key,
                    artifact_flow=invocation_flow,
                )
                if module is not None:
                    module_num += 1
                    modules.append((module, item.kwargs_dict, snapshot, source_bindings))
                    artifact_flow = _update_artifact_flow(
                        artifact_flow,
                        module_group_key,
                        module,
                    )
    return tuple(modules)


def _artifact_flow_from_source_bindings(
    source_bindings: StepSourceBindingsConfig,
    base_flow: CellProfilerCompileTimeArtifactFlow | None = None,
) -> CellProfilerCompileTimeArtifactFlow:
    """Return compiler artifact flow represented by effective source bindings."""
    source_names_by_group: dict[str, list[str]] = {}
    if not source_bindings.enabled:
        return base_flow or CellProfilerCompileTimeArtifactFlow.empty()
    for binding in source_bindings.binding_declarations:
        if binding.artifact_kind is not ImageArtifactType:
            continue
        group_keys = tuple(
            str(selector.value) for selector in binding.component_identity
        )
        if not group_keys:
            source_names_by_group.setdefault("default", []).append(binding.alias)
            continue
        for group_key in group_keys:
            source_names_by_group.setdefault(group_key, []).append(binding.alias)
    flow = base_flow or CellProfilerCompileTimeArtifactFlow.empty()
    for group_key, image_names in source_names_by_group.items():
        flow = flow.with_image_names(group_key, tuple(image_names))
    return flow


def _module_group_keys_for_item(
    item: NormalizedFunctionItem,
    *,
    snapshot: StepSnapshot,
    source_bindings: StepSourceBindingsConfig,
    artifact_flow: CellProfilerCompileTimeArtifactFlow,
) -> tuple[str, ...]:
    """Return transient CP module groups represented by one public invocation."""
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
        return (item.key.group_key,)
    metadata = CellProfilerFunctionCatalog.runtime_metadata(raw_callable)
    if metadata is None:
        return (item.key.group_key,)
    module_type = CellProfilerModule.for_module(metadata.module_name)
    if module_type is None:
        return (item.key.group_key,)
    group_keys = module_type.compile_time_source_binding_group_keys_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name=metadata.module_name,
            module_num=0,
            kwargs=item.kwargs_dict,
            invocation_options=item.invocation_options,
            source_bindings=source_bindings,
            group_key=item.key.group_key,
            artifact_flow=artifact_flow,
        )
    )
    return group_keys or (item.key.group_key,)


def _module_from_item(
    item: NormalizedFunctionItem,
    *,
    module_num: int,
    snapshot: StepSnapshot,
    source_bindings: StepSourceBindingsConfig,
    group_key: str,
    artifact_flow: CellProfilerCompileTimeArtifactFlow,
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
    module_metadata = {}
    if module_type is not None:
        request = CellProfilerCompileTimeSettingsRequest(
            module_name=metadata.module_name,
            module_num=module_num,
            kwargs=item.kwargs_dict,
            invocation_options=item.invocation_options,
            source_bindings=source_bindings,
            group_key=group_key,
            artifact_flow=artifact_flow,
        )
        records.extend(
            module_type.compile_time_setting_records_for_invocation(request)
        )
        module_metadata.update(
            module_type.compile_time_module_metadata_for_invocation(request)
        )
    settings = {record.name: record.value for record in records}
    return ModuleBlock(
        name=metadata.module_name,
        module_num=module_num,
        enabled=bool(snapshot.enabled),
        settings=settings,
        setting_records=list(records),
        metadata=module_metadata,
    )


def _update_artifact_flow(
    artifact_flow: CellProfilerCompileTimeArtifactFlow,
    group_key: str,
    module: ModuleBlock,
) -> CellProfilerCompileTimeArtifactFlow:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_module(module.name)
    if module_type is None:
        return artifact_flow
    return module_type.compile_time_artifact_flow_after_invocation(
        artifact_flow,
        group_key=group_key,
        module=module,
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
