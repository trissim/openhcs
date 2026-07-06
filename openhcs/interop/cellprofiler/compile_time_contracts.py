"""Compiler-side contract derivation for generated CellProfiler steps."""

from __future__ import annotations

from typing import Any

from openhcs.core.artifacts import ImageArtifactType
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import (
    InvocationContractProviderFactory,
    InvocationContractProviderLike,
)
from openhcs.core.pipeline.compilation_session import PIPELINE_SOURCE_SCHEMA_METADATA_KEY
from openhcs.core.pipeline_image_schema import (
    PipelineImageSchema,
    SourceArtifactAssignment,
)
from openhcs.core.source_bindings import SourceBindingOrigin, StepSourceBindingsConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.parser import ModuleBlock


def cellprofiler_module_settings_invocation_contract_provider_for_session(
    session: Any,
) -> InvocationContractProviderLike | None:
    """Derive CellProfiler runtime contracts from compile-only module settings."""
    sidecar_provider = _generated_pipeline_contract_provider_for_session(session)
    if sidecar_provider is not None:
        return sidecar_provider

    module_items = _module_items_from_steps(session.steps)
    if not module_items:
        return None

    from openhcs.interop.cellprofiler.pipeline_generator import PipelineGenerator
    from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
        CellProfilerGeneratedInvocationContractProvider,
    )
    from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbolTable

    modules = [module for module, _kwargs, _step in module_items]
    symbol_table = CellProfilerSymbolTable.compile(
        modules,
        source_schema=_source_schema_for_session(session, module_items),
    )
    contracts_by_module = {
        module.module_num: symbol_table.contract_for(module)
        for module in modules
    }
    runtime_contracts = PipelineGenerator().runtime_contracts.by_module_num(
        modules,
        contracts_by_module,
    )
    snapshots = getattr(session, "snapshots", None)
    if snapshots is not None:
        return CellProfilerGeneratedInvocationContractProvider.for_snapshots(
            runtime_contracts,
            snapshots,
        )
    return CellProfilerGeneratedInvocationContractProvider.for_steps(
        runtime_contracts,
        session.steps,
    )


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


def _generated_pipeline_contract_provider_for_session(
    session: Any,
) -> InvocationContractProviderLike | None:
    from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
        CellProfilerGeneratedInvocationContractProvider,
        CellProfilerGeneratedPipelineInvocationContracts,
    )

    value = getattr(session, "pipeline_metadata", {}).get(
        CellProfilerGeneratedPipelineInvocationContracts.module_attribute
    )
    if value is None:
        return None
    contracts = CellProfilerGeneratedPipelineInvocationContracts.from_mapping(value)
    return CellProfilerGeneratedInvocationContractProvider.for_snapshots(
        contracts.contracts_by_module_num,
        session.snapshots,
    )


def _module_items_from_steps(
    steps: Any,
) -> tuple[tuple[ModuleBlock, dict, FunctionStep], ...]:
    modules: list[tuple[ModuleBlock, dict, FunctionStep]] = []
    module_num = 0
    for step in steps:
        if not isinstance(step, FunctionStep):
            continue
        function_spec = step.function_spec()
        if function_spec is None:
            continue
        for item in normalize_function_pattern(function_spec).iter_items():
            module = _module_from_item(
                item,
                module_num=module_num + 1,
                step=step,
            )
            if module is not None:
                module_num += 1
                modules.append((module, item.kwargs_dict, step))
    return tuple(modules)


def _module_from_item(
    item: Any,
    *,
    module_num: int,
    step: FunctionStep,
) -> ModuleBlock | None:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

    metadata = CellProfilerFunctionCatalog.runtime_metadata(
        item.contract.resolve_runtime_callable()
    )
    if metadata is None:
        return None
    module_type = CellProfilerModule.for_module(metadata.module_name)
    records = list(
        module_type.compile_time_setting_records_from_kwargs(item.kwargs_dict)
        if module_type is not None
        else ()
    )
    if module_type is not None:
        records.extend(
            module_type.compile_time_public_setting_records_from_kwargs(
                item.kwargs_dict
            )
        )
        records.extend(
            _source_binding_input_setting_records(
                module_type=module_type,
                step=step,
                existing_records=tuple(records),
            )
        )
    settings = {record.name: record.value for record in records}
    return ModuleBlock(
        name=metadata.module_name,
        module_num=module_num,
        enabled=bool(step.enabled),
        settings=settings,
        setting_records=list(records),
    )


def _source_binding_input_setting_records(
    *,
    module_type: type,
    step: FunctionStep,
    existing_records: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Infer one missing declared input setting from one direct source binding."""
    from openhcs.interop.cellprofiler.parser import ModuleSetting
    from openhcs.interop.cellprofiler.setting_names import setting_names

    source_bindings = getattr(step, "source_bindings", None)
    if not isinstance(source_bindings, StepSourceBindingsConfig):
        return ()
    if not source_bindings.enabled:
        return ()

    existing_setting_names = {record.name for record in existing_records}
    missing_settings = []
    for setting, capability_type in module_type.declared_artifact_input_settings():
        concrete_names = setting_names(setting)
        if any(name in existing_setting_names for name in concrete_names):
            continue
        missing_settings.append(
            (concrete_names[0], capability_type.require_artifact_type())
        )

    if len(missing_settings) != 1:
        return ()

    setting_name, artifact_type = missing_settings[0]
    candidate_bindings = tuple(
        binding
        for binding in source_bindings.binding_declarations
        if binding.artifact_kind is artifact_type
    )
    if len(candidate_bindings) != 1:
        return ()
    return (ModuleSetting(setting_name, candidate_bindings[0].alias),)


def _source_schema_for_session(
    session: Any,
    module_items: tuple[tuple[ModuleBlock, dict, FunctionStep], ...],
) -> PipelineImageSchema:
    metadata_schema = getattr(session, "pipeline_metadata", {}).get(
        PIPELINE_SOURCE_SCHEMA_METADATA_KEY
    )
    if isinstance(metadata_schema, PipelineImageSchema):
        return metadata_schema

    source_artifacts: dict[str, SourceArtifactAssignment] = {}
    for _module, _kwargs, step in module_items:
        source_bindings = getattr(step, "source_bindings", None)
        if not isinstance(source_bindings, StepSourceBindingsConfig):
            continue
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
