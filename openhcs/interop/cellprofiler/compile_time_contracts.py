"""Compiler-side contract derivation for generated CellProfiler steps."""

from __future__ import annotations

from typing import Any

from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import InvocationContractProviderLike
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.module_settings_payload import (
    CellProfilerModuleSettingsKwarg,
    CellProfilerModuleSettingsPayload,
)


def cellprofiler_module_settings_invocation_contract_provider_for_session(
    session: Any,
) -> InvocationContractProviderLike | None:
    """Derive CellProfiler runtime contracts from compile-only module settings."""
    payloads = _payloads_from_steps(session.steps)
    if not payloads:
        return None

    from openhcs.interop.cellprofiler.pipeline_generator import PipelineGenerator
    from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
        CellProfilerGeneratedInvocationContractProvider,
    )
    from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbolTable

    modules = [payload.module_block() for payload in payloads]
    symbol_table = CellProfilerSymbolTable.compile(modules)
    contracts_by_module = {
        module.module_num: symbol_table.contract_for(module)
        for module in modules
    }
    runtime_contracts = PipelineGenerator().runtime_contracts.by_module_num(
        modules,
        contracts_by_module,
    )
    return CellProfilerGeneratedInvocationContractProvider(runtime_contracts)


def _payloads_from_steps(
    steps: Any,
) -> tuple[CellProfilerModuleSettingsPayload, ...]:
    payloads: list[CellProfilerModuleSettingsPayload] = []
    for step in steps:
        if not isinstance(step, FunctionStep):
            continue
        function_spec = step.function_spec()
        if function_spec is None:
            continue
        for item in normalize_function_pattern(function_spec).iter_items():
            payload = CellProfilerModuleSettingsKwarg.payload_from_kwargs(
                item.kwargs_dict
            )
            if payload is not None:
                payloads.append(payload)
    return tuple(payloads)
