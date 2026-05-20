"""Transport normalization for FunctionStep declarations."""

from __future__ import annotations

from dataclasses import replace
from types import FunctionType, ModuleType
from typing import Any, Callable, Mapping

from openhcs.core.callable_contract import (
    CallableContract,
    PROCESSING_PREPARE_ATTR,
    RAW_PROCESSING_FUNCTION_ATTR,
)
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
)
from openhcs.core.pipeline.compiler import FunctionReference
from openhcs.core.steps.function_step import FunctionStep


class FunctionStepTransportAuthority:
    """Canonicalize FunctionStep callables before process or ZMQ transport."""

    @classmethod
    def normalize_pipeline(cls, definition_pipeline: list[Any]) -> list[Any]:
        return [cls.normalize_step(step) for step in definition_pipeline]

    @classmethod
    def normalize_contexts(cls, contexts: Mapping[str, Any]) -> dict[str, Any]:
        return {
            axis_id: cls.normalize_context(context)
            for axis_id, context in contexts.items()
        }

    @classmethod
    def normalize_context(cls, context: Any) -> Any:
        step_plans = context.step_plans
        for step_plan in step_plans.values():
            step_plan.func = cls.normalize_function_spec(step_plan.func)
            step_plan.compiled_function_pattern = cls.normalize_compiled_pattern(
                step_plan.compiled_function_pattern
            )
        return context

    @classmethod
    def normalize_step(cls, step: Any) -> Any:
        if not isinstance(step, FunctionStep):
            return step
        func_spec = step.function_spec()
        if func_spec is None:
            return step
        normalized_func = cls.normalize_function_spec(func_spec)
        if normalized_func is func_spec:
            return step
        return step.with_function_spec(normalized_func)

    @classmethod
    def normalize_function_spec(cls, func_spec: Any) -> Any:
        if isinstance(func_spec, FunctionReference):
            return cls.normalize_function_reference(func_spec)
        if isinstance(func_spec, list):
            normalized_items = [
                cls.normalize_function_spec(item)
                for item in func_spec
            ]
            if all(
                normalized is original
                for normalized, original in zip(normalized_items, func_spec)
            ):
                return func_spec
            return normalized_items
        if isinstance(func_spec, tuple) and func_spec:
            normalized_func = cls.normalize_function_spec(func_spec[0])
            if normalized_func is func_spec[0]:
                return func_spec
            return (normalized_func, *func_spec[1:])
        if isinstance(func_spec, ModuleType):
            resolved = cls.resolve_module_callable(func_spec)
            if resolved is not None:
                return resolved
            raise TypeError(
                "Pipeline contains a module object where a callable is required: "
                f"{func_spec.__name__}. Reload or edit the step to select a function."
            )
        if callable(func_spec):
            resolved = cls.resolve_callable(func_spec)
            if resolved is not None:
                return resolved
        return func_spec

    @classmethod
    def normalize_function_reference(
        cls,
        reference: FunctionReference,
    ) -> FunctionReference:
        preserved_attrs = reference.preserved_attrs
        normalized_raw = (
            cls.normalize_function_spec(preserved_attrs[RAW_PROCESSING_FUNCTION_ATTR])
            if RAW_PROCESSING_FUNCTION_ATTR in preserved_attrs
            else None
        )
        raw_is_normalized = (
            RAW_PROCESSING_FUNCTION_ATTR not in preserved_attrs
            or normalized_raw is preserved_attrs[RAW_PROCESSING_FUNCTION_ATTR]
        )
        if (
            PROCESSING_PREPARE_ATTR not in preserved_attrs
            and raw_is_normalized
        ):
            return reference
        preserved_attrs = dict(reference.preserved_attrs)
        preserved_attrs.pop(PROCESSING_PREPARE_ATTR, None)
        if normalized_raw is not None:
            preserved_attrs[RAW_PROCESSING_FUNCTION_ATTR] = normalized_raw
        return replace(reference, preserved_attrs=preserved_attrs)

    @classmethod
    def normalize_compiled_pattern(
        cls,
        pattern: CompiledFunctionPattern | None,
    ) -> CompiledFunctionPattern | None:
        if pattern is None:
            return None
        groups = tuple(
            cls.normalize_compiled_group(group)
            for group in pattern.groups
        )
        if all(
            normalized is original
            for normalized, original in zip(groups, pattern.groups)
        ):
            return pattern
        return replace(pattern, groups=groups)

    @classmethod
    def normalize_compiled_group(
        cls,
        group: CompiledFunctionGroup,
    ) -> CompiledFunctionGroup:
        invocations = tuple(
            cls.normalize_compiled_invocation(invocation)
            for invocation in group.invocations
        )
        if all(
            normalized is original
            for normalized, original in zip(invocations, group.invocations)
        ):
            return group
        return replace(group, invocations=invocations)

    @classmethod
    def normalize_compiled_invocation(
        cls,
        invocation: CompiledFunctionInvocation,
    ) -> CompiledFunctionInvocation:
        contract = cls.normalize_callable_contract(invocation.contract)
        if contract is invocation.contract:
            return invocation
        return replace(invocation, contract=contract)

    @classmethod
    def normalize_callable_contract(
        cls,
        contract: CallableContract,
    ) -> CallableContract:
        normalized_func = cls.normalize_function_spec(contract.func)
        normalized_raw = (
            cls.normalize_function_spec(contract.raw_processing_function)
            if contract.raw_processing_function is not None
            else None
        )
        if (
            normalized_func is contract.func
            and normalized_raw is contract.raw_processing_function
        ):
            return contract
        return replace(
            contract,
            func=normalized_func,
            raw_processing_function=normalized_raw,
        )

    @classmethod
    def resolve_module_callable(cls, module: ModuleType) -> Callable | None:
        module_name = module.__name__
        cellprofiler_prefix = "openhcs.processing.backends.cellprofiler."
        if not module_name.startswith(cellprofiler_prefix):
            return None
        function_name = module_name.removeprefix(cellprofiler_prefix)
        from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

        try:
            return CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None

    @classmethod
    def resolve_callable(cls, func: Callable) -> Callable | None:
        if not isinstance(func, FunctionType):
            return None
        cellprofiler_module = "openhcs.processing.backends.cellprofiler"
        if (
            func.__module__ != cellprofiler_module
            and not func.__module__.startswith(f"{cellprofiler_module}.")
        ):
            return None
        from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

        try:
            return CellProfilerFunctionCatalog.get_function(func.__name__)
        except KeyError:
            return None
