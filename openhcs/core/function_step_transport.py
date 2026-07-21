"""Transport normalization for FunctionStep declarations."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from collections.abc import Mapping, Sequence
from typing import Any

from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
)
from openhcs.core.function_reference import (
    FunctionReference,
)
from openhcs.core.pipeline_document_fields import PipelineDocumentField
from openhcs.core.steps.function_step import FunctionStep


class FunctionStepTransportAuthority:
    """Canonicalize FunctionStep callables before process or ZMQ transport."""

    @classmethod
    def source_from_pipeline(
        cls,
        pipeline_steps: Sequence[FunctionStep],
        *,
        clean_mode: bool = True,
    ) -> str:
        """Normalize and pycodify one public FunctionStep list."""
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment
        from openhcs.serialization.source_path_factoring import (
            OpenHCSPythonSourceDocument,
        )

        steps = list(pipeline_steps)
        cls._require_function_steps(steps)
        normalized = cls.normalize_pipeline(steps)
        return OpenHCSPythonSourceDocument(
            Assignment(PipelineDocumentField.PIPELINE_STEPS.value, normalized),
            header="# Edit this pipeline and save to apply changes",
            clean_mode=clean_mode,
        ).render()

    @classmethod
    def pipeline_steps_from_namespace(
        cls,
        namespace: Mapping[str, object],
    ) -> list[FunctionStep]:
        """Require and normalize the public FunctionStep export."""
        field_name = PipelineDocumentField.PIPELINE_STEPS.value
        if field_name not in namespace:
            raise ValueError(f"Pipeline source must define {field_name!r}.")
        pipeline_steps = namespace[field_name]
        if type(pipeline_steps) is not list:
            raise TypeError(
                f"{field_name} must be a list[FunctionStep], "
                f"got {type(pipeline_steps).__name__}."
            )
        cls._require_function_steps(pipeline_steps)
        return cls.normalize_pipeline(pipeline_steps)

    @classmethod
    def _require_function_steps(cls, pipeline_steps: Sequence[object]) -> None:
        for index, step in enumerate(pipeline_steps):
            if not isinstance(step, FunctionStep):
                raise TypeError(
                    f"Pipeline member {index} must be FunctionStep, "
                    f"got {type(step).__name__}."
                )

    @classmethod
    def normalize_pipeline(cls, definition_pipeline: list[Any]) -> list[Any]:
        normalized = [cls.normalize_step(step) for step in definition_pipeline]
        return normalized

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
            normalized_items = [cls.normalize_function_spec(item) for item in func_spec]
            if all(
                normalized is original
                for normalized, original in zip(normalized_items, func_spec)
            ):
                return func_spec
            return normalized_items
        if isinstance(func_spec, dict):
            normalized_items = {
                key: cls.normalize_function_spec(item)
                for key, item in func_spec.items()
            }
            if all(normalized_items[key] is item for key, item in func_spec.items()):
                return func_spec
            return normalized_items
        if isinstance(func_spec, tuple) and func_spec:
            normalized_func = cls.normalize_function_spec(func_spec[0])
            if normalized_func is func_spec[0]:
                return func_spec
            return (normalized_func, *func_spec[1:])
        if isinstance(func_spec, ModuleType):
            raise TypeError(
                "Pipeline contains a module object where a callable is required: "
                f"{func_spec.__name__}. Reload or edit the step to select a function."
            )
        if callable(func_spec):
            from openhcs.processing.backends.lib_registry.registry_service import (
                RegistryService,
            )

            return RegistryService.registered_callable(func_spec)
        return func_spec

    @classmethod
    def normalize_function_reference(
        cls,
        reference: FunctionReference,
    ) -> FunctionReference:
        raw_processing_function = reference.metadata.raw_processing_function
        if raw_processing_function is None:
            normalized_raw = None
        else:
            normalized_raw = cls.normalize_function_spec(raw_processing_function)
        raw_is_normalized = normalized_raw is raw_processing_function
        if reference.metadata.prepare is None and raw_is_normalized:
            return reference
        metadata = reference.metadata.without_prepare()
        if not raw_is_normalized:
            metadata = metadata.with_raw_processing_function(normalized_raw)
        return replace(reference, metadata=metadata)

    @classmethod
    def normalize_compiled_pattern(
        cls,
        pattern: CompiledFunctionPattern | None,
    ) -> CompiledFunctionPattern | None:
        if pattern is None:
            return None
        groups = tuple(cls.normalize_compiled_group(group) for group in pattern.groups)
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
        metadata = contract.metadata
        if normalized_raw is not contract.raw_processing_function:
            metadata = metadata.with_raw_processing_function(normalized_raw)
        return replace(
            contract,
            func=normalized_func,
            metadata=metadata,
        )
