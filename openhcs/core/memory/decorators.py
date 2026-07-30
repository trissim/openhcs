"""Canonical OpenHCS import surface for arraybridge memory decorators.

The underlying memory conversion behavior belongs to arraybridge. OpenHCS adds
compiler/runtime metadata preservation at this boundary so memory declarations
and callable preparation contracts compose predictably.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

import arraybridge as _arraybridge


def _with_openhcs_metadata(decorator: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an arraybridge decorator with optional OpenHCS prepare metadata."""

    def openhcs_decorator(
        *args: Any,
        prepare: Any = None,
        contract: Any = None,
        **kwargs: Any,
    ) -> Any:
        if contract is None:
            from openhcs.processing.backends.lib_registry.unified_registry import (
                ProcessingContract,
            )

            contract = ProcessingContract.FLEXIBLE
        declared_processing_contract = _declared_processing_contract_name(contract)
        if declared_processing_contract is None:
            kwargs["contract"] = contract

        if args and callable(args[0]) and len(args) == 1:
            wrapped = decorator(args[0], **kwargs)
            _attach_openhcs_metadata(
                wrapped,
                prepare=prepare,
                declared_processing_contract=declared_processing_contract,
            )
            return wrapped

        arraybridge_decorator = decorator(*args, **kwargs)

        def decorate(target: Any) -> Any:
            wrapped = arraybridge_decorator(target)
            _attach_openhcs_metadata(
                wrapped,
                prepare=prepare,
                declared_processing_contract=declared_processing_contract,
            )
            return wrapped

        return decorate

    openhcs_decorator.__name__ = getattr(decorator, "__name__", "openhcs_decorator")
    openhcs_decorator.__doc__ = getattr(decorator, "__doc__", None)
    openhcs_decorator.__module__ = __name__
    return openhcs_decorator


def _declared_processing_contract_name(contract: Any) -> str | None:
    """Return OpenHCS processing-contract metadata carried by a decorator."""
    if contract is None or callable(contract):
        return None
    name = getattr(contract, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(contract, str) and contract:
        return contract
    raise TypeError(
        "OpenHCS memory decorator contract must be a callable arraybridge "
        f"validator or a processing contract declaration; got {contract!r}."
    )


def _attach_openhcs_metadata(
    wrapped: Any,
    *,
    prepare: Any,
    declared_processing_contract: str | None,
) -> None:
    from openhcs.core.callable_contract import attach_callable_contract_metadata
    from openhcs.core.config import runtime_config_parameter
    from python_introspect import add_parameter_exclusions, parameter_exclusions

    signature = _signature_with_resolved_raw_annotations(wrapped)
    raw_callable = inspect.unwrap(wrapped)
    add_parameter_exclusions(
        wrapped,
        parameter_exclusions(raw_callable),
    )
    normalized_parameter_items: list[inspect.Parameter] = []
    runtime_config_parameter_names: list[str] = []
    for parameter in signature.parameters.values():
        normalized = runtime_config_parameter(parameter)
        if normalized is not None:
            runtime_config_parameter_names.append(parameter.name)
        normalized_parameter_items.append(
            parameter if normalized is None else normalized
        )
    normalized_parameters = tuple(normalized_parameter_items)
    if normalized_parameters != tuple(signature.parameters.values()):
        wrapped.__signature__ = signature.replace(parameters=normalized_parameters)
    if runtime_config_parameter_names:
        add_parameter_exclusions(wrapped, tuple(runtime_config_parameter_names))
    if prepare is not None:
        from openhcs.core.callable_contract import attach_processing_prepare

        attach_processing_prepare(wrapped, prepare)
    if declared_processing_contract is not None:
        _strip_unowned_semantic_controls(wrapped, declared_processing_contract)
        attach_callable_contract_metadata(
            wrapped,
            declared_processing_contract=declared_processing_contract,
        )


def _signature_with_resolved_raw_annotations(wrapped: Any) -> inspect.Signature:
    """Resolve postponed annotations at their authoritative callable globals."""

    signature = inspect.signature(wrapped)
    raw_signature = inspect.signature(inspect.unwrap(wrapped), eval_str=True)
    raw_parameters = raw_signature.parameters
    parameters = tuple(
        parameter.replace(annotation=raw_parameters[parameter.name].annotation)
        if parameter.name in raw_parameters
        else parameter
        for parameter in signature.parameters.values()
    )
    return signature.replace(
        parameters=parameters,
        return_annotation=raw_signature.return_annotation,
    )


def _strip_unowned_semantic_controls(
    wrapped: Any,
    declared_processing_contract: str,
) -> None:
    from openhcs.processing.backends.lib_registry.unified_registry import (
        ContractRuntimeParameter,
        ProcessingContract,
    )

    contract = ProcessingContract.from_declared_name(declared_processing_contract)
    if contract is None:
        return
    allowed_semantic_control_names = (
        contract.declaration.injected_semantic_control_parameter_names()
    )
    semantic_control_names = {
        parameter_type.require_parameter_name()
        for parameter_type in ContractRuntimeParameter.registered_parameter_types()
        if parameter_type.is_semantic_control
    }
    params_to_strip = semantic_control_names - allowed_semantic_control_names
    if not params_to_strip:
        return
    signature = inspect.signature(wrapped)
    filtered_parameters = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.name not in params_to_strip
    )
    if len(filtered_parameters) != len(signature.parameters):
        wrapped.__signature__ = signature.replace(parameters=filtered_parameters)


memory_types = _with_openhcs_metadata(_arraybridge.memory_types)
numpy = _with_openhcs_metadata(_arraybridge.numpy)
cupy = _with_openhcs_metadata(_arraybridge.cupy)
torch = _with_openhcs_metadata(_arraybridge.torch)
tensorflow = _with_openhcs_metadata(_arraybridge.tensorflow)
jax = _with_openhcs_metadata(_arraybridge.jax)
pyclesperanto = _with_openhcs_metadata(_arraybridge.pyclesperanto)

__all__ = [
    "memory_types",
    "numpy",
    "cupy",
    "torch",
    "tensorflow",
    "jax",
    "pyclesperanto",
]
