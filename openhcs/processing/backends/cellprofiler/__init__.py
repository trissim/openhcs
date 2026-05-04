"""CellProfiler-compatible processing functions for OpenHCS.

This backend exposes the absorbed, independently implemented CellProfiler
semantics through the normal OpenHCS processing registry.  It deliberately does
not import or execute the local CellProfiler source tree.
"""

from __future__ import annotations

import inspect
import importlib
from collections.abc import Callable, Mapping
from functools import lru_cache, wraps
from typing import Any

from benchmark.cellprofiler_library import (
    function_inventory,
    get_contract,
    list_modules,
)
from openhcs.core.callable_contract import (
    CallableContract,
    attach_callable_contract_metadata,
)
from openhcs.core.callable_contract import (
    DECLARED_PROCESSING_CONTRACT_ATTR,
    PROCESSING_CONTRACT_ATTR,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


CELLPROFILER_MODULE_ATTR = "__openhcs_cellprofiler_module__"


class _LazyCellProfilerFunctionMapping(Mapping[str, Any]):
    """Mapping facade that loads absorbed functions only on first access."""

    def __init__(self, index: int) -> None:
        self._index = index

    @property
    def _mapping(self) -> Mapping[str, Any]:
        return _cellprofiler_function_maps()[self._index]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)


def list_cellprofiler_functions() -> tuple[str, ...]:
    """Return exported CellProfiler-compatible function names."""
    return tuple(CELLPROFILER_FUNCTIONS)


def get_cellprofiler_function(name: str) -> Callable[..., Any]:
    """Return one exported CellProfiler-compatible processing function."""
    return CELLPROFILER_FUNCTIONS[name]


def require_cellprofiler_function(
    module_name: str,
    *,
    function_name: str | None = None,
) -> Callable[..., Any]:
    """Return one OpenHCS-owned CellProfiler-compatible function."""
    contract_payload = get_contract(module_name)
    if contract_payload is None:
        raise KeyError(
            f"No CellProfiler-compatible processing module registered: {module_name!r}"
        )
    resolved_function_name = function_name or str(contract_payload["function_name"])
    try:
        return CELLPROFILER_FUNCTIONS[resolved_function_name]
    except KeyError as exc:
        raise KeyError(
            f"CellProfiler-compatible processing module {module_name!r} declares "
            f"missing function {resolved_function_name!r}."
        ) from exc


def unavailable_cellprofiler_functions() -> Mapping[str, str]:
    """Return absorbed modules that were skipped during backend loading."""
    return UNAVAILABLE_CELLPROFILER_FUNCTIONS


def _declared_processing_contract(
    module_name: str,
    absorbed_function: Callable[..., Any],
) -> ProcessingContract:
    contract_payload = get_contract(module_name) or {}
    declared_name = str(contract_payload.get("contract", "flexible"))
    declared_contract = ProcessingContract.from_declared_name(declared_name)
    if declared_contract is not None:
        return declared_contract
    contract = CallableContract.from_callable(absorbed_function).processing_contract
    if isinstance(contract, ProcessingContract):
        return contract
    if isinstance(contract, str):
        resolved_contract = ProcessingContract.from_declared_name(contract)
        if resolved_contract is not None:
            return resolved_contract
    return ProcessingContract.FLEXIBLE


def _make_processing_wrapper(
    *,
    module_name: str,
    func: Callable[..., Any],
    contract: ProcessingContract,
) -> Callable[..., Any]:
    """Build an OpenHCS-owned wrapper around one absorbed implementation."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__name__
    wrapper.__module__ = __name__
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__annotations__ = getattr(func, "__annotations__", {}).copy()
    callable_contract = CallableContract.from_callable(func)
    wrapper.input_memory_type = callable_contract.input_memory_type
    wrapper.output_memory_type = callable_contract.output_memory_type
    setattr(wrapper, PROCESSING_CONTRACT_ATTR, contract)
    setattr(wrapper, DECLARED_PROCESSING_CONTRACT_ATTR, contract.name)
    setattr(wrapper, CELLPROFILER_MODULE_ATTR, module_name)
    attach_callable_contract_metadata(wrapper, raw_processing_function=func)
    return wrapper


def _default_module_names_by_module_stem() -> dict[str, str]:
    default_modules: dict[str, str] = {}
    inventory = function_inventory()
    for module_name in list_modules():
        contract_payload = get_contract(module_name)
        if contract_payload is None:
            continue
        location = inventory.get(str(contract_payload["function_name"]))
        if location is None:
            continue
        default_modules.setdefault(location.module_stem, module_name)
    return default_modules


def _function_from_inventory(function_name: str) -> Callable[..., Any]:
    location = function_inventory()[function_name]
    module = importlib.import_module(location.module_name)
    function = vars(module).get(function_name)
    if not callable(function):
        raise KeyError(
            f"Absorbed CellProfiler function {function_name!r} is missing from "
            f"{location.module_name!r}."
        )
    return function


def _load_cellprofiler_functions() -> tuple[dict[str, Callable[..., Any]], dict[str, str]]:
    functions: dict[str, Callable[..., Any]] = {}
    default_modules = _default_module_names_by_module_stem()
    for function_name, location in function_inventory().items():
        module_name = default_modules.get(location.module_stem, location.module_stem)
        absorbed_function = _function_from_inventory(function_name)

        contract = _declared_processing_contract(module_name, absorbed_function)
        wrapped_function = _make_processing_wrapper(
            module_name=module_name,
            func=absorbed_function,
            contract=contract,
        )
        functions[wrapped_function.__name__] = wrapped_function
    return functions, {}


@lru_cache(maxsize=1)
def _cellprofiler_function_maps() -> tuple[
    Mapping[str, Callable[..., Any]],
    Mapping[str, str],
]:
    functions, unavailable = _load_cellprofiler_functions()
    return functions, unavailable


CELLPROFILER_FUNCTIONS: Mapping[str, Callable[..., Any]] = (
    _LazyCellProfilerFunctionMapping(0)
)
UNAVAILABLE_CELLPROFILER_FUNCTIONS: Mapping[str, str] = (
    _LazyCellProfilerFunctionMapping(1)
)


def __getattr__(name: str) -> Any:
    if name in CELLPROFILER_FUNCTIONS:
        return CELLPROFILER_FUNCTIONS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        {
            *globals(),
            *function_inventory(),
        }
    )

__all__ = tuple(
    sorted(
        (
            "CELLPROFILER_FUNCTIONS",
            "UNAVAILABLE_CELLPROFILER_FUNCTIONS",
            "get_cellprofiler_function",
            "list_cellprofiler_functions",
            "require_cellprofiler_function",
            "unavailable_cellprofiler_functions",
            *function_inventory(),
        )
    )
)
