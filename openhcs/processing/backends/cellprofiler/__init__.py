"""CellProfiler-compatible processing functions for OpenHCS.

This backend exposes the absorbed, independently implemented CellProfiler
semantics through the normal OpenHCS processing registry.  It deliberately does
not import or execute the local CellProfiler source tree.
"""

from __future__ import annotations

import inspect
import importlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, wraps
from types import ModuleType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
from python_introspect import parameter_exclusions, set_parameter_exclusions

from openhcs.processing.backends.cellprofiler.library import (
    coerce_absorbed_processing_contract,
    function_inventory,
    get_contract,
    list_modules,
)
from openhcs.processing.backends.cellprofiler.function_documentation import (
    enrich_cellprofiler_function_documentation,
)
from openhcs.core.callable_contract import (
    CallableContract,
    attach_callable_contract_metadata,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


CELLPROFILER_MODULE_ATTR = "__openhcs_cellprofiler_module__"
CELLPROFILER_FUNCTION_NAMES = frozenset(function_inventory())
_STABLE_CELLPROFILER_FUNCTIONS: dict[str, Callable[..., Any]] = {}


class CellProfilerBackendModule(ModuleType):
    """Package attribute authority for CellProfiler backend function exports."""

    def __getattribute__(self, name: str) -> Any:
        module_dict = ModuleType.__getattribute__(self, "__dict__")
        function_names = module_dict.get("CELLPROFILER_FUNCTION_NAMES", frozenset())
        if name in function_names:
            catalog = module_dict["CellProfilerFunctionCatalog"]
            return catalog.get_function(name)
        return ModuleType.__getattribute__(self, name)


@dataclass(frozen=True, slots=True)
class CellProfilerFunctionRuntimeMetadata:
    """Runtime identity attached to an absorbed CellProfiler backend callable."""

    module_name: str
    function_name: str
    processing_contract: ProcessingContract
    declared_processing_contract: str | None

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
    ) -> "CellProfilerFunctionRuntimeMetadata | None":
        """Project OpenHCS-owned CellProfiler runtime metadata from a callable."""
        if not callable(func):
            raise TypeError(
                "cellprofiler_function_runtime_metadata requires a callable, "
                f"got {type(func).__name__}."
            )
        try:
            metadata = func.__dict__
        except AttributeError:
            metadata = {}
        module_name = metadata.get(CELLPROFILER_MODULE_ATTR)
        if module_name is None:
            return None
        processing_contract = metadata.get(
            FunctionContractAttribute.processing_contract
        )
        if not isinstance(processing_contract, ProcessingContract):
            raise TypeError(
                f"CellProfiler function {func.__name__!r} has no declared "
                "ProcessingContract metadata."
            )
        return cls(
            module_name=str(module_name),
            function_name=func.__name__,
            processing_contract=processing_contract,
            declared_processing_contract=metadata.get(
                FunctionContractAttribute.declared_processing_contract,
            ),
        )


class CellProfilerFunctionCatalog:
    """Nominal authority for absorbed CellProfiler backend function lookup."""

    @classmethod
    def runtime_metadata(
        cls,
        func: Callable[..., Any],
    ) -> CellProfilerFunctionRuntimeMetadata | None:
        return CellProfilerFunctionRuntimeMetadata.from_callable(func)

    @classmethod
    def list_functions(cls) -> tuple[str, ...]:
        """Return exported CellProfiler-compatible function names."""
        return tuple(CELLPROFILER_FUNCTIONS)

    @classmethod
    def get_function(cls, name: str) -> Callable[..., Any]:
        """Return one exported CellProfiler-compatible processing function."""
        return _cellprofiler_function(name)

    @classmethod
    def require_function(
        cls,
        module_name: str,
        *,
        function_name: str | None = None,
    ) -> Callable[..., Any]:
        """Return one OpenHCS-owned CellProfiler-compatible function."""
        contract_payload = get_contract(module_name)
        if contract_payload is None:
            raise KeyError(
                "No CellProfiler-compatible processing module registered: "
                f"{module_name!r}"
            )
        resolved_function_name = function_name or str(contract_payload["function_name"])
        try:
            return cls.get_function(resolved_function_name)
        except KeyError as exc:
            raise KeyError(
                f"CellProfiler-compatible processing module {module_name!r} "
                f"declares missing function {resolved_function_name!r}."
            ) from exc

    @classmethod
    def unavailable_functions(cls) -> Mapping[str, str]:
        """Return absorbed modules that were skipped during backend loading."""
        return UNAVAILABLE_CELLPROFILER_FUNCTIONS


class CellProfilerCatalogCompatibilityExport(ABC, metaclass=AutoRegisterMeta):
    """Registered compatibility export for legacy module-level catalog helpers."""

    __registry_key__ = "export_name"
    __skip_if_no_key__ = True
    export_name: ClassVar[str | None] = None

    @classmethod
    @abstractmethod
    def value(cls) -> Callable[..., Any]:
        """Return the callable exposed for this compatibility name."""


class RuntimeMetadataCatalogCompatibilityExport(
    CellProfilerCatalogCompatibilityExport,
):
    export_name = "cellprofiler_function_runtime_metadata"

    @classmethod
    def value(cls) -> Callable[..., Any]:
        return CellProfilerFunctionCatalog.runtime_metadata


class GetFunctionCatalogCompatibilityExport(CellProfilerCatalogCompatibilityExport):
    export_name = "get_cellprofiler_function"

    @classmethod
    def value(cls) -> Callable[..., Any]:
        return CellProfilerFunctionCatalog.get_function


class ListFunctionsCatalogCompatibilityExport(CellProfilerCatalogCompatibilityExport):
    export_name = "list_cellprofiler_functions"

    @classmethod
    def value(cls) -> Callable[..., Any]:
        return CellProfilerFunctionCatalog.list_functions


class RequireFunctionCatalogCompatibilityExport(
    CellProfilerCatalogCompatibilityExport,
):
    export_name = "require_cellprofiler_function"

    @classmethod
    def value(cls) -> Callable[..., Any]:
        return CellProfilerFunctionCatalog.require_function


class UnavailableFunctionsCatalogCompatibilityExport(
    CellProfilerCatalogCompatibilityExport,
):
    export_name = "unavailable_cellprofiler_functions"

    @classmethod
    def value(cls) -> Callable[..., Any]:
        return CellProfilerFunctionCatalog.unavailable_functions


class _LazyCellProfilerFunctionMapping(Mapping[str, Any]):
    """Mapping facade that loads absorbed functions only on first access."""

    def __init__(self, index: int) -> None:
        self._index = index

    @property
    def _mapping(self) -> Mapping[str, Any]:
        return _cellprofiler_function_maps()[self._index]

    def __getitem__(self, key: str) -> Any:
        if self._index == 0:
            return _cellprofiler_function(key)
        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)


def _declared_processing_contract(
    module_name: str,
    function_name: str,
    absorbed_function: Callable[..., Any],
) -> ProcessingContract | None:
    contract = coerce_absorbed_processing_contract(
        module_name,
        function_name,
        absorbed_function,
    )
    if isinstance(contract, ProcessingContract):
        return contract
    return None


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
    wrapper.__annotations__ = inspect.get_annotations(func, eval_str=False).copy()
    callable_contract = CallableContract.from_callable(func)
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    module_type = CellProfilerModule.for_module(module_name)
    if module_type is None:
        raise KeyError(
            "CellProfiler-compatible processing function requires a "
            f"CellProfilerModule declaration for {module_name!r}."
        )
    wrapper.input_memory_type = callable_contract.input_memory_type
    wrapper.output_memory_type = callable_contract.output_memory_type
    setattr(wrapper, FunctionContractAttribute.processing_contract, contract)
    setattr(
        wrapper,
        FunctionContractAttribute.declared_processing_contract,
        contract.name,
    )
    setattr(wrapper, CELLPROFILER_MODULE_ATTR, module_name)
    setattr(
        wrapper,
        FunctionContractAttribute.allowed_group_by,
        module_type.allowed_group_by,
    )
    attach_callable_contract_metadata(
        wrapper,
        raw_processing_function=func,
        runtime_image_execution_mode=callable_contract.runtime_image_execution_mode,
    )
    enrich_cellprofiler_function_documentation(
        wrapper,
        module_name=module_name,
        source_function=func,
    )
    hidden_parameters = parameter_exclusions(func)
    if hidden_parameters:
        set_parameter_exclusions(wrapper, hidden_parameters)
    return wrapper


@lru_cache(maxsize=1)
def _default_module_names_by_function_name() -> dict[str, str]:
    """Map each declared function to the CellProfiler module that owns it."""
    default_modules: dict[str, str] = {}
    for module_name in list_modules():
        contract_payload = get_contract(module_name)
        if contract_payload is None:
            continue
        function_names = (
            str(contract_payload["function_name"]),
            *(str(name) for name in contract_payload.get("function_variants", ())),
        )
        for function_name in function_names:
            default_modules.setdefault(function_name, module_name)
    return default_modules


@lru_cache(maxsize=None)
def _cellprofiler_function(function_name: str) -> Callable[..., Any]:
    """Return one absorbed function wrapper without loading the full catalog."""
    if function_name not in function_inventory():
        raise KeyError(function_name)
    location = function_inventory()[function_name]
    module_name = _default_module_names_by_function_name().get(
        function_name,
        location.module_stem,
    )
    absorbed_function = _function_from_inventory(function_name)
    contract = _declared_processing_contract(
        module_name,
        function_name,
        absorbed_function,
    )
    if contract is None:
        raise KeyError(function_name)
    return _STABLE_CELLPROFILER_FUNCTIONS.setdefault(
        function_name,
        _make_processing_wrapper(
            module_name=module_name,
            func=absorbed_function,
            contract=contract,
        ),
    )


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
    for function_name in function_inventory():
        try:
            function = _cellprofiler_function(function_name)
        except KeyError:
            continue
        functions[function.__name__] = function
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
    catalog_export = CellProfilerCatalogCompatibilityExport.__registry__.get(name)
    if catalog_export is not None:
        return catalog_export.value()
    if name in CELLPROFILER_FUNCTIONS:
        return CELLPROFILER_FUNCTIONS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        {
            *globals(),
            *CELLPROFILER_FUNCTION_NAMES,
        }
    )

__all__ = tuple(
    sorted(
        (
            "CELLPROFILER_FUNCTIONS",
            "CellProfilerFunctionCatalog",
            "CellProfilerFunctionRuntimeMetadata",
            "UNAVAILABLE_CELLPROFILER_FUNCTIONS",
            "get_cellprofiler_function",
            "list_cellprofiler_functions",
            "require_cellprofiler_function",
            "unavailable_cellprofiler_functions",
            *CELLPROFILER_FUNCTION_NAMES,
        )
    )
)


sys.modules[__name__].__class__ = CellProfilerBackendModule
