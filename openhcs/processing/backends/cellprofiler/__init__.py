"""CellProfiler-compatible processing functions for OpenHCS.

This backend exposes the absorbed, independently implemented CellProfiler
semantics through the normal OpenHCS processing registry.  It deliberately does
not import or execute the local CellProfiler source tree.
"""

from __future__ import annotations
import inspect
import importlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, wraps
from types import FunctionType, ModuleType
from typing import Any, get_type_hints

from python_introspect import parameter_exclusions, set_parameter_exclusions
from openhcs.processing.backends.cellprofiler.library import (
    coerce_absorbed_processing_contract,
    function_name_candidates,
    function_inventory,
    get_contract,
)
from openhcs.processing.backends.cellprofiler.function_documentation import (
    enrich_cellprofiler_function_documentation,
)
from openhcs.core.callable_contract import (
    CallableContract,
    attach_callable_contract_metadata,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_reference import (
    FunctionReference,
    FunctionReferenceTransportAuthority,
    FunctionReferenceTransportStrategy,
)
from openhcs.processing.backends.lib_registry.openhcs_registry import (
    OpenHCSFunctionCatalogModule,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

# Import the compile-time contract provider with the public CellProfiler backend.
# Public pipeline documents intentionally store plain backend callables; the
# compiler discovers the CellProfiler artifact contract through this registered
# provider instead of exposing runtime wrappers in user-editable code.
from openhcs.interop.cellprofiler import compile_time_contracts as _compile_time_contracts

CELLPROFILER_MODULE_ATTR = "__openhcs_cellprofiler_module__"
CELLPROFILER_BACKEND_MODULE = "openhcs.processing.backends.cellprofiler"
CELLPROFILER_PUBLIC_API_NAMES = (
    "CellProfilerFunctionCatalog",
    "CellProfilerFunctionRuntimeMetadata",
)
_CELLPROFILER_FUNCTION_RESOLUTION_STACK: set[str] = set()


def _resolved_function_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    """Return evaluated annotations for public CellProfiler wrapper signatures."""
    try:
        return dict(get_type_hints(func, include_extras=True))
    except Exception:
        return inspect.get_annotations(func, eval_str=False).copy()


def _signature_with_resolved_annotations(func: Callable[..., Any]) -> inspect.Signature:
    """Return a signature whose parameter annotations match the runtime type view."""
    signature = inspect.signature(func)
    annotations = _resolved_function_annotations(func)
    parameters = [
        parameter.replace(annotation=annotations[name])
        if name in annotations
        else parameter
        for name, parameter in signature.parameters.items()
    ]
    return signature.replace(
        parameters=parameters,
        return_annotation=annotations.get("return", signature.return_annotation),
    )


class CellProfilerBackendModule(OpenHCSFunctionCatalogModule):
    """Package attribute authority for CellProfiler backend function exports."""

    @property
    def __all__(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    *CELLPROFILER_PUBLIC_API_NAMES,
                    *CellProfilerFunctionCatalog.list_functions(),
                )
            )
        )

    def openhcs_registry_functions(self) -> tuple[Callable[..., Any], ...]:
        return tuple(
            CellProfilerFunctionCatalog.get_function(function_name)
            for function_name in CellProfilerFunctionCatalog.list_functions()
        )

    def __getattr__(self, name: str) -> Any:
        if CellProfilerFunctionCatalog.has_function(name):
            return CellProfilerFunctionCatalog.get_function(name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, ModuleType) and CellProfilerFunctionCatalog.has_function(
            name
        ):
            existing = self.__dict__.get(name)
            if callable(existing):
                return
            if name not in _CELLPROFILER_FUNCTION_RESOLUTION_STACK:
                ModuleType.__setattr__(
                    self,
                    name,
                    CellProfilerFunctionCatalog.get_function(name),
                )
                return
        ModuleType.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CellProfilerFunctionRuntimeMetadata:
    """Runtime identity attached to an absorbed CellProfiler backend callable."""

    module_name: str
    function_name: str
    processing_contract: ProcessingContract
    declared_processing_contract: str | None

    @classmethod
    def from_callable(
        cls, func: Callable[..., Any]
    ) -> "CellProfilerFunctionRuntimeMetadata | None":
        """Project OpenHCS-owned CellProfiler runtime metadata from a callable."""
        if not callable(func):
            raise TypeError(
                f"CellProfilerFunctionCatalog.runtime_metadata requires a callable, got {type(func).__name__}."
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
                f"CellProfiler function {func.__name__!r} has no declared ProcessingContract metadata."
            )
        return cls(
            module_name=str(module_name),
            function_name=func.__name__,
            processing_contract=processing_contract,
            declared_processing_contract=metadata.get(
                FunctionContractAttribute.declared_processing_contract
            ),
        )


class CellProfilerFunctionCatalog:
    """Nominal authority for absorbed CellProfiler backend function lookup."""

    @classmethod
    def runtime_metadata(
        cls, func: Callable[..., Any]
    ) -> CellProfilerFunctionRuntimeMetadata | None:
        return CellProfilerFunctionRuntimeMetadata.from_callable(func)

    @classmethod
    def list_functions(cls) -> tuple[str, ...]:
        """Return exported CellProfiler-compatible function names."""
        return tuple(function_inventory())

    @classmethod
    def has_function(cls, name: str) -> bool:
        """Return whether a function is declared by a CellProfiler module."""
        return name in function_name_candidates()

    @classmethod
    def get_function(cls, name: str) -> Callable[..., Any]:
        """Return one exported CellProfiler-compatible processing function."""
        return _declared_cellprofiler_function(name)

    @classmethod
    def require_function(
        cls, module_name: str, *, function_name: str | None = None
    ) -> Callable[..., Any]:
        """Return one OpenHCS-owned CellProfiler-compatible function."""
        contract_payload = get_contract(module_name)
        if contract_payload is None:
            raise KeyError(
                f"No CellProfiler-compatible processing module registered: {module_name!r}"
            )
        resolved_function_name = function_name or str(contract_payload["function_name"])
        try:
            return cls.get_function(resolved_function_name)
        except KeyError as exc:
            raise KeyError(
                f"CellProfiler-compatible processing module {module_name!r} declares missing function {resolved_function_name!r}."
            ) from exc

    @classmethod
    def unavailable_functions(cls) -> Mapping[str, str]:
        """Return absorbed modules that were skipped during backend loading."""
        return {}


class CellProfilerFunctionReferenceTransportStrategy(FunctionReferenceTransportStrategy):
    """Transport strategy for CellProfiler package-catalog callables."""

    strategy_key = "cellprofiler"

    def reference_for_callable(self, func: Callable) -> FunctionReference | None:
        function_name = self.function_name_for_callable(func)
        if function_name is None:
            return None
        try:
            CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None
        contract = CallableContract.from_callable(func)
        memory_type = (
            "python"
            if contract.input_memory_type is None
            else contract.input_memory_type
        )
        return FunctionReference(
            function_name=function_name,
            registry_name="cellprofiler",
            memory_type=memory_type,
            composite_key=f"cellprofiler:{function_name}",
            original_module=CELLPROFILER_BACKEND_MODULE,
            metadata=FunctionReferenceTransportAuthority.callable_metadata(func),
        )

    def normalized_callable(self, func: Callable) -> Callable | None:
        function_name = self.function_name_for_callable(func)
        if function_name is None:
            return None
        try:
            return CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None

    def normalized_module(self, module: ModuleType) -> Callable | None:
        function_name = self.function_name_for_module(module)
        if function_name is None:
            return None
        try:
            return CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None

    def preserve_callable(self, func: Callable) -> bool:
        from openhcs.interop.cellprofiler.runtime.module_execution import (
            CellProfilerGroupedRuntimeCallable,
            CellProfilerRuntimeCallable,
        )

        return isinstance(
            func,
            (CellProfilerRuntimeCallable, CellProfilerGroupedRuntimeCallable),
        )

    @staticmethod
    def function_name_for_callable(func: Callable) -> str | None:
        """Return the CellProfiler catalog name for a backend callable."""
        if not isinstance(func, FunctionType):
            raw_processing_function = CallableContract.from_callable(
                func
            ).raw_processing_function
            if (
                callable(raw_processing_function)
                and raw_processing_function is not func
            ):
                return CellProfilerFunctionReferenceTransportStrategy.function_name_for_callable(
                    raw_processing_function
                )
            return None
        module_name = func.__module__
        if module_name != CELLPROFILER_BACKEND_MODULE and not module_name.startswith(
            f"{CELLPROFILER_BACKEND_MODULE}."
        ):
            return None
        return func.__name__

    @staticmethod
    def function_name_for_module(module: ModuleType) -> str | None:
        """Return the CellProfiler catalog name represented by a submodule."""
        module_prefix = f"{CELLPROFILER_BACKEND_MODULE}."
        if not module.__name__.startswith(module_prefix):
            return None
        return module.__name__.removeprefix(module_prefix)


def _declared_processing_contract(
    module_name: str, function_name: str, absorbed_function: Callable[..., Any]
) -> ProcessingContract | None:
    contract = coerce_absorbed_processing_contract(
        module_name, function_name, absorbed_function
    )
    if isinstance(contract, ProcessingContract):
        return contract
    return None


def _make_processing_wrapper(
    *, module_name: str, func: Callable[..., Any], contract: ProcessingContract
) -> Callable[..., Any]:
    """Build an OpenHCS-owned wrapper around one absorbed implementation."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__name__
    wrapper.__module__ = __name__
    wrapper.__signature__ = _signature_with_resolved_annotations(func)
    wrapper.__annotations__ = _resolved_function_annotations(func)
    callable_contract = CallableContract.from_callable(func)
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_module(module_name)
    if module_type is None:
        raise KeyError(
            f"CellProfiler-compatible processing function requires a CellProfilerModule declaration for {module_name!r}."
        )
    wrapper.input_memory_type = callable_contract.input_memory_type
    wrapper.output_memory_type = callable_contract.output_memory_type
    setattr(wrapper, FunctionContractAttribute.processing_contract, contract)
    setattr(
        wrapper, FunctionContractAttribute.declared_processing_contract, contract.name
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
        wrapper, module_name=module_name, source_function=func
    )
    hidden_parameters = parameter_exclusions(func)
    if hidden_parameters:
        set_parameter_exclusions(wrapper, hidden_parameters)
    return wrapper


@lru_cache(maxsize=1)
def _default_module_names_by_function_name() -> dict[str, str]:
    """Map each declared function to the CellProfiler module that owns it."""
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    default_modules: dict[str, str] = {}
    for module_type in CellProfilerModule.__registry__.values():
        module_name = str(module_type.module_name)
        function_names = (
            str(module_type.function_name),
            *(str(name) for name in module_type.function_variants),
        )
        for function_name in function_names:
            default_modules.setdefault(function_name, module_name)
    return default_modules


def _declared_cellprofiler_function(function_name: str) -> Callable[..., Any]:
    """Return one absorbed function wrapper without loading the full catalog."""
    installed = globals().get(function_name)
    if callable(installed):
        metadata = CellProfilerFunctionRuntimeMetadata.from_callable(installed)
        if metadata is not None and metadata.function_name == function_name:
            return installed

    inventory = function_inventory()
    if function_name not in inventory:
        raise KeyError(function_name)
    absorbed_function = _function_from_inventory(function_name)
    _default_module_names_by_function_name.cache_clear()
    module_name = _module_name_for_declared_function(function_name)
    contract = _declared_processing_contract(
        module_name, function_name, absorbed_function
    )
    if contract is None:
        raise KeyError(function_name)
    wrapper = _make_processing_wrapper(
        module_name=module_name, func=absorbed_function, contract=contract
    )
    globals()[function_name] = wrapper
    return wrapper


def _module_name_for_declared_function(function_name: str) -> str:
    """Return the CellProfiler module declaration that owns one function."""

    module_names = _default_module_names_by_function_name()
    if function_name not in module_names:
        raise KeyError(
            f"CellProfiler-compatible processing function {function_name!r} "
            "has no CellProfilerModule declaration."
        )
    return module_names[function_name]


def _function_from_inventory(function_name: str) -> Callable[..., Any]:
    location = function_inventory()[function_name]
    _CELLPROFILER_FUNCTION_RESOLUTION_STACK.add(function_name)
    try:
        module = importlib.import_module(location.module_name)
    finally:
        _CELLPROFILER_FUNCTION_RESOLUTION_STACK.discard(function_name)
    function = vars(module).get(function_name)
    if not callable(function):
        raise KeyError(
            f"Absorbed CellProfiler function {function_name!r} is missing from {location.module_name!r}."
        )
    return function


def __dir__() -> list[str]:
    return sorted((*globals(), *sys.modules[__name__].__all__))


sys.modules[__name__].__class__ = CellProfilerBackendModule
