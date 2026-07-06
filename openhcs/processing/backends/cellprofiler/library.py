"""Product-owned registry for absorbed CellProfiler functions."""

from __future__ import annotations
import inspect
import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule


@dataclass(frozen=True, slots=True)
class AbsorbedFunctionMetadata:
    """Validated metadata for one absorbed CellProfiler module."""

    module_name: str
    aliases: tuple[str, ...]
    function_name: str
    function_variants: tuple[str, ...]
    confidence: float
    validated: bool

    @classmethod
    def from_json(
        cls, module_name: str, payload: Mapping[str, Any]
    ) -> "AbsorbedFunctionMetadata":
        function_name = _required_string(payload, "function_name", module_name)
        return cls(
            module_name=module_name,
            aliases=_string_tuple(payload, "aliases", module_name),
            function_name=function_name,
            function_variants=_function_variant_tuple(
                payload, module_name=module_name, primary_function_name=function_name
            ),
            confidence=_required_float(payload, "confidence", module_name),
            validated=_required_bool(payload, "validated", module_name),
        )

    @classmethod
    def from_module_class(
        cls, module_type: type[CellProfilerModule]
    ) -> "AbsorbedFunctionMetadata":
        """Project compatibility metadata from one registered module class."""
        return cls(
            module_name=str(module_type.module_name),
            aliases=tuple(module_type.aliases),
            function_name=str(module_type.function_name),
            function_variants=tuple(module_type.function_variants),
            confidence=float(module_type.confidence),
            validated=bool(module_type.validated),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the historical metadata shape consumed by converter code."""
        payload: dict[str, Any] = {
            "function_name": self.function_name,
            "confidence": self.confidence,
            "validated": self.validated,
        }
        if self.aliases:
            payload["aliases"] = list(self.aliases)
        if self.function_variants:
            payload["function_variants"] = list(self.function_variants)
        return payload

    @property
    def declared_function_names(self) -> tuple[str, ...]:
        """Return all public functions owned by this module contract."""
        return (self.function_name, *self.function_variants)


@dataclass(frozen=True, slots=True)
class AbsorbedFunctionLocation:
    """Import location for one top-level absorbed function."""

    module_name: str
    function_name: str

    @property
    def source_path(self) -> Path | None:
        """Return the source file that declares this absorbed function."""
        spec = importlib.util.find_spec(self.module_name)
        if spec is None or spec.origin is None:
            return None
        return Path(spec.origin)

    def callable_source_path(self) -> Path | None:
        """Return the implementation source for this function, following facades."""
        module = importlib.import_module(self.module_name)
        function = module.__dict__.get(self.function_name)
        if not callable(function):
            return None
        source_file = inspect.getsourcefile(inspect.unwrap(function))
        return Path(source_file) if source_file is not None else None


_function_cache: dict[tuple[str, str], Callable[..., Any]] = {}


def _cellprofiler_module_root() -> type["CellProfilerModule"]:
    """Return the module declaration root without import-time registry cycles."""
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    return CellProfilerModule


def canonical_module_name(module_name: str) -> str:
    """Return the canonical absorbed module name for a CellProfiler module name."""
    return _cellprofiler_module_root().canonical_module_name(module_name)


def get_function(
    module_name: str, *, function_name: str | None = None
) -> Callable[..., Any] | None:
    """Return the absorbed function for a CellProfiler module, if registered."""
    canonical_name = canonical_module_name(module_name)
    metadata = _absorbed_contracts().get(canonical_name)
    if metadata is None:
        return None
    resolved_function_name = function_name or metadata.function_name
    cache_key = (canonical_name, resolved_function_name)
    cached = _function_cache.get(cache_key)
    if cached is not None:
        return cached
    location = _absorbed_function_locations().get(resolved_function_name)
    if location is None:
        return None
    module = importlib.import_module(location.module_name)
    function = module.__dict__.get(resolved_function_name)
    if not callable(function):
        return None
    coerce_absorbed_processing_contract(
        canonical_name, resolved_function_name, function
    )
    _function_cache[cache_key] = function
    return function


def require_function(
    module_name: str, *, function_name: str | None = None
) -> Callable[..., Any]:
    """Return one absorbed function or raise a precise registry error."""
    function = get_function(module_name, function_name=function_name)
    if function is not None:
        return function
    canonical_name = canonical_module_name(module_name)
    metadata = _absorbed_contracts().get(canonical_name)
    if metadata is None:
        raise KeyError(f"No absorbed CellProfiler module registered: {module_name!r}")
    resolved_function_name = function_name or metadata.function_name
    raise KeyError(
        f"Absorbed CellProfiler module {module_name!r} declares missing function {resolved_function_name!r}."
    )


def get_contract(module_name: str) -> dict[str, Any] | None:
    """Return contract metadata for one absorbed CellProfiler module."""
    metadata = _absorbed_contracts().get(canonical_module_name(module_name))
    if metadata is None:
        return None
    return metadata.to_json()


def list_modules() -> list[str]:
    """List absorbed CellProfiler module names."""
    return list(_absorbed_contracts().keys())


def validated_contracts() -> Mapping[str, dict[str, Any]]:
    """Return validated absorbed module contracts keyed by canonical module name."""
    return {
        module_name: metadata.to_json()
        for module_name, metadata in _absorbed_contracts().items()
        if metadata.validated
    }


def function_inventory() -> Mapping[str, AbsorbedFunctionLocation]:
    """Return the derived absorbed function location index."""
    return _absorbed_function_locations()


def function_name_candidates() -> frozenset[str]:
    """Return declared CellProfiler function names without locating source files."""
    return _declared_function_name_candidates_for(_absorbed_contract_signature())


def function_source_path(function_name: str) -> Path | None:
    """Return the source file for an absorbed function, if registered."""
    location = _absorbed_function_locations().get(function_name)
    if location is None:
        return None
    return location.callable_source_path() or location.source_path


def coerce_absorbed_processing_contract(
    module_name: str, function_name: str, function: Callable[..., Any]
) -> ProcessingContract | None:
    """Return or install nominal processing metadata for an executable function.

    The current compatibility catalog supplies canonical module-function
    metadata until the absorbed module catalog is generated from registered
    class declarations. This boundary converts that declaration into a nominal
    ProcessingContract attribute exactly once. Downstream executable registries
    only consume callable metadata; undecorated helper functions stay private to
    their implementation modules.
    """
    processing_contract_key = FunctionContractAttribute.processing_contract
    raw_contract = vars(function).get(processing_contract_key)
    if isinstance(raw_contract, ProcessingContract):
        return raw_contract
    if raw_contract is not None:
        raise TypeError(
            f"Absorbed CellProfiler function {function_name!r} declares {processing_contract_key} as {type(raw_contract).__name__}; expected ProcessingContract."
        )
    declared_contract = _absorbed_default_function_contracts().get(function_name)
    if declared_contract is None:
        return None
    canonical_name = canonical_module_name(module_name)
    metadata = _absorbed_contracts().get(canonical_name)
    if metadata is None or function_name not in metadata.declared_function_names:
        return None
    vars(function)[processing_contract_key] = declared_contract
    return declared_contract


def coerce_registered_absorbed_processing_contract(
    function_name: str, function: Callable[..., Any]
) -> ProcessingContract | None:
    """Install nominal processing metadata for a registered absorbed function."""
    for metadata in _absorbed_contracts().values():
        if metadata.function_name != function_name:
            continue
        return coerce_absorbed_processing_contract(
            metadata.module_name, function_name, function
        )
    return None


def _absorbed_contracts() -> Mapping[str, AbsorbedFunctionMetadata]:
    """Return the declaration-derived absorbed module catalog."""
    return _load_contracts()


def _absorbed_default_function_contracts() -> Mapping[str, ProcessingContract]:
    """Return function processing contracts derived from module declarations."""
    return _absorbed_default_function_contracts_for(_absorbed_contract_signature())


def _absorbed_function_locations() -> Mapping[str, AbsorbedFunctionLocation]:
    """Return function locations filtered by declared module function names."""
    return _absorbed_function_locations_for(_absorbed_contract_signature())


@lru_cache(maxsize=16)
def _absorbed_default_function_contracts_for(
    contract_signature: tuple[tuple[str, str, tuple[str, ...], str | None], ...],
) -> Mapping[str, ProcessingContract]:
    """Return cached function contracts for one registry contract signature."""
    del contract_signature
    return _load_default_function_contracts(_absorbed_contracts())


@lru_cache(maxsize=16)
def _absorbed_function_locations_for(
    contract_signature: tuple[tuple[str, str, tuple[str, ...], str | None], ...],
) -> Mapping[str, AbsorbedFunctionLocation]:
    """Return cached function locations for one registry contract signature."""
    del contract_signature
    return _discover_function_locations(_absorbed_contracts())


@lru_cache(maxsize=16)
def _declared_function_name_candidates_for(
    contract_signature: tuple[tuple[str, str, tuple[str, ...], str | None], ...],
) -> frozenset[str]:
    """Return cached function names declared by registered module classes."""
    del contract_signature
    return frozenset(
        function_name
        for module_type in _cellprofiler_module_root().__registry__.values()
        for function_name in module_type.declared_function_names()
    )


def _load_contracts() -> Mapping[str, AbsorbedFunctionMetadata]:
    _ensure_module_declarations_loaded()
    contracts = {}
    for module_type in _cellprofiler_module_root().__registry__.values():
        metadata = AbsorbedFunctionMetadata.from_module_class(module_type)
        contracts[metadata.module_name] = metadata
    return MappingProxyType(contracts)


def _ensure_module_declarations_loaded() -> None:
    """Import CellProfiler backend modules before snapshotting declarations."""
    package = importlib.import_module("openhcs.processing.backends.cellprofiler")
    for _importer, module_name, is_package in pkgutil.iter_modules(
        package.__path__,
        f"{package.__name__}.",
    ):
        if is_package or module_name.endswith(".library"):
            continue
        importlib.import_module(module_name)


def _absorbed_contract_signature() -> (
    tuple[tuple[str, str, tuple[str, ...], str | None], ...]
):
    """Return a stable cache key for the currently discovered module contracts."""
    return tuple(
        sorted(
            (
                str(module_type.module_name),
                str(module_type.function_name),
                tuple(module_type.function_variants),
                (
                    None
                    if module_type.contract is None
                    else module_type.contract.declared_name
                ),
            )
            for module_type in _cellprofiler_module_root().__registry__.values()
        )
    )


def _load_default_function_contracts(
    contracts: Mapping[str, AbsorbedFunctionMetadata],
) -> Mapping[str, ProcessingContract]:
    del contracts
    declared_contracts: dict[str, ProcessingContract] = {}
    for module_type in _cellprofiler_module_root().__registry__.values():
        contract = module_type.contract
        if contract is None:
            continue
        for function_name in module_type.declared_function_names():
            existing = declared_contracts.get(function_name)
            if existing is not None and existing is not contract:
                raise ValueError(
                    f"Absorbed CellProfiler function {function_name!r} has conflicting declared contracts {existing.name!r} and {contract.name!r}."
                )
            declared_contracts[function_name] = contract
    return MappingProxyType(declared_contracts)


def _discover_function_locations(
    contracts: Mapping[str, AbsorbedFunctionMetadata],
) -> Mapping[str, AbsorbedFunctionLocation]:
    del contracts
    locations: dict[str, AbsorbedFunctionLocation] = {}
    for module_type in _cellprofiler_module_root().__registry__.values():
        module_name = module_type.__module__
        for function_name in module_type.declared_function_names():
            existing = locations.get(function_name)
            if existing is not None and existing.module_name != module_name:
                raise ValueError(
                    f"CellProfiler function {function_name!r} is declared in both {existing.module_name!r} and {module_name!r}."
                )
            locations[function_name] = AbsorbedFunctionLocation(
                module_name=module_name,
                function_name=function_name,
            )
    return MappingProxyType(locations)


def _required_string(payload: Mapping[str, Any], key: str, module_name: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} must define {key}."
        )
    return value


def _required_float(payload: Mapping[str, Any], key: str, module_name: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} must define numeric {key}."
        )
    return float(value)


def _required_bool(payload: Mapping[str, Any], key: str, module_name: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} must define boolean {key}."
        )
    return value


def _string_tuple(
    payload: Mapping[str, Any], key: str, module_name: str
) -> tuple[str, ...]:
    if key not in payload:
        return ()
    raw_values = payload[key]
    if raw_values is None:
        return ()
    if not isinstance(raw_values, list):
        raise TypeError(
            f"Absorbed CellProfiler module {module_name!r} must declare {key} as a list of strings."
        )
    values = tuple((str(value).strip() for value in raw_values))
    if any((not value for value in values)):
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} declares an empty {key}."
        )
    return values


def _function_variant_tuple(
    payload: Mapping[str, Any], *, module_name: str, primary_function_name: str
) -> tuple[str, ...]:
    variants = _string_tuple(payload, "function_variants", module_name)
    if primary_function_name in variants:
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} declares primary function {primary_function_name!r} as a variant."
        )
    if len(set(variants)) != len(variants):
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} declares duplicate function variants."
        )
    return variants


def _is_public_api_export(name: str, value: object) -> bool:
    return (
        not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and (value.__module__ == __name__)
    )


__all__ = tuple(
    (name for name, value in globals().items() if _is_public_api_export(name, value))
)
