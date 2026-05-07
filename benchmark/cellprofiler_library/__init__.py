"""Typed registry for absorbed CellProfiler functions."""

from __future__ import annotations

import ast
import inspect
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openhcs.core.callable_contract import PROCESSING_CONTRACT_ATTR
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


_LIBRARY_ROOT = Path(__file__).parent
_CONTRACTS_PATH = _LIBRARY_ROOT / "contracts.json"
_FUNCTIONS_ROOT = _LIBRARY_ROOT / "functions"
_FUNCTIONS_PACKAGE = f"{__package__}.functions"


@dataclass(frozen=True, slots=True)
class AbsorbedFunctionMetadata:
    """Validated metadata for one absorbed CellProfiler module."""

    module_name: str
    aliases: tuple[str, ...]
    function_name: str
    contract: str
    category: str
    confidence: float
    validated: bool

    @classmethod
    def from_json(
        cls,
        module_name: str,
        payload: Mapping[str, Any],
    ) -> "AbsorbedFunctionMetadata":
        function_name = _required_string(payload, "function_name", module_name)
        return cls(
            module_name=module_name,
            aliases=_string_tuple(payload, "aliases", module_name),
            function_name=function_name,
            contract=str(payload.get("contract", "pure_2d")),
            category=str(payload.get("category", "image_operation")),
            confidence=float(payload.get("confidence", 0.5)),
            validated=bool(payload.get("validated", False)),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the historical metadata shape consumed by converter code."""
        payload: dict[str, Any] = {
            "function_name": self.function_name,
            "contract": self.contract,
            "category": self.category,
            "confidence": self.confidence,
            "validated": self.validated,
        }
        if self.aliases:
            payload["aliases"] = list(self.aliases)
        return payload


@dataclass(frozen=True, slots=True)
class AbsorbedFunctionLocation:
    """Import location for one top-level absorbed function."""

    module_stem: str
    function_name: str

    @property
    def module_name(self) -> str:
        return f"{_FUNCTIONS_PACKAGE}.{self.module_stem}"


_contracts: Mapping[str, AbsorbedFunctionMetadata] = MappingProxyType({})
_canonical_module_names: Mapping[str, str] = MappingProxyType({})
_function_locations: Mapping[str, AbsorbedFunctionLocation] = MappingProxyType({})
_default_function_contracts: Mapping[str, ProcessingContract] = MappingProxyType({})
_function_cache: dict[tuple[str, str], Callable[..., Any]] = {}


def canonical_module_name(module_name: str) -> str:
    """Return the canonical absorbed module name for a CellProfiler module name."""
    normalized = module_name.strip()
    if not normalized:
        raise ValueError("CellProfiler module name cannot be empty.")
    return _canonical_module_names.get(
        _module_lookup_key(normalized),
        normalized,
    )


def get_function(
    module_name: str,
    *,
    function_name: str | None = None,
) -> Callable[..., Any] | None:
    """Return the absorbed function for a CellProfiler module, if registered."""
    canonical_name = canonical_module_name(module_name)
    metadata = _contracts.get(canonical_name)
    if metadata is None:
        return None

    resolved_function_name = function_name or metadata.function_name
    cache_key = (canonical_name, resolved_function_name)
    cached = _function_cache.get(cache_key)
    if cached is not None:
        return cached

    location = _function_locations.get(resolved_function_name)
    if location is None:
        return None

    module = importlib.import_module(location.module_name)
    function = module.__dict__.get(resolved_function_name)
    if not callable(function):
        return None
    coerce_absorbed_processing_contract(
        canonical_name,
        resolved_function_name,
        function,
    )
    _function_cache[cache_key] = function
    return function


def require_function(
    module_name: str,
    *,
    function_name: str | None = None,
) -> Callable[..., Any]:
    """Return one absorbed function or raise a precise registry error."""
    function = get_function(module_name, function_name=function_name)
    if function is not None:
        return function

    canonical_name = canonical_module_name(module_name)
    metadata = _contracts.get(canonical_name)
    if metadata is None:
        raise KeyError(f"No absorbed CellProfiler module registered: {module_name!r}")
    resolved_function_name = function_name or metadata.function_name
    raise KeyError(
        f"Absorbed CellProfiler module {module_name!r} declares missing "
        f"function {resolved_function_name!r}."
    )


def get_contract(module_name: str) -> dict[str, Any] | None:
    """Return contract metadata for one absorbed CellProfiler module."""
    metadata = _contracts.get(canonical_module_name(module_name))
    if metadata is None:
        return None
    return metadata.to_json()


def list_modules() -> list[str]:
    """List absorbed CellProfiler module names."""
    return list(_contracts.keys())


def function_inventory() -> Mapping[str, AbsorbedFunctionLocation]:
    """Return the derived absorbed function location index."""
    return _function_locations


def coerce_absorbed_processing_contract(
    module_name: str,
    function_name: str,
    function: Callable[..., Any],
) -> ProcessingContract | None:
    """Return or install nominal processing metadata for an executable function.

    The JSON catalog is the ingestion contract for canonical module functions.
    This boundary converts that external declaration into a nominal
    ProcessingContract attribute exactly once. Downstream executable registries
    only consume callable metadata; undecorated helper functions stay private to
    their implementation modules.
    """
    raw_contract = vars(function).get(PROCESSING_CONTRACT_ATTR)
    if isinstance(raw_contract, ProcessingContract):
        return raw_contract
    if raw_contract is not None:
        raise TypeError(
            f"Absorbed CellProfiler function {function_name!r} declares "
            f"{PROCESSING_CONTRACT_ATTR} as {type(raw_contract).__name__}; "
            "expected ProcessingContract."
        )

    declared_contract = _default_function_contracts.get(function_name)
    if declared_contract is None:
        return None
    canonical_name = canonical_module_name(module_name)
    metadata = _contracts.get(canonical_name)
    if metadata is None or metadata.function_name != function_name:
        return None

    setattr(function, PROCESSING_CONTRACT_ATTR, declared_contract)
    return declared_contract


def coerce_registered_absorbed_processing_contract(
    function_name: str,
    function: Callable[..., Any],
) -> ProcessingContract | None:
    """Install nominal processing metadata for a registered absorbed function."""
    for metadata in _contracts.values():
        if metadata.function_name != function_name:
            continue
        return coerce_absorbed_processing_contract(
            metadata.module_name,
            function_name,
            function,
        )
    return None


def _load_contracts() -> Mapping[str, AbsorbedFunctionMetadata]:
    if not _CONTRACTS_PATH.exists():
        return MappingProxyType({})
    raw_registry = json.loads(_CONTRACTS_PATH.read_text())
    contracts = {
        module_name: AbsorbedFunctionMetadata.from_json(module_name, payload)
        for module_name, payload in raw_registry.items()
    }
    return MappingProxyType(contracts)


def _load_canonical_module_names(
    contracts: Mapping[str, AbsorbedFunctionMetadata],
) -> Mapping[str, str]:
    canonical_names: dict[str, str] = {}
    for module_name, metadata in contracts.items():
        _register_module_name(canonical_names, module_name, module_name)
        for alias in metadata.aliases:
            _register_module_name(canonical_names, alias, module_name)
    return MappingProxyType(canonical_names)


def _load_default_function_contracts(
    contracts: Mapping[str, AbsorbedFunctionMetadata],
) -> Mapping[str, ProcessingContract]:
    declared_contracts: dict[str, ProcessingContract] = {}
    for module_name, metadata in contracts.items():
        contract = ProcessingContract.from_declared_name(metadata.contract)
        if contract is None:
            continue
        existing = declared_contracts.get(metadata.function_name)
        if existing is not None and existing is not contract:
            raise ValueError(
                f"Absorbed CellProfiler function {metadata.function_name!r} "
                f"has conflicting declared contracts {existing.name!r} and "
                f"{contract.name!r}."
            )
        declared_contracts[metadata.function_name] = contract
    return MappingProxyType(declared_contracts)


def _register_module_name(
    canonical_names: dict[str, str],
    module_name: str,
    canonical_name: str,
) -> None:
    normalized = module_name.strip()
    if not normalized:
        raise ValueError(
            f"Absorbed CellProfiler module {canonical_name!r} declares an empty alias."
        )
    key = _module_lookup_key(normalized)
    existing = canonical_names.get(key)
    if existing is not None and existing != canonical_name:
        raise ValueError(
            f"CellProfiler module name {normalized!r} maps to both "
            f"{existing!r} and {canonical_name!r}."
        )
    canonical_names[key] = canonical_name


def _discover_function_locations() -> Mapping[str, AbsorbedFunctionLocation]:
    locations: dict[str, AbsorbedFunctionLocation] = {}
    for file_path in sorted(_FUNCTIONS_ROOT.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        module_stem = file_path.stem
        parsed_module = ast.parse(file_path.read_text(), filename=str(file_path))
        for node in parsed_module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            if node.name in locations:
                existing = locations[node.name]
                raise ValueError(
                    f"Absorbed CellProfiler function {node.name!r} is declared in "
                    f"both {existing.module_stem!r} and {module_stem!r}."
                )
            locations[node.name] = AbsorbedFunctionLocation(
                module_stem=module_stem,
                function_name=node.name,
            )
    return MappingProxyType(locations)


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    module_name: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} must define {key}."
        )
    return value


def _string_tuple(
    payload: Mapping[str, Any],
    key: str,
    module_name: str,
) -> tuple[str, ...]:
    if key not in payload:
        return ()
    raw_values = payload[key]
    if raw_values is None:
        return ()
    if not isinstance(raw_values, list):
        raise TypeError(
            f"Absorbed CellProfiler module {module_name!r} must declare {key} "
            "as a list of strings."
        )
    values = tuple(str(value).strip() for value in raw_values)
    if any(not value for value in values):
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} declares an empty {key}."
        )
    return values


def _module_lookup_key(module_name: str) -> str:
    return module_name.strip().casefold()


def _is_public_api_export(name: str, value: object) -> bool:
    return (
        not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == __name__
    )


_contracts = _load_contracts()
_canonical_module_names = _load_canonical_module_names(_contracts)
_default_function_contracts = _load_default_function_contracts(_contracts)
_function_locations = _discover_function_locations()
__all__ = tuple(
    name
    for name, value in globals().items()
    if _is_public_api_export(name, value)
)
