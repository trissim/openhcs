"""Product-owned registry for absorbed CellProfiler functions."""

from __future__ import annotations

import ast
import inspect
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule


_LIBRARY_ROOT = Path(__file__).parent
_FUNCTIONS_PACKAGE = "benchmark.cellprofiler_library.functions"
_BACKEND_FUNCTIONS_PACKAGE = "openhcs.processing.backends.cellprofiler"
_INTEROP_FUNCTIONS_PACKAGE = "openhcs.interop.cellprofiler"


def _functions_root() -> Path:
    repo_functions_root = (
        Path(__file__).resolve().parents[4]
        / "benchmark"
        / "cellprofiler_library"
        / "functions"
    )
    if repo_functions_root.exists():
        return repo_functions_root
    spec = importlib.util.find_spec(_FUNCTIONS_PACKAGE)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            f"Absorbed CellProfiler functions package {_FUNCTIONS_PACKAGE!r} "
            "is not importable."
        )
    return Path(next(iter(spec.submodule_search_locations)))


_FUNCTIONS_ROOT = _functions_root()


@dataclass(frozen=True, slots=True)
class AbsorbedFunctionMetadata:
    """Validated metadata for one absorbed CellProfiler module."""

    module_name: str
    aliases: tuple[str, ...]
    function_name: str
    function_variants: tuple[str, ...]
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
            function_variants=_function_variant_tuple(
                payload,
                module_name=module_name,
                primary_function_name=function_name,
            ),
            contract=str(
                payload.get("contract", ProcessingContract.PURE_2D.declared_name)
            ),
            category=str(payload.get("category", "image_operation")),
            confidence=float(payload.get("confidence", 0.5)),
            validated=bool(payload.get("validated", False)),
        )

    @classmethod
    def from_module_class(
        cls,
        module_type: type[CellProfilerModule],
    ) -> "AbsorbedFunctionMetadata":
        """Project compatibility metadata from one registered module class."""
        return cls(
            module_name=str(module_type.module_name),
            aliases=tuple(module_type.aliases),
            function_name=str(module_type.function_name),
            function_variants=tuple(module_type.function_variants),
            contract=str(module_type.contract),
            category=str(module_type.category),
            confidence=float(module_type.confidence),
            validated=bool(module_type.validated),
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

    package: str
    root: Path
    module_stem: str
    function_name: str

    @property
    def module_name(self) -> str:
        return f"{self.package}.{self.module_stem}"

    @property
    def source_path(self) -> Path:
        """Return the source file that declares this absorbed function."""
        return self.root / f"{self.module_stem}.py"

    def callable_source_path(self) -> Path | None:
        """Return the implementation source for this function, following facades."""
        module = importlib.import_module(self.module_name)
        function = module.__dict__.get(self.function_name)
        if not callable(function):
            return None
        source_file = inspect.getsourcefile(inspect.unwrap(function))
        return Path(source_file) if source_file is not None else None


@dataclass(frozen=True, slots=True)
class AbsorbedFunctionModuleExports:
    """AST-derived executable exports for one absorbed function module."""

    parsed_module: ast.Module
    declared_function_names: set[str]
    declared_only: bool

    def public_function_names(self) -> tuple[str, ...]:
        exports: list[str] = []
        for node in self.parsed_module.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                exports.append(node.name)
            elif isinstance(node, ast.ImportFrom):
                exports.extend(self.imported_declared_function_names(node))
        if self.declared_only:
            exports = [
                name for name in exports if name in self.declared_function_names
            ]
        return tuple(dict.fromkeys(exports))

    def imported_declared_function_names(self, node: ast.ImportFrom) -> tuple[str, ...]:
        names: list[str] = []
        for alias in node.names:
            exported_name = alias.asname or alias.name
            if (
                not exported_name.startswith("_")
                and exported_name in self.declared_function_names
            ):
                names.append(exported_name)
        return tuple(names)


_function_cache: dict[tuple[str, str], Callable[..., Any]] = {}


def _cellprofiler_module_root() -> type["CellProfilerModule"]:
    """Return the module declaration root without import-time registry cycles."""
    from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

    return CellProfilerModule


def canonical_module_name(module_name: str) -> str:
    """Return the canonical absorbed module name for a CellProfiler module name."""
    return _cellprofiler_module_root().canonical_module_name(module_name)


def get_function(
    module_name: str,
    *,
    function_name: str | None = None,
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
    metadata = _absorbed_contracts().get(canonical_name)
    if metadata is None:
        raise KeyError(f"No absorbed CellProfiler module registered: {module_name!r}")
    resolved_function_name = function_name or metadata.function_name
    raise KeyError(
        f"Absorbed CellProfiler module {module_name!r} declares missing "
        f"function {resolved_function_name!r}."
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


def function_source_path(function_name: str) -> Path | None:
    """Return the source file for an absorbed function, if registered."""
    location = _absorbed_function_locations().get(function_name)
    if location is None:
        return None
    return location.callable_source_path() or location.source_path


def coerce_absorbed_processing_contract(
    module_name: str,
    function_name: str,
    function: Callable[..., Any],
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
            f"Absorbed CellProfiler function {function_name!r} declares "
            f"{processing_contract_key} as {type(raw_contract).__name__}; "
            "expected ProcessingContract."
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
    function_name: str,
    function: Callable[..., Any],
) -> ProcessingContract | None:
    """Install nominal processing metadata for a registered absorbed function."""
    for metadata in _absorbed_contracts().values():
        if metadata.function_name != function_name:
            continue
        return coerce_absorbed_processing_contract(
            metadata.module_name,
            function_name,
            function,
        )
    return None


@lru_cache(maxsize=1)
def _absorbed_contracts() -> Mapping[str, AbsorbedFunctionMetadata]:
    """Return the declaration-derived absorbed module catalog."""
    return _load_contracts()


@lru_cache(maxsize=1)
def _absorbed_default_function_contracts() -> Mapping[str, ProcessingContract]:
    """Return function processing contracts derived from module declarations."""
    return _load_default_function_contracts(_absorbed_contracts())


@lru_cache(maxsize=1)
def _absorbed_function_locations() -> Mapping[str, AbsorbedFunctionLocation]:
    """Return function locations filtered by declared module function names."""
    return _discover_function_locations(_absorbed_contracts())


def _load_contracts() -> Mapping[str, AbsorbedFunctionMetadata]:
    contracts = {}
    for module_type in _cellprofiler_module_root().__registry__.values():
        metadata = AbsorbedFunctionMetadata.from_module_class(module_type)
        contracts[metadata.module_name] = metadata
    return MappingProxyType(contracts)


def _load_default_function_contracts(
    contracts: Mapping[str, AbsorbedFunctionMetadata],
) -> Mapping[str, ProcessingContract]:
    declared_contracts: dict[str, ProcessingContract] = {}
    for module_name, metadata in contracts.items():
        contract = ProcessingContract.from_declared_name(metadata.contract)
        if contract is None:
            continue
        for function_name in metadata.declared_function_names:
            existing = declared_contracts.get(function_name)
            if existing is not None and existing is not contract:
                raise ValueError(
                    f"Absorbed CellProfiler function {function_name!r} "
                    f"has conflicting declared contracts {existing.name!r} and "
                    f"{contract.name!r}."
                )
            declared_contracts[function_name] = contract
    return MappingProxyType(declared_contracts)


def _discover_function_locations(
    contracts: Mapping[str, AbsorbedFunctionMetadata],
) -> Mapping[str, AbsorbedFunctionLocation]:
    locations: dict[str, AbsorbedFunctionLocation] = {}
    declared_function_names = {
        function_name
        for metadata in contracts.values()
        for function_name in metadata.declared_function_names
    }
    _register_function_locations(
        locations,
        package=_BACKEND_FUNCTIONS_PACKAGE,
        root=_LIBRARY_ROOT,
        replace_existing=False,
        declared_only=True,
        declared_function_names=declared_function_names,
    )
    interop_spec = importlib.util.find_spec(_INTEROP_FUNCTIONS_PACKAGE)
    if interop_spec is not None and interop_spec.submodule_search_locations:
        _register_function_locations(
            locations,
            package=_INTEROP_FUNCTIONS_PACKAGE,
            root=Path(next(iter(interop_spec.submodule_search_locations))),
            replace_existing=False,
            declared_only=True,
            declared_function_names=declared_function_names,
        )
    _register_function_locations(
        locations,
        package=_FUNCTIONS_PACKAGE,
        root=_FUNCTIONS_ROOT,
        replace_existing=False,
        declared_only=False,
        declared_function_names=declared_function_names,
    )
    return MappingProxyType(locations)


def _register_function_locations(
    locations: dict[str, AbsorbedFunctionLocation],
    *,
    package: str,
    root: Path,
    replace_existing: bool,
    declared_only: bool,
    declared_function_names: set[str],
) -> None:
    for file_path in sorted(root.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        module_stem = file_path.stem
        parsed_module = ast.parse(
            file_path.read_text(encoding="utf-8"),
            filename=str(file_path),
        )
        module_exports = AbsorbedFunctionModuleExports(
            declared_function_names=declared_function_names,
            declared_only=declared_only,
            parsed_module=parsed_module,
        )
        for function_name in module_exports.public_function_names():
            if function_name in locations and not replace_existing:
                existing = locations[function_name]
                if existing.package != package:
                    continue
                raise ValueError(
                    f"CellProfiler function {function_name!r} is declared in both "
                    f"{existing.module_name!r} and {package}.{module_stem!r}."
                )
            locations[function_name] = AbsorbedFunctionLocation(
                package=package,
                root=root,
                module_stem=module_stem,
                function_name=function_name,
            )

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


def _function_variant_tuple(
    payload: Mapping[str, Any],
    *,
    module_name: str,
    primary_function_name: str,
) -> tuple[str, ...]:
    variants = _string_tuple(payload, "function_variants", module_name)
    if primary_function_name in variants:
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} declares primary "
            f"function {primary_function_name!r} as a variant."
        )
    if len(set(variants)) != len(variants):
        raise ValueError(
            f"Absorbed CellProfiler module {module_name!r} declares duplicate "
            "function variants."
        )
    return variants


def _is_public_api_export(name: str, value: object) -> bool:
    return (
        not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == __name__
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if _is_public_api_export(name, value)
)
