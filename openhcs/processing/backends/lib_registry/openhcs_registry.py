"""
OpenHCS native function registry.

This registry processes OpenHCS functions that have been decorated with
explicit contract declarations, allowing them to skip runtime testing
while producing the same FunctionMetadata format as external libraries.
"""

import ast
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, ClassVar, Dict, List, Tuple
from weakref import WeakKeyDictionary

import numpy as np
from metaclass_registry import import_module_preserving_root_logging

from openhcs.constants import VALID_MEMORY_TYPES, MemoryType
from openhcs.core.callable_contract import (
    CallableContract,
    FunctionStepExecutionScope,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.public_api import is_declared_public_name
from openhcs.processing.backends.lib_registry.unified_registry import (
    FunctionMetadata,
    LibraryRegistryBase,
    ProcessingContract,
)
from openhcs.utils.environment import OpenHCSProcessEnvironment

logger = logging.getLogger(__name__)


class OpenHCSFunctionCatalogModule(ModuleType, ABC):
    """Module type that exposes OpenHCS registry functions through a catalog."""

    @abstractmethod
    def openhcs_registry_functions(self) -> tuple[Callable[..., Any], ...]:
        """Return processing functions owned by this module's catalog."""


class OpenHCSFunctionCatalogDeclaration(ABC):
    """Nominal declaration that assigns local callables to one catalog module."""

    registry_catalog_module: ClassVar[str | None] = None

    @classmethod
    @abstractmethod
    def declared_function_names(cls) -> tuple[str, ...]:
        """Return the local callable names owned by this declaration."""

    @classmethod
    def require_registry_catalog_module(cls) -> str:
        """Return the non-empty catalog module declared by this owner."""

        module_name = cls.registry_catalog_module
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(
                f"{cls.__name__}.registry_catalog_module must be a non-empty string."
            )
        return module_name


def _registry_catalog_module_for_callable(func: Callable[..., Any]) -> str:
    """Return the local nominal owner's catalog module or the defining module."""

    declared = inspect.unwrap(func)
    implementation_module = importlib.import_module(declared.__module__)
    owners = tuple(
        candidate
        for candidate in vars(implementation_module).values()
        if isinstance(candidate, type)
        and candidate is not OpenHCSFunctionCatalogDeclaration
        and issubclass(candidate, OpenHCSFunctionCatalogDeclaration)
        and candidate.__module__ == declared.__module__
        and declared.__name__ in candidate.declared_function_names()
    )
    if not owners:
        return declared.__module__
    if len(owners) > 1:
        raise ValueError(
            f"OpenHCS callable {declared.__module__}.{declared.__name__} is claimed "
            "by multiple catalog declarations: "
            f"{tuple(owner.__name__ for owner in owners)!r}."
        )
    return owners[0].require_registry_catalog_module()


_MEMORY_DECORATOR_IMPORT_MODULES = frozenset(
    {
        "openhcs.core.memory",
        "openhcs.core.memory.decorators",
        "openhcs.processing",
    }
)


def _allowed_openhcs_memory_types() -> frozenset[str] | None:
    """Return the memory types eligible for OpenHCS registry imports."""
    if not OpenHCSProcessEnvironment.cpu_only_mode():
        return None
    return frozenset((MemoryType.NUMPY.value,))


def _catalog_memory_types(func: Callable) -> frozenset[MemoryType] | None:
    """Return every available framework role admitted by current catalog policy."""

    try:
        declared = CallableContract.from_callable(func).declared_memory_types
    except ValueError:
        return None
    allowed = _allowed_openhcs_memory_types()
    if allowed is not None and any(
        memory_type.value not in allowed for memory_type in declared
    ):
        return None
    if any(not memory_type.is_installed() for memory_type in declared):
        return None
    return declared


def _module_declares_allowed_memory_type(
    module_name: str,
    allowed_memory_types: frozenset[str] | None,
) -> bool:
    if allowed_memory_types is None:
        return True
    spec = importlib.util.find_spec(module_name)
    origin = spec.origin if spec is not None else None
    if spec is not None and spec.submodule_search_locations is not None:
        return True
    if origin is None:
        return True
    try:
        source_stat = Path(origin).stat()
    except OSError:
        return True
    return _source_declares_allowed_memory_type(
        origin,
        source_stat.st_mtime_ns,
        source_stat.st_size,
        allowed_memory_types,
    )


@lru_cache(maxsize=1024)
def _source_declares_allowed_memory_type(
    origin: str,
    source_mtime_ns: int,
    source_size: int,
    allowed_memory_types: frozenset[str],
) -> bool:
    """Inspect one source revision under the current admission declaration."""

    del source_mtime_ns, source_size
    try:
        source = Path(origin).read_text(encoding="utf-8")
    except OSError:
        return True
    try:
        module_ast = ast.parse(source, filename=origin)
    except SyntaxError:
        return True
    import_bindings = _module_import_bindings(module_ast)
    declared_memory_types = {
        memory_type
        for node in ast.walk(module_ast)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        for memory_type in (_memory_type_from_decorator(decorator, import_bindings),)
        if memory_type is not None
    }
    if not declared_memory_types:
        return False
    return bool(declared_memory_types & allowed_memory_types)


def _module_import_bindings(module_ast: ast.Module) -> dict[str, str]:
    """Return local names bound to exact absolute import declarations."""

    bindings: dict[str, str] = {}
    for statement in module_ast.body:
        if isinstance(statement, ast.ImportFrom):
            if statement.level or statement.module is None:
                continue
            for imported in statement.names:
                if imported.name == "*":
                    continue
                bindings[imported.asname or imported.name] = (
                    f"{statement.module}.{imported.name}"
                )
        elif isinstance(statement, ast.Import):
            for imported in statement.names:
                if imported.asname is not None:
                    bindings[imported.asname] = imported.name
                    continue
                root_name = imported.name.partition(".")[0]
                bindings[root_name] = root_name
    return bindings


def _dotted_expression_name(expression: ast.expr) -> str | None:
    """Return a dotted name without inferring runtime attribute semantics."""

    if isinstance(expression, ast.Name):
        return expression.id
    if not isinstance(expression, ast.Attribute):
        return None
    owner = _dotted_expression_name(expression.value)
    return None if owner is None else f"{owner}.{expression.attr}"


def _memory_type_from_decorator(
    decorator: ast.expr,
    import_bindings: dict[str, str],
) -> str | None:
    """Resolve a decorator through its import declaration to ArrayBridge taxonomy."""

    decorator_func = decorator.func if isinstance(decorator, ast.Call) else decorator
    dotted_name = _dotted_expression_name(decorator_func)
    if dotted_name is None:
        return None
    local_root, *attributes = dotted_name.split(".")
    imported_root = import_bindings.get(local_root)
    if imported_root is None:
        return None
    declaration_path = ".".join((imported_root, *attributes))
    module_name, _, declaration_name = declaration_path.rpartition(".")
    if module_name not in _MEMORY_DECORATOR_IMPORT_MODULES:
        return None
    try:
        return MemoryType(declaration_name).value
    except ValueError:
        return None


class OpenHCSRegistry(LibraryRegistryBase):
    """
    Registry for OpenHCS native functions with explicit contract support.

    This registry processes OpenHCS functions that have been decorated with
    explicit contract declarations, allowing them to skip runtime testing
    while producing the same FunctionMetadata format as external libraries.
    """

    # Registry name for auto-registration
    _registry_name = "openhcs"

    # Required abstract class attributes
    MODULES_TO_SCAN = []  # Will be set dynamically
    MEMORY_TYPE = None  # OpenHCS functions have their own memory type attributes
    FLOAT_DTYPE = np.float32

    _registered_callables: WeakKeyDictionary[Callable[..., Any], Callable[..., Any]] = (
        WeakKeyDictionary()
    )
    _registered_callable_lock = RLock()

    def __init__(self):
        super().__init__("openhcs")
        self.MODULES_TO_SCAN: List[str] | None = None

    @staticmethod
    def _publish_metadata_claim(
        functions: Dict[str, FunctionMetadata],
        metadata: FunctionMetadata,
    ) -> None:
        """Publish one exact canonical name claim or reject duplicate owners."""

        existing = functions.get(metadata.name)
        if (
            existing is not None
            and existing.import_identity != metadata.import_identity
        ):
            raise ValueError(
                f"OpenHCS function name {metadata.name!r} is claimed by both "
                f"{existing.import_identity.import_path!r} and "
                f"{metadata.import_identity.import_path!r}."
            )
        functions[metadata.name] = metadata

    def _ensure_module_inventory(self) -> None:
        """Discover backend modules only when the full catalog is requested."""

        if self.MODULES_TO_SCAN is not None:
            return
        self.MODULES_TO_SCAN = self._get_openhcs_modules()

    def _get_openhcs_modules(self) -> List[str]:
        """Get list of OpenHCS processing modules to scan using automatic discovery."""
        import os
        import pkgutil

        modules = []

        # Get the backends directory path
        backends_path = os.path.dirname(__file__)  # lib_registry directory
        backends_path = os.path.dirname(backends_path)  # backends directory

        # Walk through all modules in openhcs.processing.backends recursively
        for importer, module_name, ispkg in pkgutil.walk_packages(
            [backends_path], "openhcs.processing.backends."
        ):
            # Skip lib_registry modules to avoid circular imports
            if "lib_registry" in module_name:
                continue

            # Skip __pycache__ and other non-module files
            if "__pycache__" in module_name:
                continue

            if module_name.rsplit(".", maxsplit=1)[-1].startswith("test_"):
                continue

            modules.append(module_name)

        return modules

    def get_modules_to_scan(self) -> List[Tuple[str, Any]]:
        """Get modules to scan for OpenHCS functions."""
        self._ensure_module_inventory()
        assert self.MODULES_TO_SCAN is not None
        modules = []
        allowed_memory_types = _allowed_openhcs_memory_types()
        for module_name in self.MODULES_TO_SCAN:
            if not _module_declares_allowed_memory_type(
                module_name,
                allowed_memory_types,
            ):
                logger.debug(
                    "Skipping OpenHCS module %s - no allowed memory decorators",
                    module_name,
                )
                continue
            try:
                module = import_module_preserving_root_logging(module_name)
                modules.append((module_name, module))
            except Exception as e:
                logger.warning(f"Could not import OpenHCS module {module_name}: {e}")
        return modules

    # ===== ESSENTIAL ABC METHODS =====
    def get_library_version(self) -> str:
        """Get OpenHCS version."""
        try:
            import openhcs

            return openhcs.__dict__.get("__version__", "unknown")
        except Exception:
            return "unknown"

    def cache_source_mtimes(self) -> Dict[str, float]:
        """Return scanned OpenHCS backend source mtimes without importing modules."""
        self._ensure_module_inventory()
        assert self.MODULES_TO_SCAN is not None
        source_mtimes = super().cache_source_mtimes()
        for module_name in self.MODULES_TO_SCAN:
            spec = importlib.util.find_spec(module_name)
            origin = spec.origin if spec is not None else None
            if origin is None:
                continue
            try:
                source_mtimes[f"{module_name}:{origin}"] = Path(origin).stat().st_mtime
            except OSError:
                continue
        return source_mtimes

    def cache_discovery_context(self) -> dict[str, Any]:
        """Project the declaration-import policy into catalogue cache identity."""

        allowed_memory_types = _allowed_openhcs_memory_types()
        return {
            "allowed_memory_types": (
                None if allowed_memory_types is None else sorted(allowed_memory_types)
            ),
            "installed_memory_types": sorted(
                memory_type.value
                for memory_type in MemoryType
                if memory_type.is_installed()
            ),
        }

    def is_library_available(self) -> bool:
        """OpenHCS is always available."""
        return True

    def load_or_discover_functions(self) -> Dict[str, FunctionMetadata]:
        """
        Load functions from cache or discover them, then add custom functions.

        Custom functions are NOT cached - they're loaded fresh from .py files
        each time and added to the result here.
        """
        for _, module in self.get_modules_to_scan():
            if isinstance(module, OpenHCSFunctionCatalogModule):
                module.openhcs_registry_functions()

        # Get module-based functions from cache or discovery
        functions = super().load_or_discover_functions()
        if not self.MODULES_TO_SCAN:
            return functions

        from openhcs.processing.custom_functions.runtime_registry import (
            CustomFunctionRuntimeRegistry,
        )

        for metadata in CustomFunctionRuntimeRegistry.metadata_by_name().values():
            if _catalog_memory_types(metadata.func) is None:
                continue
            existing = functions.get(metadata.name)
            if existing is not None:
                raise ValueError(
                    f"Custom function {metadata.name!r} conflicts with canonical "
                    f"OpenHCS function {existing.composite_key!r}."
                )
            self._publish_metadata_claim(functions, metadata)
            logger.debug(
                "Added custom function %r to OpenHCS registry",
                metadata.name,
            )

        return functions

    def get_library_object(self):
        """Return OpenHCS processing module."""
        import openhcs.processing

        return openhcs.processing

    def get_memory_type(self) -> str:
        """Return placeholder memory type."""
        return self.MEMORY_TYPE

    def get_display_name(self) -> str:
        """Get display name for OpenHCS."""
        return "OpenHCS"

    def public_projection_module(self, metadata: FunctionMetadata) -> None:
        """Native OpenHCS declarations already own their import modules."""

        del metadata
        return None

    def get_module_patterns(self) -> List[str]:
        """Get module patterns for OpenHCS."""
        return ["openhcs"]

    def discover_functions(self) -> Dict[str, FunctionMetadata]:
        """Discover OpenHCS functions with memory type decorators and assign default contracts."""
        functions = {}
        modules = self.get_modules_to_scan()
        catalog_functions = {
            module_name: module.openhcs_registry_functions()
            for module_name, module in modules
            if isinstance(module, OpenHCSFunctionCatalogModule)
        }
        catalog_owned_modules = {
            func.__module__
            for owned_functions in catalog_functions.values()
            for func in owned_functions
        }

        logger.info(
            f"🔍 OpenHCS Registry: Scanning {len(modules)} modules for functions with memory type decorators"
        )

        for module_name, module in modules:
            import inspect

            module_function_count = 0

            if isinstance(module, OpenHCSFunctionCatalogModule):
                for func in catalog_functions[module_name]:
                    metadata = self._catalog_metadata_for_function(
                        func.__name__,
                        func,
                        module_name,
                    )
                    if metadata is None:
                        continue
                    self._publish_metadata_claim(functions, metadata)
                    module_function_count += 1
                logger.debug(
                    f"  📦 {module_name}: Found {module_function_count} OpenHCS functions"
                )
                continue

            if module_name in catalog_owned_modules:
                logger.debug(
                    "Skipping %s because its functions are owned by a catalog module",
                    module_name,
                )
                continue

            for name, func in inspect.getmembers(module, inspect.isfunction):
                # Only include functions actually defined in this module (not imported)
                if func.__module__ != module_name:
                    logger.debug(
                        f"Skipping {name} from {module_name} - defined in {func.__module__}"
                    )
                    continue

                metadata = self._catalog_metadata_for_function(
                    name,
                    func,
                    module_name,
                )
                if metadata is None:
                    continue
                self._publish_metadata_claim(functions, metadata)
                module_function_count += 1

            logger.debug(
                f"  📦 {module_name}: Found {module_function_count} OpenHCS functions"
            )

        logger.info(
            f"✅ OpenHCS Registry: Discovered {len(functions)} module-based functions"
        )

        # NOTE: Custom functions are NOT loaded here to avoid circular dependency
        # They are loaded separately in func_registry.py Phase 4 after all registries are initialized
        # Custom declarations are projected separately by their runtime registry.

        return functions

    def _metadata_for_function(
        self,
        name: str,
        func,
        module_name: str,
    ) -> FunctionMetadata | None:
        declared = inspect.unwrap(func)
        if not inspect.isfunction(declared):
            return None
        callable_contract = CallableContract.from_callable(func)
        plate_scoped = (
            callable_contract.execution_scope is FunctionStepExecutionScope.PLATE
        )

        # Look for functions with memory type attributes (added by @numpy, @cupy, etc.)
        if (
            callable_contract.input_memory_type is None
            or callable_contract.output_memory_type is None
        ):
            if not plate_scoped:
                return None
            input_type = None
            output_type = None
        else:
            input_type = callable_contract.input_memory_type
            output_type = callable_contract.output_memory_type

        if not plate_scoped and (
            input_type not in VALID_MEMORY_TYPES
            or output_type not in VALID_MEMORY_TYPES
        ):
            logger.debug(
                f"Skipping {name} - invalid memory types: {input_type} -> {output_type}"
            )
            return None

        declared_memory_types = _catalog_memory_types(func)
        if declared_memory_types is None:
            logger.debug(
                "Skipping %s - declared framework roles are invalid, unavailable, "
                "or excluded by current catalog policy",
                name,
            )
            return None

        contract = self._processing_contract_for_function(
            callable_contract,
        )

        if not plate_scoped:
            # Attach nominal contract metadata for downstream authorities.
            vars(func)[FunctionContractAttribute.processing_contract] = contract

        # Apply the shared nominal registration wrapper once per declaration.
        # Catalog discovery may later request different metadata naming for the
        # same callable, but its runtime identity remains stable.
        with self._registered_callable_lock:
            wrapped_func = self._registered_callables.get(declared)
            if wrapped_func is None:
                wrapped_func = self.apply_contract_wrapper(func, contract)
                self._registered_callables[declared] = wrapped_func

        # Generate unique function name using module information
        unique_name = self._generate_function_name(name, module_name)

        # Extract full docstring, not just first line
        doc = self._extract_function_docstring(func)

        return FunctionMetadata(
            name=unique_name,
            func=wrapped_func,
            contract=contract,
            registry=self,
            module=declared.__module__,
            doc=doc,
            tags=self._generate_tags(module_name),
            original_name=declared.__name__,
            memory_type=input_type,
        )

    def _catalog_metadata_for_function(
        self,
        name: str,
        func: Callable[..., Any],
        module_name: str,
    ) -> FunctionMetadata | None:
        """Project metadata only for declarations on the public module surface."""

        declared = inspect.unwrap(func)
        if not is_declared_public_name(
            declared.__module__,
            name,
            declared,
        ):
            return None
        return self._metadata_for_function(
            name,
            func,
            module_name,
        )

    @classmethod
    def metadata_for_declared_callable(
        cls,
        func: Callable,
    ) -> FunctionMetadata | None:
        """Project one explicitly decorated OpenHCS declaration locally."""

        from openhcs.processing.custom_functions.runtime_registry import (
            CustomFunctionRuntimeRegistry,
        )

        custom_metadata = CustomFunctionRuntimeRegistry.metadata_for_callable(func)
        if custom_metadata is not None:
            return custom_metadata

        declared = inspect.unwrap(func)
        if not inspect.isfunction(declared):
            return None

        return cls()._metadata_for_function(
            func.__name__,
            func,
            _registry_catalog_module_for_callable(func),
        )

    def reconstruct_cached_callable(
        self,
        func: Callable,
        contract,
    ) -> Callable:
        """Reconstruct cached metadata through the declaration's one wrapper."""

        declared = inspect.unwrap(func)
        with self._registered_callable_lock:
            wrapped_func = self._registered_callables.get(declared)
            if wrapped_func is not None:
                return wrapped_func
            wrapped_func = super().reconstruct_cached_callable(func, contract)
            self._registered_callables[declared] = wrapped_func
            return wrapped_func

    def _processing_contract_for_function(
        self,
        callable_contract: CallableContract,
    ) -> ProcessingContract:
        """Return the function's declared contract, defaulting to FLEXIBLE."""
        declared_contract = callable_contract.processing_contract
        if isinstance(declared_contract, ProcessingContract):
            return declared_contract
        if isinstance(declared_contract, str):
            resolved = ProcessingContract.from_declared_name(declared_contract)
            if resolved is not None:
                return resolved

        declared_name = callable_contract.declared_processing_contract
        if isinstance(declared_name, str):
            resolved = ProcessingContract.from_declared_name(declared_name)
            if resolved is not None:
                return resolved

        # Most OpenHCS functions are FLEXIBLE when not explicitly declared.
        return ProcessingContract.FLEXIBLE

    def _generate_function_name(self, original_name: str, module_name: str) -> str:
        """Generate unique function name for OpenHCS functions."""
        # Extract meaningful part from module name
        if isinstance(module_name, str):
            module_parts = module_name.split(".")
            # Find meaningful part after 'backends'
            try:
                backends_idx = module_parts.index("backends")
                meaningful_parts = module_parts[backends_idx + 1 :]
                if meaningful_parts:
                    prefix = "_".join(meaningful_parts)
                    return f"{prefix}_{original_name}"
            except ValueError:
                pass

        return original_name

    def _generate_tags(self, module_name: str) -> List[str]:
        """Generate tags for OpenHCS functions."""
        tags = ["openhcs"]

        def add_tag(tag: str) -> None:
            if tag not in tags:
                tags.append(tag)

        # Add module-specific tags
        if isinstance(module_name, str):
            module_parts = module_name.split(".")
            for part in module_parts:
                if part not in {"openhcs", "processing", "backends"}:
                    add_tag(part)
            if "analysis" in module_parts:
                add_tag("analysis")
            if "preprocessing" in module_parts:
                add_tag("preprocessing")
            if "segmentation" in module_parts:
                add_tag("segmentation")

        return tags

    def _extract_function_docstring(self, func) -> str:
        """
        Extract the full docstring from a function, with proper formatting.

        Args:
            func: Function to extract docstring from

        Returns:
            Formatted docstring or empty string if none
        """
        if not func.__doc__:
            return ""

        # Get the full docstring
        docstring = func.__doc__.strip()

        # For UI display, we want a concise but informative description
        # Take the first paragraph (up to first double newline) or first 200 chars
        lines = docstring.split("\n")

        # Find the first non-empty line (summary)
        summary_lines = []
        for line in lines:
            line = line.strip()
            if not line and summary_lines:
                # Empty line after content - end of summary
                break
            if line:
                summary_lines.append(line)

        if summary_lines:
            summary = " ".join(summary_lines)
            # Limit length for UI display
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return summary

        return ""
