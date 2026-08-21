"""
Registry Service - Clean function discovery and metadata access.

Provides unified access to all registry implementations with automatic discovery.
Follows OpenHCS generic solution principle - automatically adapts to new registries.
"""

import logging
import inspect
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict, Optional

from pyqt_reactive.process_launch import BackgroundProcessLaunchPolicy

from openhcs.utils.environment import OpenHCSProcessEnvironment
from .unified_registry import (
    FunctionMetadata,
    LIBRARY_REGISTRIES,
    LibraryRegistryBase,
    ProcessingContract,
)

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableImportIdentity
    from openhcs.core.function_reference import RegistryFunctionReference

logger = logging.getLogger(__name__)


RegistryPreparationCallback = Callable[[str], None]


class RegistryService:
    """
    Clean service for registry discovery and function metadata access.

    Automatically discovers all registry implementations and provides
    unified access to their functions with caching.
    """

    _metadata_cache: Optional[Dict[str, FunctionMetadata]] = None
    _registry_instances: tuple[LibraryRegistryBase, ...] | None = None
    _resolved_reference_callables: Dict[
        tuple[str, "CallableImportIdentity", str | None], Callable
    ] = {}
    _registry_inventory_lock = threading.RLock()

    @classmethod
    def get_all_functions_with_metadata(
        cls,
        *,
        status_callback: RegistryPreparationCallback | None = None,
    ) -> Dict[str, FunctionMetadata]:
        """Get unified metadata for all functions from all registries."""
        if cls._metadata_cache is not None:
            logger.debug(
                f"🎯 REGISTRY SERVICE: Using cached metadata ({len(cls._metadata_cache)} functions)"
            )
            return cls._metadata_cache

        emit_status = status_callback or logger.debug
        emit_status("Loading cached function catalog")
        registry_instances = cls._available_registry_instances()
        cached_functions = cls._load_valid_persistent_catalog(registry_instances)
        if cached_functions is None:
            cls._prepare_persistent_catalog(
                status_callback=emit_status,
            )
            emit_status("Loading the prepared function catalog")
            cached_functions = cls._load_valid_persistent_catalog(registry_instances)
        if cached_functions is None:
            raise RuntimeError(
                "Function registry preparation completed without producing a valid "
                "persistent catalog."
            )
        cls._metadata_cache = cached_functions
        emit_status(f"Function catalog ready ({len(cached_functions)} functions)")
        return cached_functions

    @classmethod
    def cached_metadata_snapshot(cls) -> Dict[str, FunctionMetadata]:
        """Return already-projected metadata without initiating preparation."""

        with cls._registry_inventory_lock:
            return {} if cls._metadata_cache is None else dict(cls._metadata_cache)

    @classmethod
    def prepare_in_current_process(cls) -> Dict[str, FunctionMetadata]:
        """Discover the complete catalog in this dedicated preparation process."""

        if cls._metadata_cache is not None:
            return cls._metadata_cache

        logger.debug(
            "🎯 REGISTRY SERVICE: Discovering functions from all registries..."
        )
        cls._metadata_cache = cls._metadata_from_instances(
            cls._available_registry_instances()
        )
        return cls._metadata_cache

    @classmethod
    def _load_valid_persistent_catalog(
        cls,
        registry_instances: list[LibraryRegistryBase] | None = None,
    ) -> Optional[Dict[str, FunctionMetadata]]:
        """Load every available registry only when all persistent caches are valid."""

        if registry_instances is None:
            registry_instances = cls._available_registry_instances()
        for registry_instance in registry_instances:
            if registry_instance.load_cached_functions() is None:
                return None
        return cls._metadata_from_instances(registry_instances)

    @classmethod
    def _available_registry_instances(cls) -> list[LibraryRegistryBase]:
        """Return the stable nominal registry inventory for this process."""

        with cls._registry_inventory_lock:
            if cls._registry_instances is not None:
                return list(cls._registry_instances)

            registry_instances = cls._discover_available_registry_instances()
            cls._registry_instances = tuple(registry_instances)
            return list(cls._registry_instances)

    @classmethod
    def _discover_available_registry_instances(cls) -> list[LibraryRegistryBase]:
        """Prove each registry once before admitting it to the catalog."""

        registry_instances: list[LibraryRegistryBase] = []
        registry_classes = list(LIBRARY_REGISTRIES.values())
        logger.debug(
            f"🎯 REGISTRY SERVICE: Found {len(registry_classes)} registered library registries"
        )

        for registry_class in registry_classes:
            if _cpu_only_mode_enabled() and not registry_class.supports_cpu_only():
                logger.info(
                    "CPU-only registry discovery skipping %s",
                    registry_class.__name__,
                )
                continue
            try:
                registry_instance = registry_class()
                if not registry_instance.is_available_for_catalog():
                    logger.warning(
                        "Library %s not available, skipping",
                        registry_instance.library_name,
                    )
                    continue
                registry_instances.append(registry_instance)
            except Exception as exc:
                logger.warning(
                    "Failed to load registry %s: %s",
                    registry_class.__name__,
                    exc,
                )
        return registry_instances

    @classmethod
    def _metadata_from_instances(
        cls,
        registry_instances: list[LibraryRegistryBase],
    ) -> Dict[str, FunctionMetadata]:
        """Project registry-owned declarations into the unified lookup."""

        all_functions = {}
        for registry_instance in registry_instances:
            logger.debug(
                "🎯 REGISTRY SERVICE: Calling load_or_discover_functions for %s",
                registry_instance.library_name,
            )
            functions = registry_instance.load_or_discover_functions()
            logger.debug(
                "🎯 REGISTRY SERVICE: Retrieved %d %s functions",
                len(functions),
                registry_instance.library_name,
            )

            # Use composite keys to prevent function name collisions between backends
            # Format: "backend:function_name" (e.g., "torch:stack_percentile_normalize")
            for metadata in functions.values():
                all_functions[metadata.composite_key] = metadata

        logger.info(f"Total functions discovered: {len(all_functions)}")
        return all_functions

    @classmethod
    def _prepare_persistent_catalog(
        cls,
        *,
        status_callback: RegistryPreparationCallback | None = None,
    ) -> None:
        """Run behavior probing in a dedicated interpreter main thread."""

        status_callback = status_callback or logger.debug
        status_callback("Discovering functions in an isolated execution process")

        policy = BackgroundProcessLaunchPolicy.current(detached=False)
        command = (
            policy.python_executable(sys.executable),
            "-m",
            "openhcs.runtime.zmq_execution_server_launcher",
            "--prepare-capabilities",
            "--log-level",
            "WARNING",
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            **policy.popen_arguments(),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "Function registry preparation process failed"
                + (f": {detail[-4000:]}" if detail else ".")
            )

    @classmethod
    def metadata_for_callable(
        cls,
        func: Callable,
    ) -> tuple[str, FunctionMetadata] | None:
        """Return a registry projection, preparing the catalog only as fallback."""

        local = cls.declared_metadata_for_callable(func)
        if local is not None:
            return local
        declared = inspect.unwrap(func)
        for composite_key, metadata in cls.get_all_functions_with_metadata().items():
            if inspect.unwrap(metadata.func) is declared:
                return composite_key, metadata
        return None

    @classmethod
    def declared_metadata_for_callable(
        cls,
        func: Callable,
    ) -> tuple[str, FunctionMetadata] | None:
        """Return cached or declaration-local metadata without catalog preparation.

        Registry wrappers preserve their declaration through ``__wrapped__``.
        Comparing that nominal callable identity keeps transport code blind to
        library names, generated registry keys, and module naming conventions.
        """

        declared = inspect.unwrap(func)
        cached_metadata = cls._metadata_cache
        if cached_metadata is not None:
            for composite_key, metadata in cached_metadata.items():
                if inspect.unwrap(metadata.func) is declared:
                    return composite_key, metadata

        local_metadata = cls._metadata_from_declared_owner(func)
        if local_metadata is not None:
            return local_metadata.composite_key, local_metadata

        return None

    @staticmethod
    def _metadata_from_declared_owner(func: Callable) -> FunctionMetadata | None:
        """Project one callable from a registry's declaration-local authority."""

        declared = inspect.unwrap(func)
        for registry_type in LibraryRegistryBase.loaded_registry_types():
            metadata = registry_type.metadata_for_declared_callable(func)
            if metadata is not None and inspect.unwrap(metadata.func) is declared:
                return metadata
        return None

    @classmethod
    def registered_callable(cls, func: Callable) -> Callable:
        """Project a declaration onto its registered runtime owner when present."""

        declared = inspect.unwrap(func)
        cached_metadata = cls._metadata_cache
        if cached_metadata is not None:
            for metadata in cached_metadata.values():
                if inspect.unwrap(metadata.func) is declared:
                    return metadata.func

        local_metadata = cls._metadata_from_declared_owner(func)
        if local_metadata is not None:
            return local_metadata.func

        return func

    @classmethod
    def resolve_function_reference(
        cls,
        reference: "RegistryFunctionReference",
    ) -> Callable:
        """Resolve one transported callable without preparing the global catalogue."""

        cache_key = (
            reference.composite_key,
            reference.import_identity,
            reference.declaration_revision,
        )
        with cls._registry_inventory_lock:
            cached = cls._resolved_reference_callables.get(cache_key)
            if cached is not None:
                return cached
            catalog_metadata = (
                None
                if cls._metadata_cache is None
                else cls._metadata_cache.get(reference.composite_key)
            )
            if catalog_metadata is not None:
                cls._validate_reference_metadata(reference, catalog_metadata)
                resolved = reference.require_current_declaration(catalog_metadata.func)
                cls._resolved_reference_callables[cache_key] = resolved
                return resolved

        try:
            registry_type = LIBRARY_REGISTRIES[reference.registry_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Function registry {reference.registry_name!r} is unavailable."
            ) from exc

        from openhcs.core.function_reference import (
            FunctionReferenceTransportAuthority,
        )

        declared = FunctionReferenceTransportAuthority.importable_function(
            reference.original_module,
            reference.function_name,
        )
        if not callable(declared):
            raise RuntimeError(
                f"Function {reference.original_module}.{reference.function_name} "
                "is not importable in this process."
            )
        declared = reference.require_current_declaration(declared)

        registry = registry_type()
        expected_composite_key = registry.composite_key_for_declared_callable(declared)
        if reference.composite_key != expected_composite_key:
            raise RuntimeError(
                f"Function reference {reference.composite_key!r} contradicts "
                f"declaration-owned identity {expected_composite_key!r}."
            )
        contract = reference.metadata.processing_contract
        if not isinstance(contract, ProcessingContract):
            contract = ProcessingContract.from_declared_name(
                reference.metadata.declared_processing_contract
            )
        if contract is None:
            metadata = registry.metadata_for_declared_callable(declared)
            if metadata is None:
                raise RuntimeError(
                    f"Function {reference.composite_key!r} has no transported "
                    "processing contract."
                )
            resolved = metadata.func
        else:
            resolved = registry.reconstruct_cached_callable(declared, contract)

        with cls._registry_inventory_lock:
            cls._resolved_reference_callables[cache_key] = resolved
        return resolved

    @staticmethod
    def _validate_reference_metadata(
        reference: "RegistryFunctionReference",
        metadata: FunctionMetadata,
    ) -> None:
        """Reject transported identity fields that contradict catalog authority."""

        if metadata.composite_key != reference.composite_key:
            raise RuntimeError(
                f"Catalog metadata {metadata.composite_key!r} does not match "
                f"reference {reference.composite_key!r}."
            )
        if metadata.import_identity != reference.import_identity:
            raise RuntimeError(
                f"Function reference import identity "
                f"{reference.import_identity.import_path!r} contradicts canonical "
                f"identity {metadata.import_identity.import_path!r}."
            )

    @classmethod
    def clear_metadata_cache(cls) -> None:
        """Clear cached metadata to force re-discovery."""
        with cls._registry_inventory_lock:
            cls._metadata_cache = None
            cls._registry_instances = None
            cls._resolved_reference_callables.clear()
        logger.info("Registry metadata cache cleared")


def _cpu_only_mode_enabled() -> bool:
    return OpenHCSProcessEnvironment.cpu_only_mode()
