"""
OpenHCS native function registry.

This registry processes OpenHCS functions that have been decorated with
explicit contract declarations, allowing them to skip runtime testing
while producing the same FunctionMetadata format as external libraries.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib

from openhcs.constants import MemoryType, VALID_MEMORY_TYPES
from openhcs.core.callable_contract import CallableContract
from openhcs.processing.backends.lib_registry.unified_registry import LibraryRegistryBase, FunctionMetadata

logger = logging.getLogger(__name__)


class OpenHCSRegistry(LibraryRegistryBase):
    """
    Registry for OpenHCS native functions with explicit contract support.

    This registry processes OpenHCS functions that have been decorated with
    explicit contract declarations, allowing them to skip runtime testing
    while producing the same FunctionMetadata format as external libraries.
    """

    # Registry name for auto-registration
    _registry_name = 'openhcs'

    # Required abstract class attributes
    MODULES_TO_SCAN = []  # Will be set dynamically
    MEMORY_TYPE = None  # OpenHCS functions have their own memory type attributes
    FLOAT_DTYPE = np.float32

    def __init__(self):
        super().__init__("openhcs")
        # Set modules to scan to OpenHCS processing modules
        self.MODULES_TO_SCAN = self._get_openhcs_modules()

    def _get_openhcs_modules(self) -> List[str]:
        """Get list of OpenHCS processing modules to scan using automatic discovery."""
        import pkgutil
        import os

        modules = []

        # Get the backends directory path
        backends_path = os.path.dirname(__file__)  # lib_registry directory
        backends_path = os.path.dirname(backends_path)  # backends directory

        # Walk through all modules in openhcs.processing.backends recursively
        for importer, module_name, ispkg in pkgutil.walk_packages(
            [backends_path],
            "openhcs.processing.backends."
        ):
            # Skip lib_registry modules to avoid circular imports
            if "lib_registry" in module_name:
                continue

            # Skip __pycache__ and other non-module files
            if "__pycache__" in module_name:
                continue

            modules.append(module_name)

        return modules

    def get_modules_to_scan(self) -> List[Tuple[str, Any]]:
        """Get modules to scan for OpenHCS functions."""
        modules = []
        for module_name in self.MODULES_TO_SCAN:
            try:
                module = importlib.import_module(module_name)
                modules.append((module_name, module))
            except Exception as e:
                logger.warning(f"Could not import OpenHCS module {module_name}: {e}")
        return modules



    # ===== ESSENTIAL ABC METHODS =====
    def get_library_version(self) -> str:
        """Get OpenHCS version."""
        try:
            import openhcs
            return openhcs.__dict__.get('__version__', 'unknown')
        except:
            return 'unknown'

    def cache_source_mtimes(self) -> Dict[str, float]:
        """Return scanned OpenHCS backend source mtimes without importing modules."""
        source_mtimes: Dict[str, float] = {}
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

    def is_library_available(self) -> bool:
        """OpenHCS is always available."""
        return True

    def load_or_discover_functions(self) -> Dict[str, FunctionMetadata]:
        """
        Load functions from cache or discover them, then add custom functions.

        Custom functions are NOT cached - they're loaded fresh from .py files
        each time and added to the result here.
        """
        # Get module-based functions from cache or discovery
        functions = super().load_or_discover_functions()

        # Add custom functions from FUNC_REGISTRY
        # Custom functions are registered via register_function() when loaded from .py files
        from openhcs.processing.func_registry import FUNC_REGISTRY

        custom_funcs = FUNC_REGISTRY.get('openhcs', [])
        for func in custom_funcs:
            # Check if this function has metadata (custom functions have __function_metadata__)
            if hasattr(func, '__function_metadata__'):
                metadata = func.__function_metadata__
                # Only add if not already in functions (avoid duplicates)
                if metadata.name not in functions:
                    functions[metadata.name] = metadata
                    logger.debug(f"Added custom function '{metadata.name}' to OpenHCS registry")

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

    def get_module_patterns(self) -> List[str]:
        """Get module patterns for OpenHCS."""
        return ["openhcs"]



    def discover_functions(self) -> Dict[str, FunctionMetadata]:
        """Discover OpenHCS functions with memory type decorators and assign default contracts."""
        from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

        functions = {}
        modules = self.get_modules_to_scan()

        logger.info(f"🔍 OpenHCS Registry: Scanning {len(modules)} modules for functions with memory type decorators")

        for module_name, module in modules:
            import inspect
            module_function_count = 0

            for name, func in inspect.getmembers(module, inspect.isfunction):
                # Only include functions actually defined in this module (not imported)
                if func.__module__ != module_name:
                    logger.debug(f"Skipping {name} from {module_name} - defined in {func.__module__}")
                    continue

                callable_contract = CallableContract.from_callable(func)

                # Look for functions with memory type attributes (added by @numpy, @cupy, etc.)
                if (
                    callable_contract.input_memory_type is not None
                    and callable_contract.output_memory_type is not None
                ):
                    input_type = callable_contract.input_memory_type
                    output_type = callable_contract.output_memory_type

                    if input_type not in VALID_MEMORY_TYPES or output_type not in VALID_MEMORY_TYPES:
                        logger.debug(f"Skipping {name} - invalid memory types: {input_type} -> {output_type}")
                        continue

                    # Check if function's backend is available before including it
                    if not self._is_function_backend_available(input_type):
                        logger.debug(f"Skipping {name} - backend not available")
                        continue

                    contract = self._processing_contract_for_function(
                        callable_contract,
                        ProcessingContract,
                    )

                    # Add the contract attribute so other parts of the system can find it
                    func.__processing_contract__ = contract

                    # Apply contract wrapper (adds slice_by_slice for FLEXIBLE)
                    wrapped_func = self.apply_contract_wrapper(func, contract)

                    # Override the function in the module with the wrapped version
                    # This ensures that imports from the module get the wrapped version with 'enabled'
                    setattr(module, name, wrapped_func)

                    # Generate unique function name using module information
                    unique_name = self._generate_function_name(name, module_name)

                    # Extract full docstring, not just first line
                    doc = self._extract_function_docstring(func)

                    metadata = FunctionMetadata(
                        name=unique_name,
                        func=wrapped_func,
                        contract=contract,
                        registry=self,
                        module=func.__module__ or "",
                        doc=doc,
                        tags=self._generate_tags(module_name),
                        original_name=name
                    )

                    functions[unique_name] = metadata
                    module_function_count += 1

            logger.debug(f"  📦 {module_name}: Found {module_function_count} OpenHCS functions")

        logger.info(f"✅ OpenHCS Registry: Discovered {len(functions)} module-based functions")

        # NOTE: Custom functions are NOT loaded here to avoid circular dependency
        # They are loaded separately in func_registry.py Phase 4 after all registries are initialized
        # Custom functions will be registered via register_function() which wraps them with contracts

        return functions

    def _processing_contract_for_function(self, callable_contract, ProcessingContract):
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
            module_parts = module_name.split('.')
            # Find meaningful part after 'backends'
            try:
                backends_idx = module_parts.index('backends')
                meaningful_parts = module_parts[backends_idx+1:]
                if meaningful_parts:
                    prefix = '_'.join(meaningful_parts)
                    return f"{prefix}_{original_name}"
            except ValueError:
                pass
        
        return original_name

    def _generate_tags(self, module_name: str) -> List[str]:
        """Generate tags for OpenHCS functions."""
        tags = ['openhcs']

        def add_tag(tag: str) -> None:
            if tag not in tags:
                tags.append(tag)
        
        # Add module-specific tags
        if isinstance(module_name, str):
            module_parts = module_name.split('.')
            for part in module_parts:
                if part not in {'openhcs', 'processing', 'backends'}:
                    add_tag(part)
            if 'analysis' in module_parts:
                add_tag('analysis')
            if 'preprocessing' in module_parts:
                add_tag('preprocessing')
            if 'segmentation' in module_parts:
                add_tag('segmentation')
        
        return tags

    def _is_function_backend_available(self, memory_type: str | None) -> bool:
        """
        Check if the function's backend is available.

        For OpenHCS functions with mixed backends, we need to check each function
        individually based on its declared memory type.

        Args:
            memory_type: Function input memory type to check

        Returns:
            True if the function's backend is available, False otherwise
        """
        if not memory_type:
            # If no memory type specified, assume numpy (always available)
            return True

        # Check backend availability based on memory type
        return self._check_backend_availability(memory_type)

    def _check_backend_availability(self, memory_type: str) -> bool:
        """
        Check if a specific backend/memory type is available using the registry system.

        This uses the canonical LIBRARY_REGISTRIES as the source of truth for backend availability,
        avoiding hardcoded checks and ensuring consistency across the system.

        Args:
            memory_type: Memory type to check (e.g., "cupy", "torch", "numpy", "pyclesperanto")

        Returns:
            True if backend is available, False otherwise
        """
        # Import canonical registry dict (auto-discovers on first access)
        from openhcs.processing.backends.lib_registry.unified_registry import LIBRARY_REGISTRIES

        # Special case: numpy is always available (no dedicated registry)
        if memory_type == MemoryType.NUMPY.value:
            return True

        # Use canonical registry system - LIBRARY_REGISTRIES auto-discovers on access
        try:
            if memory_type not in LIBRARY_REGISTRIES:
                logger.debug(f"No registry found for memory type: {memory_type}")
                return False

            # Get registry class and instantiate it
            registry_class = LIBRARY_REGISTRIES[memory_type]
            registry_instance = registry_class()

            # Use the registry's own availability check as source of truth
            return registry_instance.is_library_available()

        except Exception as e:
            logger.warning(f"Failed to check backend availability for {memory_type}: {e}")
            return False

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
        lines = docstring.split('\n')

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
            summary = ' '.join(summary_lines)
            # Limit length for UI display
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return summary

        return ""
