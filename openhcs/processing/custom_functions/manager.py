"""
Custom function manager for lifecycle operations.

Manages registration, persistence, loading, and deletion of custom functions.
All custom functions are stored in ~/.local/share/openhcs/custom_functions/
and automatically loaded on startup.

Architecture:
    - Uses exec() to execute user code in controlled namespace
    - Validates functions before and after execution
    - Registers valid functions via func_registry.register_function()
    - Persists to disk as .py files
    - Emits Qt signals for UI updates
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from openhcs.core.xdg_paths import get_data_file_path
from openhcs.processing.func_registry import register_function
from openhcs.processing.custom_functions.validation import (
    ValidationError,
    ValidationResult,
    validate_code,
    validate_function,
)
from openhcs.processing.custom_functions.signals import custom_function_signals

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CustomFunctionInfo:
    """
    Metadata for a registered custom function.

    Attributes:
        name: Function name
        file_path: Path to source .py file
        memory_type: Memory type (numpy, cupy, etc.)
        doc: Function docstring
    """

    name: str
    file_path: Path
    memory_type: str
    doc: str


class CustomFunctionManager:
    """
    Manager for custom function lifecycle operations.

    Handles registration, persistence, loading, and deletion of user-defined
    custom functions. All operations emit signals to notify UI components.

    Attributes:
        storage_dir: Directory where custom functions are stored
    """

    def __init__(self):
        """Initialize manager and create storage directory if needed."""
        self.storage_dir: Path = get_data_file_path("custom_functions")
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created custom functions directory: {self.storage_dir}")

    def register_from_code(
        self,
        code: str,
        persist: bool = True,
        *,
        clear_caches: bool = True,
        emit_signal: bool = True,
        collision_metadata: Dict[str, "FunctionMetadata"] | None = None,
    ) -> List[Callable]:
        """
        Execute code and register all decorated functions found.

        Validates code before execution, executes in controlled namespace,
        extracts decorated functions, validates them, and registers via
        func_registry.register_function().

        Args:
            code: Python code containing function definitions with decorators
            persist: If True, save code to storage directory

        Returns:
            List of successfully registered functions

        Raises:
            ValidationError: If code validation fails
            ValueError: If no valid functions found
            RuntimeError: If function registration fails
        """
        # Validate code before execution
        validation_result: ValidationResult = validate_code(code)
        if not validation_result.is_valid:
            error_messages = "\n".join(validation_result.errors)
            raise ValidationError(f"Code validation failed:\n{error_messages}")

        # Create controlled namespace with memory decorators
        namespace: Dict[str, Any] = self._create_execution_namespace()

        # Execute code
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValidationError(f"Code execution failed: {str(e)}")

        # Extract decorated functions
        registered_functions: List[Callable] = []
        for name, obj in namespace.items():
            # Skip special names and imports
            if name.startswith('_') or not callable(obj):
                continue

            # Check if this is a function with memory type attributes
            if not hasattr(obj, 'input_memory_type'):
                continue

            # Set module name for custom functions (required for code generation)
            # Custom functions executed via exec() don't have __module__ set properly
            if not hasattr(obj, '__module__') or obj.__module__ is None or obj.__module__ == '__main__':
                obj.__module__ = 'openhcs.processing.custom_functions'

            # Check for name collisions with existing OpenHCS functions
            self._check_name_collision(name, collision_metadata)

            # Validate function
            func_validation: ValidationResult = validate_function(obj)
            if not func_validation.is_valid:
                error_messages = "\n".join(func_validation.errors)
                raise ValidationError(
                    f"Function '{name}' validation failed:\n{error_messages}"
                )

            # Register function
            try:
                register_function(obj, backend='openhcs')
                registered_functions.append(obj)
                logger.info(
                    f"Registered custom function: {name} "
                    f"({obj.input_memory_type} -> {obj.output_memory_type})"
                )
            except ValueError as e:
                raise RuntimeError(f"Failed to register function '{name}': {str(e)}")

        # Ensure at least one function was registered
        if not registered_functions:
            raise ValueError(
                "No valid functions found in code. "
                "Functions must be decorated with @numpy, @cupy, etc."
            )

        # Persist to disk if requested
        if persist:
            for func in registered_functions:
                self._save_function_code(func.__name__, code)

        if clear_caches:
            self._clear_caches()

        if emit_signal:
            custom_function_signals.functions_changed.emit()

        return registered_functions

    def load_all_custom_functions(self) -> int:
        """
        Load all .py files from storage directory.

        Scans storage directory for .py files and registers all valid functions.
        Errors are logged but don't stop processing of other files.

        Returns:
            Number of functions successfully loaded
        """
        if not self.storage_dir.exists():
            return 0

        loaded_count: int = 0
        from openhcs.processing.backends.lib_registry.registry_service import RegistryService

        collision_metadata = RegistryService.get_all_functions_with_metadata()

        for py_file in self.storage_dir.glob("*.py"):
            try:
                code: str = py_file.read_text(encoding='utf-8')
                functions: List[Callable] = self.register_from_code(
                    code,
                    persist=False,  # Already persisted
                    clear_caches=False,
                    emit_signal=False,
                    collision_metadata=collision_metadata,
                )
                loaded_count += len(functions)
                logger.info(
                    f"Loaded {len(functions)} custom function(s) from {py_file.name}"
                )
            except Exception as e:
                logger.error(f"Failed to load custom functions from {py_file.name}: {e}")
                continue

        if loaded_count > 0:
            # Clear caches after bulk load
            self._clear_caches()
            custom_function_signals.functions_changed.emit()

        return loaded_count

    def load_custom_function(
        self,
        func_name: str,
        *,
        clear_caches: bool = True,
        emit_signal: bool = False,
    ) -> int:
        """
        Load one persisted custom function by name.

        Args:
            func_name: Name of the persisted custom function.
            clear_caches: Whether to clear function metadata caches after loading.
            emit_signal: Whether to emit the custom-functions-changed signal.

        Returns:
            Number of functions registered from the persisted file.
        """
        file_path: Path = self.storage_dir / f"{func_name}.py"
        if not file_path.exists():
            return 0

        code: str = file_path.read_text(encoding='utf-8')
        functions: List[Callable] = self.register_from_code(
            code,
            persist=False,
            clear_caches=clear_caches,
            emit_signal=emit_signal,
        )
        logger.info(
            "Loaded %d custom function(s) from %s",
            len(functions),
            file_path.name,
        )
        return len(functions)

    def delete_custom_function(self, func_name: str) -> bool:
        """
        Remove function from registry and delete source file.

        Note: This removes the file but does not unregister the function from
        FUNC_REGISTRY (requires application restart for full removal).

        Args:
            func_name: Name of function to delete

        Returns:
            True if function file was deleted, False if not found
        """
        file_path: Path = self.storage_dir / f"{func_name}.py"

        if not file_path.exists():
            logger.warning(f"Custom function file not found: {file_path}")
            return False

        # Delete file
        file_path.unlink()
        logger.info(f"Deleted custom function file: {file_path}")

        # Clear caches
        self._clear_caches()

        # Emit signal
        custom_function_signals.functions_changed.emit()

        return True

    def list_custom_functions(self) -> List[CustomFunctionInfo]:
        """
        Return metadata for all custom functions in storage.

        Returns:
            List of CustomFunctionInfo objects
        """
        if not self.storage_dir.exists():
            return []

        functions: List[CustomFunctionInfo] = []

        for py_file in self.storage_dir.glob("*.py"):
            try:
                # Read file to extract metadata
                code: str = py_file.read_text(encoding='utf-8')

                # Create temporary namespace to extract function
                namespace: Dict[str, Any] = self._create_execution_namespace()
                exec(code, namespace)

                # Find first decorated function
                for name, obj in namespace.items():
                    if (
                        not name.startswith('_')
                        and callable(obj)
                        and hasattr(obj, 'input_memory_type')
                    ):
                        info = CustomFunctionInfo(
                            name=name,
                            file_path=py_file,
                            memory_type=obj.input_memory_type,
                            doc=obj.__doc__ or ""
                        )
                        functions.append(info)
                        break

            except Exception as e:
                logger.error(f"Failed to read metadata from {py_file.name}: {e}")
                continue

        return functions

    def get_function_code(self, func_name: str) -> str:
        """
        Get source code for a custom function.

        Args:
            func_name: Name of function

        Returns:
            Python source code

        Raises:
            ValueError: If function file not found
        """
        file_path: Path = self.storage_dir / f"{func_name}.py"

        if not file_path.exists():
            raise ValueError(f"Custom function '{func_name}' not found")

        return file_path.read_text(encoding='utf-8')

    def update_custom_function(self, old_name: str, new_code: str) -> str:
        """
        Atomically update a custom function.

        Validates new code first. If valid, writes new file with temp name,
        renames temp to final location, then deletes old file if name changed.
        If any step fails, old function is preserved.

        Args:
            old_name: Name of existing function to replace
            new_code: New Python code

        Returns:
            Name of the new function (may differ if renamed)

        Raises:
            ValueError: If old function not found
            ValidationError: If new code is invalid
            OSError: If file operations fail
        """
        import ast
        import os
        import tempfile

        # Verify old function exists
        old_file_path: Path = self.storage_dir / f"{old_name}.py"
        if not old_file_path.exists():
            raise ValueError(f"Custom function '{old_name}' not found")

        # Validate new code FIRST (fail-loud if invalid)
        validation_result: ValidationResult = validate_code(new_code)
        if not validation_result.is_valid:
            error_messages = "\n".join(validation_result.errors)
            raise ValidationError(f"Code validation failed:\n{error_messages}")

        # Extract new function name
        tree = ast.parse(new_code)
        new_name: str = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                new_name = node.name
                break

        if not new_name:
            raise ValidationError("No function definition found in code")

        # Atomic operation using temp file pattern:
        # 1. Write new code to temp file
        # 2. Rename temp to final location (atomic filesystem operation)
        # 3. Delete old function file (only if name changed and new file exists)
        # 4. Load and register from the file that now definitely exists
        # This ensures new function exists before old is deleted

        new_file_path: Path = self.storage_dir / f"{new_name}.py"

        # Write to temp file first
        temp_fd, temp_path = tempfile.mkstemp(suffix='.py', dir=self.storage_dir)
        try:
            with os.fdopen(temp_fd, 'w') as f:
                f.write(new_code)

            # Rename temp to final location FIRST (atomic FS operation)
            # After this, new function file definitely exists
            Path(temp_path).rename(new_file_path)

            # Only THEN delete old function (if name changed)
            # New function already exists, so this is safe
            if old_name != new_name and old_file_path.exists():
                old_file_path.unlink()
                logger.info(f"Deleted old function file: {old_file_path}")

            # Load from file that now definitely exists
            # Don't use register_from_code() to avoid double-parsing
            self._load_function_from_file(new_file_path, new_name)

            return new_name

        except (OSError, FileNotFoundError, ValidationError) as e:
            # Cleanup temp file on failure (if it still exists)
            Path(temp_path).unlink(missing_ok=True)
            raise

    def _load_function_from_file(self, file_path: Path, func_name: str) -> None:
        """
        Load a custom function from an existing file without re-parsing code.

        Used by update_custom_function() to avoid double-parsing after file rename.

        Args:
            file_path: Path to the function file
            func_name: Name of the function

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If function can't be loaded
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Function file not found: {file_path}")

        # Read and register the code
        code: str = file_path.read_text(encoding='utf-8')

        try:
            self.register_from_code(code, persist=False)  # Already persisted
            logger.info(f"Loaded function '{func_name}' from {file_path}")
        except Exception as e:
            raise ValidationError(f"Failed to load function '{func_name}': {str(e)}")

    def _create_execution_namespace(self) -> Dict[str, Any]:
        """
        Create controlled namespace for exec().

        Includes all memory type decorators and common imports.
        Does not restrict builtins (custom functions need full Python).

        Returns:
            Namespace dict for exec()
        """
        from openhcs.core.memory import decorators

        # Import all memory type decorators
        namespace: Dict[str, Any] = {
            'numpy': decorators.numpy,
            'cupy': decorators.cupy,
            'torch': decorators.torch,
            'tensorflow': decorators.tensorflow,
            'jax': decorators.jax,
            'pyclesperanto': decorators.pyclesperanto,
        }

        return namespace

    def _check_name_collision(
        self,
        function_name: str,
        all_functions: Dict[str, "FunctionMetadata"] | None = None,
    ) -> None:
        """
        Check if a function name collides with existing OpenHCS functions.

        Args:
            function_name: Name of the custom function to check

        Raises:
            ValueError: If the function name collides with an existing function
        """
        if all_functions is None:
            from openhcs.processing.backends.lib_registry.registry_service import RegistryService

            all_functions = RegistryService.get_all_functions_with_metadata()

        # Check for collisions with non-custom functions
        for composite_key, metadata in all_functions.items():
            # Skip custom functions (they can override each other)
            if 'custom' in metadata.tags:
                continue

            # Check if name matches
            if metadata.name == function_name:
                raise ValueError(
                    f"Function name '{function_name}' collides with existing "
                    f"{metadata.registry.library_name} function. "
                    f"Please choose a different name for your custom function."
                )

    def _save_function_code(self, func_name: str, code: str) -> None:
        """
        Save function code to storage directory.

        Args:
            func_name: Name of function (used as filename)
            code: Python source code
        """
        file_path: Path = self.storage_dir / f"{func_name}.py"
        file_path.write_text(code, encoding='utf-8')
        logger.info(f"Saved custom function to: {file_path}")

    def _clear_caches(self) -> None:
        """
        Clear function registry metadata caches.

        Required after registration to ensure UI sees new functions.
        """
        from openhcs.processing.backends.lib_registry.registry_service import RegistryService

        # Clear RegistryService metadata cache
        RegistryService.clear_metadata_cache()
        logger.debug("Cleared RegistryService metadata cache")

        # Custom functions are loaded from FUNC_REGISTRY and are intentionally
        # not stored in the OpenHCSRegistry disk cache.
