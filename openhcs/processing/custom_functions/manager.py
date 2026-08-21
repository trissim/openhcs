"""
Custom function manager for lifecycle operations.

Manages registration, persistence, loading, and deletion of custom functions.
All custom functions are stored in ~/.local/share/openhcs/custom_functions/
and automatically loaded on startup.

Architecture:
    - Uses exec() to execute user code in controlled namespace
    - Validates functions before and after execution
    - Projects valid declarations through the custom runtime registry
    - Persists to disk as .py files
    - Emits Qt signals for UI updates
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from arraybridge import MemoryType

from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.xdg_paths import get_data_file_path
from openhcs.processing.custom_functions.runtime_registry import (
    CustomFunctionRuntimeRegistry,
    project_custom_function,
)
from openhcs.processing.custom_functions.validation import (
    ValidationError,
    validate_code,
    validate_function,
)
from openhcs.processing.custom_functions.signals import custom_function_signals

if TYPE_CHECKING:
    from openhcs.processing.backends.lib_registry.unified_registry import (
        FunctionMetadata,
    )

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


@dataclass(frozen=True, slots=True)
class CustomFunctionSource:
    """Content identity of one persisted custom-function declaration."""

    function_name: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class CustomFunctionSourceRevision:
    """Exact persisted source set owned by ``CustomFunctionManager``."""

    sources: tuple[CustomFunctionSource, ...]

    @property
    def function_names(self) -> frozenset[str]:
        """Return declaration names derived from the manager's file convention."""

        return frozenset(source.function_name for source in self.sources)


@dataclass(frozen=True, slots=True)
class CustomFunctionSourceSnapshot:
    """Exact source bytes decoded for preparation with their content proof."""

    source: CustomFunctionSource
    code: str


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
    ) -> list[Callable]:
        """
        Validate and register the single declaration owned by this source.

        Validates code before execution, executes in controlled namespace,
        extracts decorated functions, validates them, and projects their
        runtime metadata through the custom-function owner.

        Args:
            code: Python code containing function definitions with decorators
            persist: If True, save code to storage directory

        Returns:
            A one-item list containing the registered runtime callable

        Raises:
            ValidationError: If code validation fails
            ValueError: If no valid functions found
            RuntimeError: If function registration fails
        """
        metadata = self._prepare_source(code)
        with CustomFunctionRuntimeRegistry.lifecycle():
            if persist:
                source_path = self.source_path_for_function(metadata.func)
                if source_path.exists():
                    raise ValueError(
                        f"Custom function '{metadata.original_name}' already exists; "
                        "use update_custom_function() to replace it."
                    )
                CustomFunctionRuntimeRegistry.ensure_can_publish(metadata)
                self._save_function_code(metadata.original_name, code)
            CustomFunctionRuntimeRegistry.publish(metadata)

        registered_functions = [metadata.func]
        contract = CallableContract.from_callable(metadata.func)
        logger.info(
            "Registered custom function: %s (%s -> %s)",
            metadata.original_name,
            contract.input_memory_type,
            contract.output_memory_type,
        )

        if clear_caches:
            self._clear_caches()

        if emit_signal:
            custom_function_signals.functions_changed.emit()

        return registered_functions

    def source_path_for_function(self, func: Callable) -> Path:
        """Return the persisted source path used for a registered function."""
        return self.storage_dir / f"{func.__name__}.py"

    def source_revision(self) -> CustomFunctionSourceRevision:
        """Return a content-derived revision of all persisted declarations."""

        sources = tuple(
            CustomFunctionSource(
                function_name=source_path.stem,
                content_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            for source_path in sorted(self.storage_dir.glob("*.py"))
        )
        return CustomFunctionSourceRevision(sources=sources)

    def load_all_custom_functions(self) -> int:
        """
        Load all .py files from storage directory.

        Prepare every persisted declaration, then publish the exact set once.
        Any invalid source fails the reconciliation without disturbing the
        previously published runtime view.

        Returns:
            Number of functions successfully loaded
        """
        if not self.storage_dir.exists():
            return 0

        revision, snapshots = self._snapshot_all_sources()
        with CustomFunctionRuntimeRegistry.lifecycle():
            if revision == CustomFunctionRuntimeRegistry.source_revision():
                return len(revision.sources)

        prepared = {}
        for snapshot in snapshots:
            metadata = CustomFunctionRuntimeRegistry.prepare_source_once(
                snapshot.source.function_name,
                snapshot.source.content_sha256,
                lambda snapshot=snapshot: self._prepare_source(snapshot.code),
            )
            if metadata.original_name != snapshot.source.function_name:
                raise ValidationError(
                    f"Persisted custom function "
                    f"'{snapshot.source.function_name}.py' declares "
                    f"'{metadata.original_name}'. The filename and declaration "
                    "name must match."
                )
            if metadata.original_name in prepared:
                raise ValidationError(
                    f"Duplicate persisted custom function "
                    f"'{metadata.original_name}'."
                )
            prepared[metadata.original_name] = metadata

        with CustomFunctionRuntimeRegistry.lifecycle():
            self._require_revision(revision)
            CustomFunctionRuntimeRegistry.replace_all(prepared, revision)

        self._clear_caches()
        if prepared:
            custom_function_signals.functions_changed.emit()
        logger.info("Loaded %d persisted custom function(s)", len(prepared))
        return len(prepared)

    def load_custom_function(
        self,
        func_name: str,
        *,
        clear_caches: bool = True,
        emit_signal: bool = False,
        publish_only_if_missing: bool = False,
    ) -> int:
        """
        Load one persisted custom function by name.

        Args:
            func_name: Name of the persisted custom function.
            clear_caches: Whether to clear function metadata caches after loading.
            emit_signal: Whether to emit the custom-functions-changed signal.
            publish_only_if_missing: Reuse an exact export published by a
                concurrent lazy loader instead of replacing its identity.

        Returns:
            Number of functions registered from the persisted file.
        """
        file_path: Path = self.storage_dir / f"{func_name}.py"
        snapshot = self._snapshot_source(file_path)
        if snapshot is None:
            return 0

        metadata = CustomFunctionRuntimeRegistry.prepare_source_once(
            snapshot.source.function_name,
            snapshot.source.content_sha256,
            lambda: self._prepare_source(snapshot.code),
        )
        if metadata.original_name != func_name:
            raise ValidationError(
                f"Persisted custom function '{file_path.name}' declares "
                f"'{metadata.original_name}'. The filename and declaration "
                "name must match."
            )

        with CustomFunctionRuntimeRegistry.lifecycle():
            self._require_snapshot(file_path, snapshot)
            if (
                publish_only_if_missing
                and CustomFunctionRuntimeRegistry.owns_published_export(func_name)
            ):
                return 1
            CustomFunctionRuntimeRegistry.publish(metadata)

        functions = [metadata.func]
        if clear_caches:
            self._clear_caches()
        if emit_signal:
            custom_function_signals.functions_changed.emit()
        logger.info(
            "Loaded %d custom function(s) from %s",
            len(functions),
            file_path.name,
        )
        return len(functions)

    def delete_custom_function(self, func_name: str) -> bool:
        """
        Remove function from registry and delete source file.

        The process-local runtime projection is removed with the persisted source.

        Args:
            func_name: Name of function to delete

        Returns:
            True if function file was deleted, False if not found
        """
        file_path: Path = self.storage_dir / f"{func_name}.py"

        with CustomFunctionRuntimeRegistry.lifecycle():
            if not file_path.exists():
                logger.warning("Custom function file not found: %s", file_path)
                return False
            file_path.unlink()
            CustomFunctionRuntimeRegistry.remove(func_name)
        logger.info("Deleted custom function file: %s", file_path)

        # Clear caches
        self._clear_caches()

        # Emit signal
        custom_function_signals.functions_changed.emit()

        return True

    def list_custom_functions(self) -> list[CustomFunctionInfo]:
        """
        Return metadata for all custom functions in storage.

        Returns:
            List of CustomFunctionInfo objects
        """
        if not self.storage_dir.exists():
            return []

        functions: list[CustomFunctionInfo] = []

        for py_file in sorted(self.storage_dir.glob("*.py")):
            try:
                metadata = self._prepare_source(py_file.read_text(encoding="utf-8"))
                contract = CallableContract.from_callable(metadata.func)
                if contract.input_memory_type is None:
                    raise ValidationError(
                        f"Custom function '{metadata.original_name}' does not "
                        "declare an input memory type."
                    )
                functions.append(
                    CustomFunctionInfo(
                        name=metadata.original_name,
                        file_path=py_file,
                        memory_type=contract.input_memory_type,
                        doc=metadata.func.__doc__ or "",
                    )
                )

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

        return file_path.read_text(encoding="utf-8")

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
        old_file_path = self.storage_dir / f"{old_name}.py"
        old_snapshot = self._snapshot_source(old_file_path)
        if old_snapshot is None:
            raise ValueError(f"Custom function '{old_name}' not found")

        metadata = self._prepare_source(new_code)
        new_name = metadata.original_name
        new_file_path = self.storage_dir / f"{new_name}.py"
        temp_path = self._write_temporary_source(new_code)
        try:
            with CustomFunctionRuntimeRegistry.lifecycle():
                self._require_snapshot(old_file_path, old_snapshot)
                runtime_names = CustomFunctionRuntimeRegistry.metadata_by_name()
                if new_name != old_name and (
                    new_file_path.exists() or new_name in runtime_names
                ):
                    raise ValueError(
                        f"Custom function '{new_name}' already exists; rename aborted."
                    )
                CustomFunctionRuntimeRegistry.ensure_can_replace(old_name, metadata)

                os.replace(temp_path, new_file_path)
                if new_name != old_name:
                    try:
                        old_file_path.unlink()
                    except OSError:
                        new_file_path.unlink(missing_ok=True)
                        raise
                CustomFunctionRuntimeRegistry.replace(old_name, metadata)
        finally:
            temp_path.unlink(missing_ok=True)

        self._clear_caches()
        custom_function_signals.functions_changed.emit()
        return new_name

    def _create_execution_namespace(self) -> dict[str, Any]:
        """
        Create controlled namespace for exec().

        Includes all memory type decorators and common imports.
        Does not restrict builtins (custom functions need full Python).

        Returns:
            Namespace dict for exec()
        """
        from openhcs.core.memory import decorators

        return {
            "__name__": "openhcs.processing.custom_functions",
            **{
                memory_type.value: getattr(decorators, memory_type.value)
                for memory_type in MemoryType
            },
        }

    def _prepare_source(
        self,
        code: str,
    ) -> "FunctionMetadata":
        """Validate and project one source without mutating runtime or disk state."""

        validation_result = validate_code(code)
        if not validation_result.is_valid:
            raise ValidationError(
                "Code validation failed:\n" + "\n".join(validation_result.errors)
            )

        namespace = self._create_execution_namespace()
        try:
            exec(code, namespace)
        except Exception as exc:
            raise ValidationError(f"Code execution failed: {exc}") from exc

        declared_names = set(validation_result.function_names)
        declarations = [
            (obj, CallableMetadata.from_callable(obj))
            for name, obj in namespace.items()
            if name in declared_names and not name.startswith("_") and callable(obj)
        ]
        declarations = [
            (declaration, metadata)
            for declaration, metadata in declarations
            if any(
                memory_type is not None
                for memory_type in (
                    metadata.input_memory_type,
                    metadata.output_memory_type,
                    metadata.execution_memory_type,
                )
            )
        ]
        if len(declarations) != 1:
            raise ValidationError(
                "Each custom-function source must declare exactly one decorated "
                f"processing function; found {len(declarations)}."
            )

        declaration, _metadata = declarations[0]
        self._check_name_collision(declaration.__name__)
        function_validation = validate_function(declaration)
        if not function_validation.is_valid:
            raise ValidationError(
                f"Function '{declaration.__name__}' validation failed:\n"
                + "\n".join(function_validation.errors)
            )
        try:
            return project_custom_function(
                declaration,
                declaration_revision=hashlib.sha256(code.encode("utf-8")).hexdigest(),
            )
        except ValueError as exc:
            raise ValidationError(
                f"Function '{declaration.__name__}' projection failed: {exc}"
            ) from exc

    def _check_name_collision(
        self,
        function_name: str,
    ) -> None:
        """Reject a claim already proven by the loaded canonical catalog."""

        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        composite_key = f"openhcs:{function_name}"
        metadata = RegistryService.cached_metadata_snapshot().get(composite_key)
        if metadata is not None and "custom" not in metadata.tags:
            raise ValueError(
                f"Function name {function_name!r} collides with a canonical "
                "OpenHCS function. Please choose a different name."
            )

    def _save_function_code(self, func_name: str, code: str) -> None:
        """
        Save function code to storage directory.

        Args:
            func_name: Name of function (used as filename)
            code: Python source code
        """
        file_path = self.storage_dir / f"{func_name}.py"
        temp_path = self._write_temporary_source(code)
        try:
            os.replace(temp_path, file_path)
        finally:
            temp_path.unlink(missing_ok=True)
        logger.info(f"Saved custom function to: {file_path}")

    def _snapshot_source(
        self,
        source_path: Path,
    ) -> CustomFunctionSourceSnapshot | None:
        """Read one exact persisted source without executing it."""

        try:
            source_bytes = source_path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            code = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                f"Persisted custom function '{source_path.name}' is not UTF-8."
            ) from exc
        return CustomFunctionSourceSnapshot(
            source=CustomFunctionSource(
                function_name=source_path.stem,
                content_sha256=hashlib.sha256(source_bytes).hexdigest(),
            ),
            code=code,
        )

    def _snapshot_all_sources(
        self,
    ) -> tuple[
        CustomFunctionSourceRevision,
        tuple[CustomFunctionSourceSnapshot, ...],
    ]:
        """Read the persisted source set that one reconciliation will prepare."""

        snapshots = []
        for source_path in sorted(self.storage_dir.glob("*.py")):
            snapshot = self._snapshot_source(source_path)
            if snapshot is None:
                raise ValidationError(
                    "Persisted custom-function sources changed while being read; "
                    "retry reconciliation."
                )
            snapshots.append(snapshot)
        revision = CustomFunctionSourceRevision(
            sources=tuple(snapshot.source for snapshot in snapshots)
        )
        return revision, tuple(snapshots)

    def _require_snapshot(
        self,
        source_path: Path,
        expected: CustomFunctionSourceSnapshot,
    ) -> None:
        """Fail loudly when one prepared source no longer owns the same bytes."""

        current = self._snapshot_source(source_path)
        if current is None or current.source != expected.source:
            raise ValidationError(
                f"Persisted custom function '{source_path.name}' changed during "
                "preparation; retry the lifecycle operation."
            )

    def _require_revision(self, expected: CustomFunctionSourceRevision) -> None:
        """Fail loudly unless the prepared persisted source set is still exact."""

        try:
            current = self.source_revision()
        except FileNotFoundError as exc:
            raise ValidationError(
                "Persisted custom-function sources changed during preparation; "
                "retry reconciliation."
            ) from exc
        if current != expected:
            raise ValidationError(
                "Persisted custom-function sources changed during preparation; "
                "retry reconciliation."
            )

    def _write_temporary_source(self, code: str) -> Path:
        """Write source beside its destination for an atomic replacement."""

        file_descriptor, raw_path = tempfile.mkstemp(
            suffix=".py",
            dir=self.storage_dir,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                stream.write(code)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            Path(raw_path).unlink(missing_ok=True)
            raise
        return Path(raw_path)

    def _clear_caches(self) -> None:
        """
        Clear function registry metadata caches.

        Required after registration to ensure UI sees new functions.
        """
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        # Clear RegistryService metadata cache
        RegistryService.clear_metadata_cache()
        logger.debug("Cleared RegistryService metadata cache")

        # Custom functions are process-local projections and are intentionally
        # not stored in the OpenHCSRegistry disk cache.
