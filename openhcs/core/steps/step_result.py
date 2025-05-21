"""
StepResult class for structured step outputs.

This module provides the StepResult class, which is the canonical interface for step outputs
in the OpenHCS pipeline architecture. It ensures that steps can communicate their results
and context updates in a structured, immutable way.

This class is used by pipeline.py as the only allowed return type for step execution.
Context updates and metadata are applied via ProcessingContext.update_from_step_result(),
not via .apply().

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 12 — Absolute Clean Execution
- Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
- Clause 17-B — Path Format Discipline
- Clause 21 — Context Immunity
- Clause 65 — No Fallback Logic
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 106-A — Declared Memory Types
- Clause 246 — Statelessness Mandate
- Clause 251 — Declarative Memory Conversion
- Clause 297 — StepResult is the only legal step return type
- Clause 311 — All declarative schemas are deprecated
- Clause 503 — Cognitive Load Transfer
- Clause 524 — Step identity and output are UID-bound
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openhcs.core.memory import MemoryWrapper


@dataclass(frozen=True)
class StepResult:
    """
    Immutable container for step execution results with memory type support.

    This class provides a clear structure for step results, separating normal processing
    results from context updates and storage operations. It ensures proper memory typing
    for all data.

    All steps must return a StepResult object. This is the only legal mechanism for steps
    to report output paths, forward context updates, and emit diagnostic metadata.

    Context updates and metadata are applied via ProcessingContext.update_from_step_result(),
    not via .apply().

    # Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
    # Clause 17-B — Path Format Discipline
    # Clause 66 — Immutability After Construction
    # Clause 88 — No Inferred Capabilities
    # Clause 106-A — Declared Memory Types
    # Clause 251 — Declarative Memory Conversion
    # Clause 297 — StepResult is the only legal step return type
    # Clause 311 — All declarative schemas are deprecated
    # Clause 503 — Cognitive Load Transfer
    # Clause 524 — Step identity and output are UID-bound

    Attributes:
        output_path: The primary output path of the step (as str or Path)
        context_updates: Updates to the processing context
        metadata: Diagnostic and metadata information
        results: Dictionary of normal processing results (should include memory-typed data)
        storage_operations: List of storage operations to perform
    """
    output_path: Optional[Union[str, Path]] = None
    context_updates: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    storage_operations: List[Tuple[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, *,
               output_path: Optional[Union[str, Path]] = None,
               context_updates: Optional[Dict[str, Any]] = None,
               metadata: Optional[Dict[str, Any]] = None,
               results: Optional[Dict[str, Any]] = None,
               storage_operations: Optional[List[Tuple[str, Any]]] = None) -> 'StepResult':
        """
        Create a new StepResult object.

        This factory method allows for creating StepResult objects with default values
        for missing fields, which is not possible with the frozen dataclass constructor.

        Args:
            output_path: The primary output path of the step (as str or Path)
            context_updates: Updates to the processing context
            metadata: Diagnostic and metadata information
            results: Dictionary of normal processing results
            storage_operations: List of storage operations to perform

        Returns:
            A new StepResult object

        # Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
        # Clause 17-B — Path Format Discipline
        # Clause 311 — All declarative schemas are deprecated
        """
        return cls(
            output_path=output_path,
            context_updates=context_updates or {},
            metadata=metadata or {},
            results=results or {},
            storage_operations=storage_operations or []
        )

    @classmethod
    def create_empty(cls) -> 'StepResult':
        """
        Create an empty step result.

        Returns:
            An empty step result
        """
        return cls()

    @classmethod
    def create_with_memory_typed_output(
        cls,
        *,
        output: Any,
        memory_type: str,
        output_path: Optional[Union[str, Path]] = None,
        context_updates: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'StepResult':
        """
        Create a new StepResult with memory-typed output.

        Args:
            output: The output data
            memory_type: The memory type for the output
            output_path: The primary output path of the step (as str or Path)
            context_updates: Updates to the processing context
            metadata: Diagnostic and metadata information

        Returns:
            A new StepResult with memory-typed output

        Raises:
            ValueError: If output is not compatible with the specified memory type

        # Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
        # Clause 17-B — Path Format Discipline
        # Clause 106-A — Declared Memory Types
        # Clause 311 — All declarative schemas are deprecated
        """
        # Always create a new MemoryWrapper with the declared memory type
        # No runtime type checking or conditional logic
        output_wrapper = MemoryWrapper(output, memory_type=memory_type)

        # Create metadata with memory type
        full_metadata = dict(metadata or {})
        full_metadata["memory_type"] = memory_type

        # Create results with memory-typed output
        results = {"output": output_wrapper}

        return cls(
            output_path=output_path,
            context_updates=context_updates or {},
            metadata=full_metadata,
            results=results,
            storage_operations=[]
        )



    def merge(self, other: 'StepResult') -> 'StepResult':
        """
        Merge another StepResult into a new StepResult.

        Args:
            other: Another StepResult object to merge

        Returns:
            A new StepResult with merged data

        Raises:
            TypeError: If other is not a StepResult
        """
        if not isinstance(other, StepResult):
            raise TypeError(f"Can only merge StepResult objects, got {type(other)}")

        # Merge context updates
        merged_context_updates = dict(self.context_updates)
        merged_context_updates.update(other.context_updates)

        # Merge metadata
        merged_metadata = dict(self.metadata)
        merged_metadata.update(other.metadata)

        # Merge results
        merged_results = dict(self.results)
        merged_results.update(other.results)

        # Merge storage operations
        merged_storage_operations = list(self.storage_operations)
        merged_storage_operations.extend(other.storage_operations)

        # Use the most recent output path
        output_path = other.output_path if other.output_path is not None else self.output_path

        return StepResult(
            output_path=output_path,
            context_updates=merged_context_updates,
            metadata=merged_metadata,
            results=merged_results,
            storage_operations=merged_storage_operations
        )

    def store(self, key: str, data: Any) -> 'StepResult':
        """
        Create a new StepResult with an additional storage operation.

        Args:
            key: The storage key
            data: The data to store

        Returns:
            A new StepResult with the additional storage operation
        """
        # Create a new list of storage operations with the additional operation
        new_storage_operations = list(self.storage_operations)
        new_storage_operations.append((key, data))

        # Create a new StepResult with the updated storage operations
        return StepResult(
            output_path=self.output_path,
            context_updates=self.context_updates,
            metadata=self.metadata,
            results=self.results,
            storage_operations=new_storage_operations
        )

    def add_storage_operation(self, key: str, data: Any) -> 'StepResult':
        """
        Create a new step result with an additional storage operation.
        Alias for store() method.

        Args:
            key: The storage key
            data: The data to store

        Returns:
            A new step result with the additional storage operation
        """
        return self.store(key, data)

    def update_context(self, key: str, value: Any) -> 'StepResult':
        """
        Create a new StepResult with an additional context update.

        Args:
            key: The context key
            value: The context value

        Returns:
            A new StepResult with the additional context update
        """
        # Create a new dict of context updates with the additional update
        new_context_updates = dict(self.context_updates)
        new_context_updates[key] = value

        # Create a new StepResult with the updated context updates
        return StepResult(
            output_path=self.output_path,
            context_updates=new_context_updates,
            metadata=self.metadata,
            results=self.results,
            storage_operations=self.storage_operations
        )

    def add_result(self, key: str, value: Any) -> 'StepResult':
        """
        Create a new StepResult with an additional result.

        Args:
            key: The result key
            value: The result value

        Returns:
            A new StepResult with the additional result
        """
        # Create a new dict of results with the additional result
        new_results = dict(self.results)
        new_results[key] = value

        # Create a new StepResult with the updated results
        return StepResult(
            output_path=self.output_path,
            context_updates=self.context_updates,
            metadata=self.metadata,
            results=new_results,
            storage_operations=self.storage_operations
        )

    def add_metadata(self, key: str, value: Any) -> 'StepResult':
        """
        Create a new StepResult with additional metadata.

        Args:
            key: The metadata key
            value: The metadata value

        Returns:
            A new StepResult with the additional metadata
        """
        # Create a new dict of metadata with the additional metadata
        new_metadata = dict(self.metadata)
        new_metadata[key] = value

        # Create a new StepResult with the updated metadata
        return StepResult(
            output_path=self.output_path,
            context_updates=self.context_updates,
            metadata=new_metadata,
            results=self.results,
            storage_operations=self.storage_operations
        )

    def with_output_path(self, output_path: Union[str, Path]) -> 'StepResult':
        """
        Create a new StepResult with a different output path.

        Args:
            output_path: The new output path (as str or Path)

        Returns:
            A new StepResult with the updated output path

        # Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
        # Clause 17-B — Path Format Discipline
        """
        # Create a new StepResult with the updated output path
        return StepResult(
            output_path=output_path,
            context_updates=self.context_updates,
            metadata=self.metadata,
            results=self.results,
            storage_operations=self.storage_operations
        )

    def with_context_updates(self, context_updates: Dict[str, Any]) -> 'StepResult':
        """
        Create a new step result with the given context updates.

        Args:
            context_updates: The context updates

        Returns:
            A new step result with the context updates
        """
        return StepResult(
            output_path=self.output_path,
            context_updates=context_updates,
            metadata=self.metadata,
            results=self.results,
            storage_operations=self.storage_operations
        )

    def with_metadata(self, metadata: Dict[str, Any]) -> 'StepResult':
        """
        Create a new step result with the given metadata.

        Args:
            metadata: The metadata

        Returns:
            A new step result with the metadata
        """
        return StepResult(
            output_path=self.output_path,
            context_updates=self.context_updates,
            metadata=metadata,
            results=self.results,
            storage_operations=self.storage_operations
        )

    def with_results(self, results: Dict[str, Any]) -> 'StepResult':
        """
        Create a new step result with the given results.

        Args:
            results: The results

        Returns:
            A new step result with the results
        """
        return StepResult(
            output_path=self.output_path,
            context_updates=self.context_updates,
            metadata=self.metadata,
            results=results,
            storage_operations=self.storage_operations
        )

    def with_storage_operations(self, storage_operations: List[Tuple[str, Any]]) -> 'StepResult':
        """
        Create a new step result with the given storage operations.

        Args:
            storage_operations: The storage operations

        Returns:
            A new step result with the storage operations
        """
        return StepResult(
            output_path=self.output_path,
            context_updates=self.context_updates,
            metadata=self.metadata,
            results=self.results,
            storage_operations=storage_operations
        )


