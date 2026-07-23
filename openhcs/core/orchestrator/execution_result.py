"""
Execution result types for pipeline orchestration.

This module defines typed execution results to replace dict-based results,
following OpenHCS standards for explicit contracts and type safety.
"""

import copyreg
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.runtime_stores import StoredRuntimeValue


class RuntimeExecutionTransportSerialization:
    """Pickle reducers required for worker-to-parent execution results."""

    @staticmethod
    def mapping_proxy_from_dict(value: dict) -> MappingProxyType:
        """Rebuild an immutable mapping view from transport state."""
        return MappingProxyType(value)

    @staticmethod
    def reduce_mapping_proxy(
        value: MappingProxyType,
    ) -> tuple[object, tuple[dict]]:
        """Serialize immutable mapping views as immutable mapping views."""
        return RuntimeExecutionTransportSerialization.mapping_proxy_from_dict, (
            dict(value),
        )

    @classmethod
    def register(cls) -> None:
        """Install execution-result transport reducers in this interpreter."""
        copyreg.pickle(type(MappingProxyType({})), cls.reduce_mapping_proxy)


RuntimeExecutionTransportSerialization.register()


class ExecutionStatus(Enum):
    """Status of pipeline execution for an axis or combination."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RuntimeContextObservation:
    """Runtime records produced for one compiled execution context."""

    context_key: str
    records: tuple[StoredRuntimeValue, ...]


@dataclass(frozen=True)
class RuntimeExecutionObservation:
    """Runtime records returned from worker-owned execution contexts."""

    contexts: tuple[RuntimeContextObservation, ...] = field(default_factory=tuple)

    def merge_into(self, execution_contexts: Mapping[str, ProcessingContext]) -> None:
        """Merge returned runtime records into parent-owned compiled contexts."""
        for context_observation in self.contexts:
            context = execution_contexts[context_observation.context_key]
            store = context.runtime_value_store
            store.merge_observed_values(context_observation.records)


class RuntimeObservationMode(Enum):
    """Controls whether worker runtime records are returned to the parent."""

    MERGE_INTO_PARENT = "merge_into_parent"
    OMIT = "omit"

    @property
    def collects_records(self) -> bool:
        return self is RuntimeObservationMode.MERGE_INTO_PARENT

    @property
    def releases_worker_records(self) -> bool:
        return self is RuntimeObservationMode.OMIT


@dataclass(frozen=True)
class ExecutionResult:
    """
    Typed result of pipeline execution for a single axis.
    
    Replaces dict-based results to provide compile-time type safety
    and explicit contracts.
    
    Attributes:
        status: Execution status (success, error, etc.)
        axis_id: Identifier for the axis that was executed
        failed_combination: Optional key of the combination that failed (for sequential mode)
        error_message: Optional error message if status is ERROR
    """
    status: ExecutionStatus
    axis_id: str
    failed_combination: Optional[str] = None
    error_message: Optional[str] = None
    runtime_observation: RuntimeExecutionObservation = field(
        default_factory=RuntimeExecutionObservation
    )
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS
    
    def is_error(self) -> bool:
        """Check if execution failed."""
        return self.status == ExecutionStatus.ERROR
    
    @classmethod
    def success(
        cls,
        axis_id: str,
        runtime_observation: RuntimeExecutionObservation | None = None,
    ) -> 'ExecutionResult':
        """Create a successful execution result."""
        return cls(
            status=ExecutionStatus.SUCCESS,
            axis_id=axis_id,
            runtime_observation=runtime_observation or RuntimeExecutionObservation(),
        )
    
    @classmethod
    def error(cls, axis_id: str, failed_combination: Optional[str] = None, 
              error_message: Optional[str] = None) -> 'ExecutionResult':
        """Create an error execution result."""
        return cls(
            status=ExecutionStatus.ERROR,
            axis_id=axis_id,
            failed_combination=failed_combination,
            error_message=error_message
        )
