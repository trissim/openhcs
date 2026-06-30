"""Invocation-aware artifact declarations for function-pattern compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    StepSourceBindingsConfig,
)

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableContract


InvocationArtifactSpecItems = tuple[tuple[str, ArtifactSpec], ...]


@dataclass(frozen=True, slots=True)
class ArtifactDeclarationStepContext:
    """Compile-time step context available to artifact declaration providers."""

    step_name: str | None = None
    step_index: int | None = None
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS
    processing_config: Any | None = None
    source_provenance: Any | None = None

    @classmethod
    def empty(cls) -> "ArtifactDeclarationStepContext":
        """Return a context for provider paths that are not step-aware."""
        return cls()

    def __post_init__(self) -> None:
        if self.step_name is not None and not isinstance(self.step_name, str):
            raise TypeError(
                "ArtifactDeclarationStepContext.step_name must be str or None, "
                f"got {type(self.step_name).__name__}."
            )
        if self.step_index is not None and not isinstance(self.step_index, int):
            raise TypeError(
                "ArtifactDeclarationStepContext.step_index must be int or None, "
                f"got {type(self.step_index).__name__}."
            )
        if not isinstance(self.source_bindings, StepSourceBindingsConfig):
            raise TypeError(
                "ArtifactDeclarationStepContext.source_bindings must be "
                f"StepSourceBindingsConfig, got {type(self.source_bindings).__name__}."
            )


@dataclass(frozen=True, slots=True)
class InvocationArtifactDeclarations(ArtifactPlanKeySelector):
    """Artifact declarations owned by one normalized function invocation."""

    inputs: InvocationArtifactSpecItems = ()
    outputs: InvocationArtifactSpecItems = ()

    @classmethod
    def from_contract(cls, contract: Any) -> "InvocationArtifactDeclarations":
        """Build declarations from the callable contract metadata fallback."""
        if contract.module_artifact_contract is not None:
            return cls.from_module_contract(contract.module_artifact_contract)
        return cls(
            inputs=tuple(contract.artifact_inputs),
            outputs=tuple(contract.artifact_outputs),
        )

    @classmethod
    def from_module_contract(
        cls,
        contract: ModuleArtifactContract,
    ) -> "InvocationArtifactDeclarations":
        """Build declarations from a typed executable-module contract."""
        return cls(
            inputs=tuple((spec.name, spec) for spec in contract.runtime_artifact_inputs),
            outputs=tuple((spec.name, spec) for spec in contract.outputs),
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        """Declared artifact input names in declaration order."""
        return tuple(name for name, _spec in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Declared artifact output names in declaration order."""
        return tuple(name for name, _spec in self.outputs)


class InvocationArtifactDeclarationProvider(ABC):
    """Callable extension point for invocation-specific artifact declarations."""

    @abstractmethod
    def __call__(
        self,
        invocation: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationArtifactDeclarations:
        """Return artifact declarations for one normalized invocation."""


def callable_contract_artifact_declarations(
    invocation: Any,
    step_context: ArtifactDeclarationStepContext,
) -> InvocationArtifactDeclarations:
    """Default provider that preserves existing callable-contract behavior."""
    del step_context
    return InvocationArtifactDeclarations.from_contract(invocation.contract)


InvocationArtifactDeclarationProviderLike = Callable[
    [Any, ArtifactDeclarationStepContext],
    InvocationArtifactDeclarations,
]


class InvocationContractProvider(ABC):
    """Compile-time hook for replacing public callables with runtime contracts."""

    @abstractmethod
    def __call__(
        self,
        invocation: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> "CallableContract | None":
        """Return a compile-only callable contract for this invocation."""


def public_callable_invocation_contract(
    invocation: Any,
    step_context: ArtifactDeclarationStepContext,
) -> "CallableContract | None":
    """Default provider: public callable metadata is already the contract."""
    del invocation, step_context
    return None


InvocationContractProviderLike = Callable[
    [Any, ArtifactDeclarationStepContext],
    "CallableContract | None",
]


@dataclass(frozen=True, slots=True)
class PipelineInvocationContractProviderMetadata:
    """Typed metadata key for pipeline-owned compile-time contract providers."""

    metadata_key: ClassVar[str] = "invocation_contract_provider"

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> InvocationContractProviderLike:
        value = metadata.get(cls.metadata_key)
        if value is None:
            return public_callable_invocation_contract
        if not callable(value):
            raise TypeError(
                "Pipeline invocation contract provider metadata must be callable, "
                f"got {type(value).__name__}."
            )
        return value

    @classmethod
    def with_provider(
        cls,
        metadata: Mapping[str, object],
        provider: InvocationContractProviderLike,
    ) -> dict[str, object]:
        if not callable(provider):
            raise TypeError(
                "Pipeline invocation contract provider must be callable, "
                f"got {type(provider).__name__}."
            )
        updated = dict(metadata)
        updated[cls.metadata_key] = provider
        return updated
