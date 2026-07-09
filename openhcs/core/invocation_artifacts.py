"""Invocation-aware artifact declarations for function-pattern compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
)
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

    artifacts: tuple[ArtifactSpec, ...] = ()
    plan_key_artifacts: tuple[ArtifactSpec, ...] | None = None

    def __post_init__(self) -> None:
        collection = ArtifactSpecCollection(self.artifacts)
        object.__setattr__(self, "artifacts", collection.specs)
        if self.plan_key_artifacts is not None:
            key_collection = ArtifactSpecCollection(self.plan_key_artifacts)
            object.__setattr__(self, "plan_key_artifacts", key_collection.specs)
        self.validate_artifact_relation_refs(
            owner_name="InvocationArtifactDeclarations",
        )

    @classmethod
    def from_contract(cls, contract: Any) -> "InvocationArtifactDeclarations":
        """Build declarations from the callable contract metadata fallback."""
        if contract.module_artifact_contract is not None:
            return cls.from_module_contract(contract.module_artifact_contract)
        return cls(
            artifacts=tuple(
                spec
                for _name, spec in (
                    *contract.artifact_inputs,
                    *contract.artifact_outputs,
                )
            ),
        )

    @classmethod
    def from_module_contract(
        cls,
        contract: ModuleArtifactContract,
    ) -> "InvocationArtifactDeclarations":
        """Build declarations from a typed executable-module contract."""
        return cls(
            artifacts=(
                *contract.declared_input_specs(),
                *contract.outputs,
            ),
            plan_key_artifacts=(
                *contract.runtime_artifact_inputs,
                *contract.outputs,
            ),
        )

    @property
    def artifact_specs(self) -> ArtifactSpecCollection:
        """All artifact specs declared by this invocation."""
        return ArtifactSpecCollection(self.artifacts)

    @property
    def artifact_key_specs(self) -> ArtifactSpecCollection:
        """Artifact specs that participate in runtime plan-key selection."""
        if self.plan_key_artifacts is None:
            return self.artifact_specs
        return ArtifactSpecCollection(self.plan_key_artifacts)

    @property
    def inputs(self) -> InvocationArtifactSpecItems:
        """Input declarations projected from the canonical artifact collection."""
        return tuple(
            (spec.name, spec)
            for spec in self.artifact_specs.for_plan_type(ArtifactInputPlan).specs
        )

    @property
    def outputs(self) -> InvocationArtifactSpecItems:
        """Output declarations projected from the canonical artifact collection."""
        return tuple(
            (spec.name, spec)
            for spec in self.artifact_specs.for_plan_type(ArtifactOutputPlan).specs
        )


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
    ) -> "InvocationContractPlan | None":
        """Return a compile-only callable contract plan for this invocation."""


@dataclass(frozen=True, slots=True)
class InvocationContractPlan:
    """Compile-time replacement contract plus kwargs consumed by planning."""

    contract: "CallableContract"
    consumed_kwarg_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "consumed_kwarg_names",
            tuple(dict.fromkeys(str(name) for name in self.consumed_kwarg_names)),
        )


def public_callable_invocation_contract(
    invocation: Any,
    step_context: ArtifactDeclarationStepContext,
) -> InvocationContractPlan | None:
    """Default provider: public callable metadata is already the contract."""
    del invocation, step_context
    return None


InvocationContractProviderLike = Callable[
    [Any, ArtifactDeclarationStepContext],
    InvocationContractPlan | None,
]


class InvocationContractProviderFactory(ABC, metaclass=AutoRegisterMeta):
    """Registered owner for compile-time invocation contract providers."""

    __registry_key__ = "__name__"

    @classmethod
    @abstractmethod
    def provider_for_session(
        cls,
        session: Any,
    ) -> InvocationContractProviderLike | None:
        """Return an invocation-contract provider for one compilation session."""


@dataclass(frozen=True, slots=True)
class CompositeInvocationContractProvider:
    """Try compile-time invocation-contract providers in declaration order."""

    providers: tuple[InvocationContractProviderLike, ...]

    def __call__(
        self,
        invocation: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationContractPlan | None:
        for provider in self.providers:
            plan = provider(invocation, step_context)
            if plan is not None:
                return plan
        return None


class PipelineInvocationContractProviderAuthority:
    """Resolve all registered compile-time invocation-contract providers."""

    @classmethod
    def provider_for_session(
        cls,
        session: Any,
    ) -> InvocationContractProviderLike:
        providers: list[InvocationContractProviderLike] = []

        for provider_factory in InvocationContractProviderFactory.__registry__.values():
            provider = provider_factory.provider_for_session(session)
            if provider is not None:
                providers.append(provider)
        if not providers:
            return public_callable_invocation_contract
        return CompositeInvocationContractProvider(tuple(providers))
