"""Invocation-aware artifact declarations for function-pattern compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import GroupBy
from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
)
from openhcs.core.config import StepSourceBindingsConfig
from openhcs.core.source_bindings import EMPTY_SOURCE_BINDINGS

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.core.pipeline.artifact_planning import ArtifactGraph, ArtifactProducer
    from openhcs.core.pipeline.compilation_session import CompilationSession


def unnamed_main_flow_artifact_name(
    step_index: int,
    invocation_key: "FunctionInvocationKey",
) -> str:
    """Return the deterministic compiler-only identity for unnamed main flow."""

    from openhcs.core.function_patterns import FunctionInvocationKey

    if not isinstance(step_index, int) or step_index < 0:
        raise TypeError(
            "Unnamed main-flow identity requires a non-negative step index."
        )
    if not isinstance(invocation_key, FunctionInvocationKey):
        raise TypeError(
            "Unnamed main-flow identity requires FunctionInvocationKey, got "
            f"{type(invocation_key).__name__}."
        )
    return (
        f"__openhcs_main_flow_step_{step_index + 1}_"
        f"{invocation_key.group_key}_{invocation_key.position + 1}_"
        f"{invocation_key.function_name}"
    )


@dataclass(frozen=True, slots=True)
class ArtifactDeclarationStepContext:
    """Compile-time step context available to artifact declaration providers."""

    step_name: str | None = None
    step_index: int | None = None
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS
    group_by: GroupBy = GroupBy.NONE
    input_source: InputSource = InputSource.PREVIOUS_STEP
    available_artifacts: ArtifactSpecCollection = field(
        default_factory=lambda: ArtifactSpecCollection(())
    )
    main_flow_artifacts: ArtifactSpecCollection = field(
        default_factory=lambda: ArtifactSpecCollection(())
    )
    available_artifact_producers: tuple["ArtifactProducer", ...] = ()

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
        if not isinstance(self.group_by, GroupBy):
            raise TypeError(
                "ArtifactDeclarationStepContext.group_by must be GroupBy, got "
                f"{type(self.group_by).__name__}."
            )
        if not isinstance(self.input_source, InputSource):
            raise TypeError(
                "ArtifactDeclarationStepContext.input_source must be InputSource, got "
                f"{type(self.input_source).__name__}."
            )
        for field_name, collection in (
            ("available_artifacts", self.available_artifacts),
            ("main_flow_artifacts", self.main_flow_artifacts),
        ):
            if not isinstance(collection, ArtifactSpecCollection):
                raise TypeError(
                    f"ArtifactDeclarationStepContext.{field_name} must be "
                    "ArtifactSpecCollection, got "
                    f"{type(collection).__name__}."
                )
            object.__setattr__(
                self,
                field_name,
                ArtifactSpecCollection(collection.specs),
            )
        producers = tuple(self.available_artifact_producers)
        if producers:
            from openhcs.core.pipeline.artifact_planning import ArtifactProducer

            invalid = tuple(
                producer
                for producer in producers
                if not isinstance(producer, ArtifactProducer)
            )
            if invalid:
                raise TypeError(
                    "ArtifactDeclarationStepContext.available_artifact_producers "
                    "must contain ArtifactProducer values."
                )
        object.__setattr__(self, "available_artifact_producers", producers)

    def with_source_declarations(
        self,
        source_specs: Iterable[ArtifactSpec],
    ) -> "ArtifactDeclarationStepContext":
        """Return this context after adding source declarations for the step."""

        declared_sources = ArtifactSpecCollection(source_specs)
        source_refs = frozenset(
            spec.ref().for_plan_type(ArtifactInputPlan)
            for spec in declared_sources.specs
        )
        main_flow_artifacts = self.main_flow_artifacts
        primary_source_refs = frozenset(
            binding.input_spec().ref()
            for binding in self.source_bindings.primary_plane_bindings
        )
        if self.input_source is InputSource.PIPELINE_START:
            main_flow_artifacts = ArtifactSpecCollection(
                spec for spec in declared_sources if spec.ref() in primary_source_refs
            )
        return replace(
            self,
            available_artifacts=self.available_artifacts.rebind(declared_sources.specs),
            main_flow_artifacts=main_flow_artifacts,
            available_artifact_producers=tuple(
                producer
                for producer in self.available_artifact_producers
                if producer.spec.ref().for_plan_type(ArtifactInputPlan)
                not in source_refs
            ),
        )

    def available_artifact_producer_for(
        self,
        spec: ArtifactSpec,
    ) -> "ArtifactProducer | None":
        """Return the exact active producer for an artifact declaration."""

        if not isinstance(spec, ArtifactSpec):
            raise TypeError(
                "Artifact producer lookup requires ArtifactSpec, got "
                f"{type(spec).__name__}."
            )
        input_ref = spec.ref().for_plan_type(ArtifactInputPlan)
        matches = tuple(
            producer
            for producer in self.available_artifact_producers
            if producer.spec.ref().for_plan_type(ArtifactInputPlan) == input_ref
        )
        if len(matches) > 1:
            raise ValueError(
                "Artifact declaration context has multiple active producers for "
                f"{input_ref!r}."
            )
        return matches[0] if matches else None

    def with_source_binding_scope(
        self,
        *,
        source_bindings: StepSourceBindingsConfig,
        group_by: GroupBy,
        input_source: InputSource,
    ) -> "ArtifactDeclarationStepContext":
        """Apply one resolved step's source scope and declared source artifacts."""

        scoped = replace(
            self,
            source_bindings=source_bindings,
            group_by=group_by,
            input_source=input_source,
        )
        return scoped.with_source_declarations(
            binding.input_spec() for binding in source_bindings.binding_declarations
        )

    def advance_artifact_graph(
        self,
        graph: "ArtifactGraph",
        *,
        main_flow_artifacts: ArtifactSpecCollection,
    ) -> "ArtifactDeclarationStepContext":
        """Return the forward declaration context after one artifact graph."""

        from openhcs.core.pipeline.artifact_planning import ArtifactGraph

        if not isinstance(graph, ArtifactGraph):
            raise TypeError(
                "Artifact declaration advancement requires ArtifactGraph, got "
                f"{type(graph).__name__}."
            )
        if not isinstance(main_flow_artifacts, ArtifactSpecCollection):
            raise TypeError(
                "Artifact graph advancement requires an ArtifactSpecCollection "
                "for main flow."
            )
        produced_refs = frozenset(
            producer.spec.ref().for_plan_type(ArtifactInputPlan)
            for producer in graph.producers
        )
        return replace(
            self,
            available_artifacts=self.available_artifacts.rebind(
                producer.spec for producer in graph.producers
            ),
            main_flow_artifacts=main_flow_artifacts,
            available_artifact_producers=(
                *(
                    producer
                    for producer in self.available_artifact_producers
                    if producer.spec.ref().for_plan_type(ArtifactInputPlan)
                    not in produced_refs
                ),
                *graph.producers,
            ),
        )


def callable_contract_artifact_declarations(
    invocation: "NormalizedFunctionItem",
    step_context: ArtifactDeclarationStepContext,
) -> ArtifactPlanKeySelector:
    """Return the nominal artifact-plan selector for one invocation contract."""
    del step_context
    return invocation.contract


InvocationArtifactDeclarationProviderLike = Callable[
    ["NormalizedFunctionItem", ArtifactDeclarationStepContext],
    ArtifactPlanKeySelector,
]


class InvocationContractProvider(ABC):
    """Compile-time hook for replacing public callables with runtime contracts."""

    @abstractmethod
    def __call__(
        self,
        invocation: "NormalizedFunctionItem",
        step_context: ArtifactDeclarationStepContext,
    ) -> "InvocationContractPlan | None":
        """Return a compile-only callable contract plan for this invocation."""


@dataclass(frozen=True, slots=True)
class InvocationContractPlan:
    """Compile-time replacement contract plus kwargs consumed by planning."""

    contract: "CallableContract"
    consumed_kwarg_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        consumed_names = tuple(self.consumed_kwarg_names)
        if any(not isinstance(name, str) or not name for name in consumed_names):
            raise TypeError(
                "InvocationContractPlan.consumed_kwarg_names must contain "
                "non-empty strings."
            )
        if len(frozenset(consumed_names)) != len(consumed_names):
            raise ValueError(
                "InvocationContractPlan.consumed_kwarg_names cannot contain duplicates."
            )
        object.__setattr__(self, "consumed_kwarg_names", consumed_names)

    def consume_authored_kwargs(
        self,
        invocation: "NormalizedFunctionItem",
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[tuple[object, object], ...]:
        """Remove compile-only kwargs after proving the user authored them."""

        authored_names = frozenset(name for name, _value in invocation.kwargs)
        missing = tuple(
            name for name in self.consumed_kwarg_names if name not in authored_names
        )
        if missing:
            raise ValueError(
                "Invocation contract consumed kwargs that were not authored for "
                f"step {step_context.step_index!r} ({step_context.step_name!r}), "
                f"invocation {invocation.key!r}: {missing!r}."
            )
        consumed_names = frozenset(self.consumed_kwarg_names)
        return tuple(
            (name, value)
            for name, value in invocation.kwargs
            if name not in consumed_names
        )


class InvocationContractProviderFactory(ABC, metaclass=AutoRegisterMeta):
    """Registered owner for compile-time invocation contract providers."""

    __registry_key__ = "__name__"

    @classmethod
    @abstractmethod
    def provider_for_session(
        cls,
        session: "CompilationSession",
    ) -> InvocationContractProvider | None:
        """Return an invocation-contract provider for one compilation session."""


@dataclass(frozen=True, slots=True)
class CompositeInvocationContractProvider:
    """Require at most one compile-time invocation-contract provider claim."""

    providers: tuple[InvocationContractProvider, ...]

    def __post_init__(self) -> None:
        providers = tuple(self.providers)
        for provider in providers:
            if not isinstance(provider, InvocationContractProvider):
                raise TypeError(
                    "CompositeInvocationContractProvider requires nominal "
                    "InvocationContractProvider instances, got "
                    f"{type(provider).__name__}."
                )
        object.__setattr__(self, "providers", providers)

    def __call__(
        self,
        invocation: "NormalizedFunctionItem",
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationContractPlan | None:
        claims = tuple(
            (provider, plan)
            for provider in self.providers
            for plan in (provider(invocation, step_context),)
            if plan is not None
        )
        if len(claims) > 1:
            raise ValueError(
                "Multiple invocation contract providers claimed one callable: "
                f"{tuple(type(provider).__name__ for provider, _plan in claims)!r}."
            )
        return claims[0][1] if claims else None


class PipelineInvocationContractProviderAuthority:
    """Resolve all registered compile-time invocation-contract providers."""

    @classmethod
    def provider_for_session(
        cls,
        session: "CompilationSession",
    ) -> InvocationContractProvider:
        providers: list[InvocationContractProvider] = []

        for provider_factory in InvocationContractProviderFactory.__registry__.values():
            provider = provider_factory.provider_for_session(session)
            if provider is not None:
                providers.append(provider)
        return CompositeInvocationContractProvider(tuple(providers))
