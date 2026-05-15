"""Artifact graph extraction for compiled function patterns."""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.function_patterns import DEFAULT_GROUP_KEY
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
)


@dataclass(frozen=True, slots=True)
class ArtifactProducer:
    """Compiled artifact producer identity and scope."""

    name: str
    spec: ArtifactSpec
    groups: tuple[Optional[str], ...]
    invocation_keys: tuple[FunctionInvocationKey, ...]

    def __post_init__(self) -> None:
        if self.name != self.spec.name:
            raise ValueError(
                f"Artifact producer name '{self.name}' does not match "
                f"ArtifactSpec.name '{self.spec.name}'."
            )


@dataclass(frozen=True, slots=True)
class ArtifactConsumer:
    """Compiled artifact consumer identity and declared contract."""

    name: str
    spec: ArtifactSpec
    invocation_keys: tuple[FunctionInvocationKey, ...]

    def __post_init__(self) -> None:
        if self.name != self.spec.name:
            raise ValueError(
                f"Artifact consumer name '{self.name}' does not match "
                f"ArtifactSpec.name '{self.spec.name}'."
            )


@dataclass(frozen=True, slots=True)
class ArtifactGraph:
    """Producer/consumer graph owned by one FunctionStep pattern.

    The graph is the compiler source of truth for artifact names, kinds,
    materialization intent, invocation ownership, and grouped output scope.
    """

    producers: tuple[ArtifactProducer, ...] = ()
    consumers: tuple[ArtifactConsumer, ...] = ()

    @classmethod
    def empty(cls) -> "ArtifactGraph":
        return cls()

    @property
    def outputs(self) -> OrderedDict[str, ArtifactSpec]:
        """Produced artifact specs in first declaration order."""
        return OrderedDict((producer.name, producer.spec) for producer in self.producers)

    @property
    def output_names(self) -> set[str]:
        """Produced artifact names."""
        return set(self.outputs)

    @property
    def output_groups(self) -> dict[str, set[Optional[str]]]:
        """Runtime groups that may produce each artifact."""
        groups: dict[str, set[Optional[str]]] = defaultdict(set)
        for producer in self.producers:
            groups[producer.name].update(producer.groups)
        return groups

    @property
    def inputs(self) -> OrderedDict[str, ArtifactSpec]:
        """Consumed artifact specs in first declaration order."""
        return OrderedDict((consumer.name, consumer.spec) for consumer in self.consumers)

    @property
    def materializations(self) -> dict[str, Any]:
        """Explicit materialization specs keyed by artifact name."""
        return {
            producer.name: producer.spec.materialization
            for producer in self.producers
            if producer.spec.materialization is not None
        }

    def with_output_groups(
        self,
        output_groups: Mapping[str, Iterable[Optional[str]]],
    ) -> "ArtifactGraph":
        """Return a graph with compiler-resolved output scopes."""
        return ArtifactGraph(
            producers=tuple(
                ArtifactProducer(
                    name=producer.name,
                    spec=producer.spec,
                    groups=_unique_preserving_order(
                        list(output_groups.get(producer.name, producer.groups))
                    ),
                    invocation_keys=producer.invocation_keys,
                )
                for producer in self.producers
            ),
            consumers=self.consumers,
        )


def normalize_pattern(pattern: Any) -> Iterator[tuple[Callable, str, int]]:
    """Extract enabled functions from any pattern with runtime invocation positions."""
    for invocation in normalize_function_pattern(pattern).iter_items():
        yield (
            invocation.func,
            invocation.key.group_key,
            invocation.key.position,
        )


def extract_artifact_declarations(
    pattern: Any,
    declaration_provider: InvocationArtifactDeclarationProviderLike = (
        callable_contract_artifact_declarations
    ),
    step_context: ArtifactDeclarationStepContext = (
        ArtifactDeclarationStepContext.empty()
    ),
) -> ArtifactGraph:
    """Extract artifact metadata and per-group ownership from a function pattern."""
    producer_specs: OrderedDict[str, ArtifactSpec] = OrderedDict()
    producer_groups: defaultdict[str, list[Optional[str]]] = defaultdict(list)
    producer_invocations: defaultdict[str, list[FunctionInvocationKey]] = defaultdict(list)
    consumer_specs: OrderedDict[str, ArtifactSpec] = OrderedDict()
    consumer_invocations: defaultdict[str, list[FunctionInvocationKey]] = defaultdict(list)

    for invocation in normalize_function_pattern(pattern).iter_items():
        declarations = declaration_provider(invocation, step_context)
        group_key = invocation.key.group_key
        normalized_key = None if group_key == DEFAULT_GROUP_KEY else group_key

        for name, spec in declarations.outputs:
            producer_specs[name] = _merge_artifact_spec(
                existing=producer_specs.get(name),
                incoming=spec,
                role="producer",
            )
            producer_groups[name].append(normalized_key)
            producer_invocations[name].append(invocation.key)

        for name, spec in declarations.inputs:
            consumer_specs[name] = _merge_artifact_spec(
                existing=consumer_specs.get(name),
                incoming=spec,
                role="consumer",
            )
            consumer_invocations[name].append(invocation.key)

    _validate_local_consumer_producer_kinds(producer_specs, consumer_specs)

    return ArtifactGraph(
        producers=tuple(
            ArtifactProducer(
                name=name,
                spec=spec,
                groups=_unique_preserving_order(producer_groups[name]),
                invocation_keys=tuple(producer_invocations[name]),
            )
            for name, spec in producer_specs.items()
        ),
        consumers=tuple(
            ArtifactConsumer(
                name=name,
                spec=spec,
                invocation_keys=tuple(consumer_invocations[name]),
            )
            for name, spec in consumer_specs.items()
        ),
    )


def _merge_artifact_spec(
    existing: ArtifactSpec | None,
    incoming: ArtifactSpec,
    role: str,
) -> ArtifactSpec:
    if existing is None:
        return incoming
    if existing.kind != incoming.kind:
        raise ValueError(
            f"Conflicting {role} artifact kind for '{incoming.name}': "
            f"{existing.kind.value} vs {incoming.kind.value}."
        )
    if (
        existing.materialization is not None
        and incoming.materialization is not None
        and existing.materialization != incoming.materialization
    ):
        raise ValueError(
            f"Conflicting {role} artifact materialization for '{incoming.name}'."
        )

    materialization = (
        existing.materialization
        if existing.materialization is not None
        else incoming.materialization
    )
    return ArtifactSpec(
        name=existing.name,
        kind=existing.kind,
        materialization=materialization,
        required=existing.required or incoming.required,
    )


def _validate_local_consumer_producer_kinds(
    producer_specs: Mapping[str, ArtifactSpec],
    consumer_specs: Mapping[str, ArtifactSpec],
) -> None:
    for name, consumer_spec in consumer_specs.items():
        producer_spec = producer_specs.get(name)
        if producer_spec is None:
            continue
        if producer_spec.kind != consumer_spec.kind:
            raise ValueError(
                f"Artifact '{name}' is produced as {producer_spec.kind.value} "
                f"but consumed as {consumer_spec.kind.value} in the same FunctionStep."
            )


def _unique_preserving_order(values: list[Optional[str]]) -> tuple[Optional[str], ...]:
    unique: list[Optional[str]] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)
