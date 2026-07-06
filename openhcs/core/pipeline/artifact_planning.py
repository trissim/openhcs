"""Artifact graph extraction for compiled function patterns."""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactSpec
from openhcs.core.function_patterns import DEFAULT_GROUP_KEY
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractProviderLike,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
    public_callable_invocation_contract,
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
                    groups=ArtifactGraph.unique_preserving_order(
                        list(output_groups.get(producer.name, producer.groups))
                    ),
                    invocation_keys=producer.invocation_keys,
                )
                for producer in self.producers
            ),
            consumers=self.consumers,
        )

    @staticmethod
    def unique_preserving_order(
        values: Iterable[Optional[str]],
    ) -> tuple[Optional[str], ...]:
        """Return unique group keys while preserving declaration order."""
        unique: list[Optional[str]] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return tuple(unique)


@dataclass(frozen=True, slots=True)
class ArtifactSpecAccumulator:
    """Ordered artifact-spec merge authority for one producer/consumer role."""

    role: str
    specs: OrderedDict[str, ArtifactSpec]

    @classmethod
    def empty(cls, role: str) -> "ArtifactSpecAccumulator":
        """Create an empty ordered accumulator for an artifact role."""
        return cls(role=role, specs=OrderedDict())

    def add(self, incoming: ArtifactSpec) -> None:
        """Merge an incoming spec into this accumulator."""
        if incoming.name not in self.specs:
            self.specs[incoming.name] = incoming
            return
        self.specs[incoming.name] = self.merge_existing(
            existing=self.specs[incoming.name],
            incoming=incoming,
        )

    def merge_existing(
        self,
        *,
        existing: ArtifactSpec,
        incoming: ArtifactSpec,
    ) -> ArtifactSpec:
        """Merge two declarations for the same artifact name."""
        if existing.plan_type is not incoming.plan_type:
            raise ValueError(
                f"Conflicting {self.role} artifact role for '{incoming.name}': "
                f"{existing.plan_type.plan_role} vs {incoming.plan_type.plan_role}."
            )
        if existing.artifact_type != incoming.artifact_type:
            raise ValueError(
                f"Conflicting {self.role} artifact type for '{incoming.name}': "
                f"{existing.artifact_type.value} vs {incoming.artifact_type.value}."
            )
        if (
            existing.materialization is not None
            and incoming.materialization is not None
            and existing.materialization != incoming.materialization
        ):
            raise ValueError(
                f"Conflicting {self.role} artifact materialization for "
                f"'{incoming.name}'."
            )

        materialization = (
            existing.materialization
            if existing.materialization is not None
            else incoming.materialization
        )
        if (
            existing.sidecar_role is not None
            and incoming.sidecar_role is not None
            and existing.sidecar_role is not incoming.sidecar_role
        ):
            raise ValueError(
                f"Conflicting {self.role} artifact sidecar role for "
                f"'{incoming.name}'."
            )
        sidecar_role = (
            existing.sidecar_role
            if existing.sidecar_role is not None
            else incoming.sidecar_role
        )
        relations = tuple(dict.fromkeys((*existing.relations, *incoming.relations)))
        return replace(
            existing,
            materialization=materialization,
            required=existing.required or incoming.required,
            sidecar_role=sidecar_role,
            relations=relations,
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
    invocation_contract_provider: InvocationContractProviderLike = (
        public_callable_invocation_contract
    ),
    step_context: ArtifactDeclarationStepContext = (
        ArtifactDeclarationStepContext.empty()
    ),
) -> ArtifactGraph:
    """Extract artifact metadata and per-group ownership from a function pattern."""
    producer_specs = ArtifactSpecAccumulator.empty("producer")
    producer_groups: defaultdict[str, list[Optional[str]]] = defaultdict(list)
    producer_invocations: defaultdict[str, list[FunctionInvocationKey]] = defaultdict(list)
    consumer_specs = ArtifactSpecAccumulator.empty("consumer")
    consumer_invocations: defaultdict[str, list[FunctionInvocationKey]] = defaultdict(list)

    for invocation in normalize_function_pattern(pattern).iter_items():
        compile_contract = invocation_contract_provider(invocation, step_context)
        if compile_contract is not None:
            invocation = replace(invocation, contract=compile_contract)
        declarations = declaration_provider(invocation, step_context)
        group_key = invocation.key.group_key
        normalized_key = None if group_key == DEFAULT_GROUP_KEY else group_key

        for name, spec in declarations.outputs:
            producer_specs.add(spec)
            producer_groups[name].append(normalized_key)
            producer_invocations[name].append(invocation.key)

        for spec in declarations.artifact_key_specs.for_plan_type(
            ArtifactInputPlan
        ).specs:
            name = spec.name
            consumer_specs.add(spec)
            consumer_invocations[name].append(invocation.key)

    _validate_local_consumer_producer_kinds(producer_specs.specs, consumer_specs.specs)

    return ArtifactGraph(
        producers=tuple(
            ArtifactProducer(
                name=name,
                spec=spec,
                groups=ArtifactGraph.unique_preserving_order(producer_groups[name]),
                invocation_keys=tuple(producer_invocations[name]),
            )
            for name, spec in producer_specs.specs.items()
        ),
        consumers=tuple(
            ArtifactConsumer(
                name=name,
                spec=spec,
                invocation_keys=tuple(consumer_invocations[name]),
            )
            for name, spec in consumer_specs.specs.items()
        ),
    )


def _validate_local_consumer_producer_kinds(
    producer_specs: Mapping[str, ArtifactSpec],
    consumer_specs: Mapping[str, ArtifactSpec],
) -> None:
    for name, consumer_spec in consumer_specs.items():
        producer_spec = producer_specs.get(name)
        if producer_spec is None:
            continue
        if producer_spec.artifact_type != consumer_spec.artifact_type:
            raise ValueError(
                f"Artifact '{name}' is produced as {producer_spec.artifact_type.value} "
                f"but consumed as {consumer_spec.artifact_type.value} in the same FunctionStep."
            )
