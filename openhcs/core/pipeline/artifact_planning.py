"""Artifact graph extraction for compiled function patterns."""

from collections import Counter, OrderedDict, defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, ClassVar, Iterable, Iterator, Mapping, Optional

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactMaterializationPayload,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecAccumulator,
    ArtifactSpecRef,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import DEFAULT_GROUP_KEY
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    CompositeInvocationContractProvider,
    InvocationContractProvider,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.processing.materialization import (
    ImageFileOptions,
    MaterializedFilenameIdentity,
    MaterializationSpec,
    ROIOptions,
)


class TerminalMaterializationSpec(MaterializationSpec):
    """Compiler-added persistence excluded from declared export comparison."""

    def participates_in_runtime_export_observation(self) -> bool:
        return False


class StreamingOnlyMaterializationSpec(MaterializationSpec):
    """Compiler-added viewer materialization excluded from persistent exports."""

    def participates_in_runtime_export_observation(self) -> bool:
        return False

    def participates_in_persistent_materialization(self) -> bool:
        return False


class AutomaticArtifactOutputMaterializationStrategy(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Nominal owner of automatic materialization by artifact type."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None

    @abstractmethod
    def materialization(self) -> ArtifactMaterializationPayload:
        """Return the explicit materialization contract for this artifact family."""

    def materializes_consumed_outputs(self) -> bool:
        """Return whether consumed outputs retain automatic materialization."""

        return False


class AutomaticImageArtifactOutputMaterializationStrategy(
    AutomaticArtifactOutputMaterializationStrategy,
):
    """Materialize unconsumed image-family outputs as TIFF files."""

    artifact_type = ImageArtifactType

    def materialization(self) -> ArtifactMaterializationPayload:
        return TerminalMaterializationSpec(
            ImageFileOptions(filename_suffix=".tif")
        )


class AutomaticObjectLabelsArtifactOutputMaterializationStrategy(
    AutomaticArtifactOutputMaterializationStrategy,
):
    """Stream every object-label output through the canonical ROI writer."""

    artifact_type = ObjectLabelsArtifactType

    def materialization(self) -> ArtifactMaterializationPayload:
        return StreamingOnlyMaterializationSpec(
            ROIOptions(
                min_area=1,
                filename_identity=MaterializedFilenameIdentity.ARTIFACT_NAME,
            )
        )

    def materializes_consumed_outputs(self) -> bool:
        return True


class ArtifactOutputMaterializationPlanner:
    """Promote terminal outputs to explicit nominal materialization contracts."""

    @staticmethod
    def materialization_for(
        spec: ArtifactSpec,
        future_input_refs: Iterable[ArtifactSpecRef],
    ) -> ArtifactMaterializationPayload | None:
        """Preserve explicit contracts and materialize unconsumed nominal outputs."""

        if spec.materialization is not None:
            return spec.materialization
        strategy = AutomaticArtifactOutputMaterializationStrategy.for_context(
            spec.artifact_type,
            required=False,
        )
        if strategy is None:
            return None
        input_ref = spec.ref().for_plan_type(ArtifactInputPlan)
        if (
            input_ref in frozenset(future_input_refs)
            and not strategy.materializes_consumed_outputs()
        ):
            return None
        return strategy.materialization()


@dataclass(frozen=True, slots=True)
class ArtifactProducer:
    """Compiled artifact producer identity and scope."""

    spec: ArtifactSpec
    groups: tuple[Optional[str], ...]
    invocation_keys: tuple[FunctionInvocationKey, ...]
    producer_step_index: int | None = None

    def __post_init__(self) -> None:
        if self.producer_step_index is not None and (
            type(self.producer_step_index) is not int
            or self.producer_step_index < 0
        ):
            raise ValueError(
                "ArtifactProducer.producer_step_index must be a non-negative "
                "integer or None."
            )

    def has_explicit_invocation_group_ownership(self) -> bool:
        """Return whether grouped pattern dispatch owns this output's groups."""

        return bool(self.invocation_keys) and all(
            key.group_key != DEFAULT_GROUP_KEY for key in self.invocation_keys
        )

    def owns_invocation(self, invocation_key: FunctionInvocationKey) -> bool:
        """Return whether this producer owns the consumer's compile-time group."""

        return invocation_key in self.invocation_keys


def artifact_producers_for_outputs(
    outputs: Iterable[ArtifactSpec],
    *,
    groups: Iterable[Optional[str]],
    invocation_keys: Iterable[FunctionInvocationKey],
) -> tuple[ArtifactProducer, ...]:
    """Bind declared outputs to their exact invocation and group ownership."""

    resolved_groups = ArtifactGraph.unique_preserving_order(groups)
    resolved_invocation_keys = tuple(dict.fromkeys(invocation_keys))
    if not resolved_invocation_keys:
        raise ValueError("Artifact producers require at least one invocation key.")
    return tuple(
        ArtifactProducer(
            spec=spec,
            groups=resolved_groups,
            invocation_keys=resolved_invocation_keys,
        )
        for spec in outputs
    )


@dataclass(frozen=True, slots=True)
class ArtifactConsumer:
    """Compiled artifact consumer identity and declared contract."""

    spec: ArtifactSpec
    invocation_keys: tuple[FunctionInvocationKey, ...]

    @property
    def groups(self) -> tuple[Optional[str], ...]:
        """Return invocation groups that consume this artifact."""

        return ArtifactGraph.unique_preserving_order(
            None if key.group_key == DEFAULT_GROUP_KEY else key.group_key
            for key in self.invocation_keys
        )


@dataclass(frozen=True, slots=True)
class ArtifactGraph:
    """Producer/consumer graph owned by one FunctionStep pattern.

    The graph is the compiler source of truth for artifact names, kinds,
    materialization intent, invocation ownership, and grouped output scope.
    """

    producers: tuple[ArtifactProducer, ...] = ()
    consumers: tuple[ArtifactConsumer, ...] = ()
    non_plan_consumers: tuple[ArtifactConsumer, ...] = ()

    @classmethod
    def empty(cls) -> "ArtifactGraph":
        return cls()

    @property
    def outputs(self) -> OrderedDict[ArtifactSpecRef, ArtifactSpec]:
        """Produced artifact specs in first declaration order."""
        return OrderedDict(
            (producer.spec.ref(), producer.spec) for producer in self.producers
        )

    def output_storage_keys(self) -> OrderedDict[ArtifactSpecRef, str]:
        """Return collision-safe storage keys for exact output declarations."""

        output_refs = tuple(self.outputs)
        name_counts = Counter(ref.name for ref in output_refs)
        storage_keys = OrderedDict(
            (
                ref,
                (
                    ref.name
                    if name_counts[ref.name] == 1
                    else f"{ref.name}__{ref.artifact_type.require_value()}"
                ),
            )
            for ref in output_refs
        )
        refs_by_storage_key: dict[str, list[ArtifactSpecRef]] = defaultdict(list)
        for ref, storage_key in storage_keys.items():
            refs_by_storage_key[storage_key].append(ref)
        collisions = {
            storage_key: tuple(refs)
            for storage_key, refs in refs_by_storage_key.items()
            if len(refs) > 1
        }
        if collisions:
            raise ValueError(
                "Artifact output declarations produce conflicting storage keys: "
                f"{collisions!r}. Rename the conflicting artifact outputs."
            )
        return storage_keys

    def require_output_storage_key(self, ref: ArtifactSpecRef) -> str:
        """Return the storage key derived for one exact output declaration."""

        storage_key = self.output_storage_keys().get(ref)
        if storage_key is None:
            raise KeyError(f"Artifact graph has no output declaration {ref!r}.")
        return storage_key

    @property
    def output_groups(self) -> dict[ArtifactSpecRef, set[Optional[str]]]:
        """Runtime groups that may produce each exact artifact."""
        groups: dict[ArtifactSpecRef, set[Optional[str]]] = defaultdict(set)
        for producer in self.producers:
            groups[producer.spec.ref()].update(producer.groups)
        return groups

    @property
    def inputs(self) -> OrderedDict[ArtifactSpecRef, ArtifactSpec]:
        """Consumed artifact specs in first declaration order."""
        return OrderedDict(
            (consumer.spec.ref(), consumer.spec) for consumer in self.consumers
        )

    def invocation_keys(self) -> tuple[FunctionInvocationKey, ...]:
        """Return exact invocation identities represented by this graph."""

        return tuple(
            dict.fromkeys(
                key
                for endpoint in (*self.producers, *self.consumers)
                for key in endpoint.invocation_keys
            )
        )

    def with_output_groups(
        self,
        output_groups: Mapping[ArtifactSpecRef, Iterable[Optional[str]]],
    ) -> "ArtifactGraph":
        """Return a graph with compiler-resolved output scopes."""
        declared_outputs = self.outputs
        for output_ref in output_groups:
            if not isinstance(output_ref, ArtifactSpecRef):
                raise TypeError(
                    "Artifact output-group maps require ArtifactSpecRef keys, "
                    f"got {type(output_ref).__name__}."
                )
            if output_ref not in declared_outputs:
                raise ValueError(
                    f"Artifact output-group key {output_ref!r} is not an exact "
                    "declared output."
                )

        producers: list[ArtifactProducer] = []
        for producer in self.producers:
            output_ref = producer.spec.ref()
            groups = (
                self._require_output_group_values(
                    output_ref,
                    output_groups[output_ref],
                )
                if output_ref in output_groups
                else producer.groups
            )
            producers.append(
                ArtifactProducer(
                    spec=producer.spec,
                    groups=groups,
                    invocation_keys=producer.invocation_keys,
                    producer_step_index=producer.producer_step_index,
                )
            )
        return ArtifactGraph(
            producers=tuple(producers),
            consumers=self.consumers,
            non_plan_consumers=self.non_plan_consumers,
        )

    @staticmethod
    def _require_output_group_values(
        output_ref: ArtifactSpecRef,
        groups: Iterable[Optional[str]],
    ) -> tuple[Optional[str], ...]:
        """Return one declared output's validated unique group keys."""

        if isinstance(groups, (str, bytes)):
            raise TypeError(
                f"Artifact output groups for {output_ref!r} must be an iterable "
                "of string or None keys, not a string."
            )
        try:
            group_values = tuple(groups)
        except TypeError as exc:
            raise TypeError(
                f"Artifact output groups for {output_ref!r} must be iterable."
            ) from exc
        for group in group_values:
            if group is not None and not isinstance(group, str):
                raise TypeError(
                    f"Artifact output groups for {output_ref!r} require string "
                    f"or None keys, got {type(group).__name__}."
                )
        return ArtifactGraph.unique_preserving_order(group_values)

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
    invocation_contract_provider: InvocationContractProvider = (
        CompositeInvocationContractProvider(())
    ),
    step_context: ArtifactDeclarationStepContext = (
        ArtifactDeclarationStepContext.empty()
    ),
) -> ArtifactGraph:
    """Extract artifact metadata and per-group ownership from a function pattern."""
    producer_specs = ArtifactSpecAccumulator.empty("producer")
    producer_groups: defaultdict[
        ArtifactSpecRef,
        list[Optional[str]],
    ] = defaultdict(list)
    producer_invocations: defaultdict[
        ArtifactSpecRef,
        list[FunctionInvocationKey],
    ] = defaultdict(list)
    consumers: list[ArtifactConsumer] = []
    declared_input_consumers: list[ArtifactConsumer] = []

    for invocation in normalize_function_pattern(pattern).iter_items():
        contract_plan = invocation_contract_provider(invocation, step_context)
        if contract_plan is not None:
            invocation = replace(invocation, contract=contract_plan.contract)
        artifact_selector = declaration_provider(invocation, step_context)
        artifact_selector.validate_artifact_relation_refs(
            owner_name=invocation.contract.function_name,
        )
        group_key = invocation.key.group_key
        normalized_key = None if group_key == DEFAULT_GROUP_KEY else group_key

        for spec in artifact_selector.artifact_specs.for_plan_type(
            ArtifactInputPlan
        ).specs:
            declared_input_consumers.append(
                ArtifactConsumer(
                    spec=spec,
                    invocation_keys=(invocation.key,),
                )
            )

        for spec in artifact_selector.artifact_key_specs.for_plan_type(
            ArtifactOutputPlan
        ).specs:
            ref = spec.ref()
            producer_specs.add(spec)
            producer_groups[ref].append(normalized_key)
            producer_invocations[ref].append(invocation.key)

        for spec in artifact_selector.artifact_key_specs.for_plan_type(
            ArtifactInputPlan
        ).specs:
            consumers.append(
                ArtifactConsumer(
                    spec=spec,
                    invocation_keys=(invocation.key,),
                )
            )

    planned_input_refs = frozenset(consumer.spec.ref() for consumer in consumers)
    return ArtifactGraph(
        producers=tuple(
            ArtifactProducer(
                spec=spec,
                groups=ArtifactGraph.unique_preserving_order(producer_groups[ref]),
                invocation_keys=tuple(producer_invocations[ref]),
            )
            for ref, spec in producer_specs.specs.items()
        ),
        consumers=tuple(consumers),
        non_plan_consumers=tuple(
            consumer
            for consumer in declared_input_consumers
            if consumer.spec.ref() not in planned_input_refs
        ),
    )
