from __future__ import annotations

from dataclasses import replace

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    InputGroupLineageSourceRelation,
    ObjectLabelsArtifactType,
)
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_set import ComponentSet
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractPlan,
    InvocationContractProvider,
)
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage


class _RuntimeInputContractProvider(InvocationContractProvider):
    def __init__(self, contract: CallableContract) -> None:
        self.contract = contract

    def __call__(self, invocation, step_context: ArtifactDeclarationStepContext):
        del invocation, step_context
        return InvocationContractPlan(self.contract)


def _runtime_input_provider(func, *specs: ArtifactSpec) -> InvocationContractProvider:
    contract = CallableContract.from_callable(func)
    return _RuntimeInputContractProvider(
        replace(
            contract,
            metadata=replace(contract.metadata, artifact_inputs=specs),
        )
    )


def _artifact_stage() -> PathPlannerArtifactStage:
    return PathPlannerArtifactStage(PathPlanner.__new__(PathPlanner))


def test_repeated_input_roles_compile_as_exact_edge_occurrences() -> None:
    objects = ArtifactSpec.input("objects", ObjectLabelsArtifactType)

    def consume_repeated_objects(image):
        return image

    storage_plan = ArtifactInputPlan(
        name=objects.name,
        path="/memory/objects.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    compiled = compile_function_pattern(
        consume_repeated_objects,
        {plan.ref(): plan for plan in (storage_plan,)},
        {},
        invocation_contract_provider=_runtime_input_provider(
            consume_repeated_objects,
            objects,
            objects,
        ),
    )
    invocation = next(compiled.iter_invocations())

    assert tuple(spec.ref() for spec in invocation.contract.artifact_inputs) == (
        objects.ref(),
        objects.ref(),
    )

    compiled = _artifact_stage().compile_invocation_input_edges(
        compiled,
        artifact_inputs={storage_plan.ref(): storage_plan},
        relation_source_scopes={
            objects.ref(): storage_plan.producer_group_scope(),
        },
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
    )
    invocation = next(compiled.iter_invocations())

    assert tuple(
        edge.key.input_index for edge in invocation.artifact_input_edges
    ) == (0, 1)
    assert tuple(edge.spec.ref() for edge in invocation.artifact_input_edges) == (
        objects.ref(),
        objects.ref(),
    )
    assert len(compiled.artifact_input_edges_by_key()) == 2


def test_repeated_input_roles_compile_distinct_relation_owned_edge_projections() -> None:
    source = ArtifactSpec.input("source", ObjectLabelsArtifactType)
    plain_objects = ArtifactSpec.input("objects", ObjectLabelsArtifactType)
    source_scoped_objects = ArtifactSpec.input(
        "objects",
        ObjectLabelsArtifactType,
        relations=(InputGroupLineageSourceRelation(source.ref()),),
    )

    def consume_ambiguous_objects(image):
        return image

    object_plan = ArtifactInputPlan(
        name=plain_objects.name,
        path="/memory/objects.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    source_plan = ArtifactInputPlan(
        name=source.name,
        path="/memory/source.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
    )
    compiled = compile_function_pattern(
        consume_ambiguous_objects,
        {plan.ref(): plan for plan in (object_plan, source_plan)},
        {},
        invocation_contract_provider=_runtime_input_provider(
            consume_ambiguous_objects,
            plain_objects,
            source_scoped_objects,
            source,
        ),
    )

    compiled = _artifact_stage().compile_invocation_input_edges(
        compiled,
        artifact_inputs={
            object_plan.ref(): object_plan,
            source_plan.ref(): source_plan,
        },
        relation_source_scopes={
            plain_objects.ref(): object_plan.producer_group_scope(),
            source.ref(): source_plan.producer_group_scope(),
        },
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
    )
    invocation = next(compiled.iter_invocations())

    assert tuple(edge.key.input_index for edge in invocation.artifact_input_edges) == (
        0,
        1,
        2,
    )
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        plain_objects,
        source_scoped_objects,
        source,
    )
    assert invocation.artifact_input_edges[0].projection.component_scopes == ()
    assert invocation.artifact_input_edges[1].projection.component_scopes == (
        source_plan.producer_group_scope(),
    )
    assert (
        invocation.artifact_input_edges[2].projection.producer_selection_scope
        == source_plan.producer_group_scope()
    )
