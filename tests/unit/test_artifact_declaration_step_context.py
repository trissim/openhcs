"""Forward artifact-state behavior owned by ArtifactDeclarationStepContext."""

from dataclasses import fields

from openhcs.constants.constants import GroupBy
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.function_patterns import DEFAULT_GROUP_KEY, FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import (
    ArtifactGraph,
    artifact_producers_for_outputs,
)
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    StepSourceBindingsConfig,
)


def test_context_owns_exact_step_artifact_projection_facts() -> None:
    field_names = {field.name for field in fields(ArtifactDeclarationStepContext)}

    assert "group_by" in field_names
    assert "input_source" in field_names
    assert "invocation_ordinal" not in field_names
    assert "processing_config" not in field_names
    assert ArtifactDeclarationStepContext.empty().group_by is GroupBy.NONE
    assert (
        ArtifactDeclarationStepContext.empty().input_source
        is InputSource.PREVIOUS_STEP
    )


def test_context_pipeline_start_sources_replace_main_flow() -> None:
    source_image = ArtifactSpec.input("DNA", ImageArtifactType)
    upstream_image = ArtifactSpec.input("FilteredDNA", ImageArtifactType)
    measurement = ArtifactSpec.input("Measurements", MeasurementsArtifactType)

    advanced = ArtifactDeclarationStepContext(
        source_bindings=StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        ),
        input_source=InputSource.PIPELINE_START,
        main_flow_artifacts=ArtifactSpecCollection((upstream_image,)),
    ).with_source_declarations((source_image, measurement))

    assert advanced.available_artifacts.specs == (source_image, measurement)
    assert advanced.main_flow_artifacts.specs == (source_image,)
    assert advanced.available_artifact_producers == ()


def test_context_pipeline_start_main_flow_uses_declared_source_subset() -> None:
    dna = ArtifactSpec.input("DNA", ImageArtifactType)

    advanced = ArtifactDeclarationStepContext(
        source_bindings=StepSourceBindingsConfig(
            bindings=(
                NamedSourceBinding(alias="DNA"),
                NamedSourceBinding(alias="Actin"),
            ),
        ),
        input_source=InputSource.PIPELINE_START,
    ).with_source_declarations((dna,))

    assert advanced.available_artifacts.specs == (dna,)
    assert advanced.main_flow_artifacts.specs == (dna,)


def test_context_supplemental_step_sources_do_not_replace_previous_main_flow() -> None:
    source_image = ArtifactSpec.input("DNA", ImageArtifactType)
    upstream_image = ArtifactSpec.input("FilteredDNA", ImageArtifactType)

    advanced = ArtifactDeclarationStepContext(
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(alias="DNA"),
            ),
        ),
        input_source=InputSource.PREVIOUS_STEP,
        main_flow_artifacts=ArtifactSpecCollection((upstream_image,)),
    ).with_source_declarations((source_image,))

    assert advanced.available_artifacts.specs == (source_image,)
    assert advanced.main_flow_artifacts.specs == (upstream_image,)


def test_context_preserves_main_flow_for_previous_step_source_declarations() -> None:
    prior_image = ArtifactSpec.input("Prior", ImageArtifactType)
    source_image = ArtifactSpec.input("DNA", ImageArtifactType)

    advanced = ArtifactDeclarationStepContext(
        input_source=InputSource.PREVIOUS_STEP,
        main_flow_artifacts=ArtifactSpecCollection((prior_image,)),
    ).with_source_declarations((source_image,))

    assert advanced.available_artifacts.specs == (source_image,)
    assert advanced.main_flow_artifacts.specs == (prior_image,)


def test_context_advances_one_artifact_graph_without_mutating_prior_state() -> None:
    source_image = ArtifactSpec.input("DNA", ImageArtifactType)
    output_image = ArtifactSpec.output("CorrectedDNA", ImageArtifactType)
    measurement = ArtifactSpec.output("Area", MeasurementsArtifactType)
    original = ArtifactDeclarationStepContext(
        available_artifacts=ArtifactSpecCollection((source_image,)),
        main_flow_artifacts=ArtifactSpecCollection((source_image,)),
    )
    invocation_key = FunctionInvocationKey(
        function_name="measure",
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )
    graph = ArtifactGraph(
        producers=artifact_producers_for_outputs(
            (output_image, measurement),
            groups=(None,),
            invocation_keys=(invocation_key,),
        )
    )
    next_main_flow = ArtifactSpecCollection(
        (output_image.for_plan_type(ArtifactInputPlan),)
    )

    advanced = original.advance_artifact_graph(
        graph,
        main_flow_artifacts=next_main_flow,
    )

    assert original.available_artifacts.specs == (source_image,)
    assert original.available_artifact_producers == ()
    assert advanced.available_artifacts.specs == (
        source_image,
        output_image,
        measurement,
    )
    assert advanced.main_flow_artifacts.specs == next_main_flow.specs
    assert advanced.available_artifact_producers == graph.producers


def test_context_preserves_grouped_invocation_ownership_exactly() -> None:
    output = ArtifactSpec.output("Corrected", ImageArtifactType)
    invocation_keys = tuple(
        FunctionInvocationKey(
            function_name="correct",
            group_key=group_key,
            position=0,
        )
        for group_key in ("1", "2")
    )
    graph = ArtifactGraph(
        producers=artifact_producers_for_outputs(
            (output,),
            groups=("1", "2"),
            invocation_keys=invocation_keys,
        )
    )

    advanced = ArtifactDeclarationStepContext.empty().advance_artifact_graph(
        graph,
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    (producer,) = advanced.available_artifact_producers
    assert producer.groups == ("1", "2")
    assert producer.invocation_keys == invocation_keys


def test_context_replaces_active_artifact_and_exact_producer() -> None:
    source = ArtifactSpec.input("Image", ImageArtifactType)
    first_output = ArtifactSpec.output("Image", ImageArtifactType)
    second_output = ArtifactSpec.output(
        "Image",
        ImageArtifactType,
        required=False,
    )
    first_key = FunctionInvocationKey("first", DEFAULT_GROUP_KEY, 0)
    second_key = FunctionInvocationKey("second", DEFAULT_GROUP_KEY, 0)
    first_graph = ArtifactGraph(
        producers=artifact_producers_for_outputs(
            (first_output,),
            groups=(None,),
            invocation_keys=(first_key,),
        )
    )
    second_graph = ArtifactGraph(
        producers=artifact_producers_for_outputs(
            (second_output,),
            groups=(None,),
            invocation_keys=(second_key,),
        )
    )

    context = (
        ArtifactDeclarationStepContext(
            available_artifacts=ArtifactSpecCollection((source,)),
        )
        .advance_artifact_graph(
            first_graph,
            main_flow_artifacts=ArtifactSpecCollection(()),
        )
        .advance_artifact_graph(
            second_graph,
            main_flow_artifacts=ArtifactSpecCollection(()),
        )
    )

    assert context.available_artifacts.specs == (second_output,)
    assert context.available_artifact_producers == second_graph.producers
