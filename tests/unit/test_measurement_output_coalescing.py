"""Compiled-pattern regressions for repeated exact measurement outputs."""

from dataclasses import replace

import pytest

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    FunctionInvocationKey,
)


def _compiled_measurement_pattern(
    local_outputs: tuple[ArtifactSpec, ...],
) -> CompiledFunctionPattern:
    def measure():
        return None

    base_contract = CallableContract.from_callable(measure)
    output_plan = ArtifactOutputPlan(
        name=local_outputs[0].name,
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    invocations = tuple(
        CompiledFunctionInvocation(
            key=FunctionInvocationKey.from_contract(
                contract,
                DEFAULT_GROUP_KEY,
                position,
            ),
            contract=contract,
            artifact_output_plans=(output_plan,),
        )
        for position, output in enumerate(local_outputs)
        for contract in (
            replace(
                base_contract,
                metadata=replace(
                    base_contract.metadata,
                    artifact_outputs=(output,),
                ),
            ),
        )
    )
    return CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key=DEFAULT_GROUP_KEY,
                invocations=invocations,
            ),
        ),
        is_grouped=False,
    )


@pytest.mark.parametrize(
    "source_specs_by_invocation",
    (
        (
            (
                ArtifactSpec.input("Protein", ImageArtifactType),
                ArtifactSpec.input("Cells", ObjectLabelsArtifactType),
            ),
            (
                ArtifactSpec.input("DNA", ImageArtifactType),
                ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
            ),
            (
                ArtifactSpec.input("Protein", ImageArtifactType),
                ArtifactSpec.input("Cytoplasm", ObjectLabelsArtifactType),
            ),
        ),
        tuple(
            (
                *(
                    ArtifactSpec.input(image_name, ImageArtifactType)
                    for image_name in ("Hoechst", "ER", "Syto", "Golgi", "Mito")
                ),
                ArtifactSpec.input(object_name, ObjectLabelsArtifactType),
            )
            for object_name in (
                "Nuclei",
                "Cells",
                "Cytoplasm",
                "GolgiObjects",
                "MitoObjects",
            )
        ),
    ),
    ids=("scalar-image-object-invocations", "image-tuple-object-invocations"),
)
def test_compiled_pattern_coalesces_repeated_exact_measurement_outputs(
    source_specs_by_invocation: tuple[tuple[ArtifactSpec, ...], ...],
) -> None:
    local_outputs = tuple(
        ArtifactSpec.output(
            "Measurements",
            MeasurementsArtifactType,
            relations=tuple(
                ArtifactSpecRelation(source.ref()) for source in source_specs
            ),
        )
        for source_specs in source_specs_by_invocation
    )
    merged_relations = tuple(
        dict.fromkeys(
            relation
            for output in local_outputs
            for relation in output.relations
        )
    )
    compiled = _compiled_measurement_pattern(local_outputs)

    assert compiled.coalesced_artifact_output_specs() == (
        replace(local_outputs[0], relations=merged_relations),
    )
    assert tuple(
        invocation.contract.artifact_outputs[0].relations
        for invocation in compiled.iter_invocations()
    ) == tuple(output.relations for output in local_outputs)


def test_compiled_pattern_requires_exact_selected_output_ref() -> None:
    output = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
    )
    compiled = _compiled_measurement_pattern((output,))
    (group,) = compiled.groups
    (invocation,) = group.invocations
    compiled = replace(
        compiled,
        groups=(
            replace(
                group,
                invocations=(replace(invocation, artifact_output_plans=()),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires one exact selected plan"):
        compiled.coalesced_artifact_output_specs()
