"""Regression tests for prior-measurement output group ownership."""

from __future__ import annotations

import pytest

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputGroupLineageSourceRelation,
    MeasurementsArtifactType,
)
from openhcs.core.function_patterns import DEFAULT_GROUP_KEY, FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathModule,
)


def _calculate_math_output(
    measurement_input: ArtifactSpec,
    *,
    output_name: str = "CalculateMath_8_measurements",
) -> ArtifactSpec:
    artifact_inputs = ArtifactSpecCollection((measurement_input,))
    relations = CalculateMathModule.measurement_output_relations(
        ModuleBlock(name="CalculateMath", module_num=8, setting_records=[]),
        invocation_key=FunctionInvocationKey(
            "calculate_math",
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name="CalculateMath",
            step_index=5,
        ),
        artifact_inputs=artifact_inputs,
    )
    return ArtifactSpec.output(
        output_name,
        MeasurementsArtifactType,
        relations=relations,
    )


def test_prior_measurement_output_group_scope_stays_on_direct_input() -> None:
    raw_gfp = ArtifactSpec.input("rawGFP", ImageArtifactType)
    measurement_input = ArtifactSpec.input(
        "MeasureObjectIntensity_4_measurements",
        MeasurementsArtifactType,
        relations=(InputGroupLineageSourceRelation(raw_gfp.ref()),),
    )

    output = _calculate_math_output(measurement_input)
    contract_specs = ArtifactSpecCollection((measurement_input, output))

    assert measurement_input.group_scope_sources() == (raw_gfp.ref(),)
    assert output.group_scope_sources() == (measurement_input.ref(),)
    contract_specs.validate_registered_relation_refs(
        owner_name="CalculateMathModule",
        relation_specs=(output,),
    )


def test_measurement_output_validation_rejects_undeclared_external_source() -> None:
    raw_gfp = ArtifactSpec.input("rawGFP", ImageArtifactType)
    output = ArtifactSpec.output(
        "CalculateMath_8_measurements",
        MeasurementsArtifactType,
        relations=(GroupLineageSourceRelation(raw_gfp.ref()),),
    )

    with pytest.raises(ValueError, match="unknown artifact specs"):
        ArtifactSpecCollection((output,)).validate_registered_relation_refs(
            owner_name="CalculateMathModule",
            relation_specs=(output,),
        )
