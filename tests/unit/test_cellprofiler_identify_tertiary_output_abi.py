from __future__ import annotations

from typing import get_args, get_type_hints

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
)
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.secondary import (
    IdentifyTertiaryObjectsModule,
    identify_tertiary_objects,
)


def _tertiary_contract(
    *,
    larger_name: str,
    smaller_name: str,
):
    output_specs = tuple(
        ArtifactSpec.output(name, ObjectLabelsArtifactType)
        for name in dict.fromkeys((larger_name, smaller_name))
    )
    invocation_key = FunctionInvocationKey(
        "identify_tertiary_objects",
        "default",
        0,
    )
    module = ModuleBlock(
        name="IdentifyTertiaryObjects",
        module_num=5,
        setting_records=[
            ModuleSetting(
                IdentifyTertiaryObjectsModule.larger_objects_setting,
                larger_name,
            ),
            ModuleSetting(
                IdentifyTertiaryObjectsModule.smaller_objects_setting,
                smaller_name,
            ),
            ModuleSetting(
                IdentifyTertiaryObjectsModule.output_objects_setting,
                "Cytoplasm",
            ),
        ],
    )
    return IdentifyTertiaryObjectsModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection(output_specs),
            available_artifact_producers=artifact_producers_for_outputs(
                output_specs,
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", "default", 0),
                ),
            ),
        ),
    )


def test_tertiary_same_object_probe_preserves_both_input_and_output_roles() -> None:
    contract = _tertiary_contract(larger_name="Cells", smaller_name="Cells")

    assert tuple(
        (spec.name, spec.parameter_name) for spec in contract.artifact_inputs
    ) == (
        ("Cells", "secondary_labels"),
        ("Cells", "primary_labels"),
    )

    relationships = contract.artifact_outputs.of_artifact_type(
        ObjectLineageArtifactType
    )
    assert len(relationships) == 2
    assert relationships[0] == relationships[1]
    assert len(get_args(get_type_hints(identify_tertiary_objects)["return"])) == (
        len(contract.trailing_return_output_specs) + 1
    )


def test_tertiary_relationship_roles_preserve_exact_external_endpoints() -> None:
    contract = _tertiary_contract(larger_name="Cells", smaller_name="Nuclei")

    relationships = contract.artifact_outputs.of_artifact_type(
        ObjectLineageArtifactType
    )
    declarations = tuple(
        relation
        for spec in relationships
        for relation in spec.relations
        if isinstance(relation, ObjectRelationshipDeclaration)
    )

    assert tuple(
        (
            declaration.source.name,
            declaration.target.name,
            declaration.relationship_type,
            declaration.source_role,
            declaration.target_role,
        )
        for declaration in declarations
    ) == (
        ("Cells", "Cytoplasm", "parent_child", "parent", "child"),
        ("Nuclei", "Cytoplasm", "parent_child", "parent", "child"),
    )
    (measurement,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert measurement.group_scope_sources() == (
        ArtifactSpec.input("Cells", ObjectLabelsArtifactType).ref(),
    )
