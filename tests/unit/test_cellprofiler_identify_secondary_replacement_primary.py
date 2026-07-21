from __future__ import annotations

import inspect

import numpy as np

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    NormalizedFunctionItem,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.secondary import (
    IdentifySecondaryObjectsModule,
    IdentifySecondaryObjectsReplacementPrimarySourceRelation,
    SecondaryMethod,
    identify_secondary_objects,
    identify_secondary_objects_with_replacement_primary,
)


def _module(*, discard_edge: bool, discard_primary: bool) -> ModuleBlock:
    return ModuleBlock(
        name="IdentifySecondaryObjects",
        module_num=1,
        setting_records=[
            ModuleSetting(IdentifySecondaryObjectsModule.input_image_setting, "DNA"),
            ModuleSetting(
                IdentifySecondaryObjectsModule.input_objects_setting, "Nuclei"
            ),
            ModuleSetting(
                IdentifySecondaryObjectsModule.output_objects_setting, "Cells"
            ),
            ModuleSetting(
                IdentifySecondaryObjectsModule.discard_edge_objects_setting,
                ("Yes" if discard_edge else "No"),
            ),
            ModuleSetting(
                IdentifySecondaryObjectsModule.discard_associated_primary_objects_setting,
                ("Yes" if discard_primary else "No"),
            ),
            ModuleSetting(
                IdentifySecondaryObjectsModule.replacement_primary_objects_setting,
                ("FilteredNuclei"),
            ),
        ],
    )


def _contract(module: ModuleBlock):
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    invocation_key = FunctionInvocationKey(
        function_name=str(IdentifySecondaryObjectsModule.function_name),
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )
    context = ArtifactDeclarationStepContext(
        step_index=0,
        available_artifacts=ArtifactSpecCollection((image, objects)),
        main_flow_artifacts=ArtifactSpecCollection((image,)),
        available_artifact_producers=artifact_producers_for_outputs(
            (objects,),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "identify_primary_objects",
                    invocation_key.group_key,
                    0,
                ),
            ),
        ),
    )
    return IdentifySecondaryObjectsModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=context,
    )


def _relationship_endpoints(contract) -> tuple[tuple[str, str], ...]:
    return tuple(
        (declaration.source.name, declaration.target.name)
        for _spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
    )


def _measurement_dependency_refs(contract):
    (measurement,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    return frozenset(
        relation.source
        for relation in measurement.relations
        if type(relation) is ArtifactSpecRelation
    )


def test_replacement_primary_inactive_topology_uses_original_contract() -> None:
    module = _module(discard_edge=False, discard_primary=True)
    contract = _contract(module)
    outputs = contract.artifact_outputs

    assert tuple(
        spec.name for spec in outputs.of_artifact_type(ObjectLabelsArtifactType)
    ) == ("Cells",)
    assert _relationship_endpoints(contract) == (("Nuclei", "Cells"),)
    expected_dependencies = frozenset(
        spec.ref()
        for spec in (
            *contract.artifact_inputs,
            *outputs.of_artifact_type(ObjectLabelsArtifactType),
            *outputs.of_artifact_type(ObjectLineageArtifactType),
        )
    )
    assert _measurement_dependency_refs(contract) == expected_dependencies

    selected = IdentifySecondaryObjectsModule.resolve_function(
        module,
        contract=contract,
        source_bindings=StepSourceBindingsConfig(),
    )
    assert selected is identify_secondary_objects
    assert (
        "_emit_replacement_primary_output" not in inspect.signature(selected).parameters
    )
    assert (
        "discard_associated_primary_objects"
        not in inspect.signature(selected).parameters
    )
    IdentifySecondaryObjectsModule.validate_callable_artifact_abi(selected, contract)


def test_replacement_primary_active_topology_declares_exact_dependencies() -> None:
    module = _module(discard_edge=True, discard_primary=True)
    contract = _contract(module)
    outputs = contract.artifact_outputs

    assert tuple(
        spec.name for spec in outputs.of_artifact_type(ObjectLabelsArtifactType)
    ) == ("Cells", "FilteredNuclei")
    assert _relationship_endpoints(contract) == (
        ("Nuclei", "Cells"),
        ("Nuclei", "FilteredNuclei"),
        ("FilteredNuclei", "Cells"),
    )
    expected_dependencies = frozenset(
        spec.ref()
        for spec in (
            *contract.artifact_inputs,
            *outputs.of_artifact_type(ObjectLabelsArtifactType),
            *outputs.of_artifact_type(ObjectLineageArtifactType),
        )
    )
    assert _measurement_dependency_refs(contract) == expected_dependencies

    selected = IdentifySecondaryObjectsModule.resolve_function(
        module,
        contract=contract,
        source_bindings=StepSourceBindingsConfig(),
    )
    assert selected is identify_secondary_objects_with_replacement_primary
    assert (
        "discard_associated_primary_objects"
        not in inspect.signature(selected).parameters
    )
    assert tuple(
        spec.name
        for spec, _relation in outputs.relation_refs(
            IdentifySecondaryObjectsReplacementPrimarySourceRelation
        )
    ) == ("FilteredNuclei",)
    IdentifySecondaryObjectsModule.validate_callable_artifact_abi(selected, contract)

    bound = IdentifySecondaryObjectsModule.bind_settings(
        module,
        binder=SettingsBinder(),
    )
    assert "discard_associated_primary_objects" not in bound.kwargs
    assert (
        bound.kwargs[
            IdentifySecondaryObjectsModule.replacement_primary_output_binding.require_parameter_name()
        ]
        == "FilteredNuclei"
    )


def test_replacement_primary_runtime_emits_typed_labels_and_relationships() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    primary_array = np.zeros(image.shape, dtype=np.int32)
    primary_array[4, 4] = 1
    primary_array[0, 0] = 2
    primary_labels = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=primary_array,
        unedited_labels=primary_array.copy(),
        small_removed_labels=primary_array.copy(),
        declared_object_ids=(1, 2),
    ).payload()

    result = identify_secondary_objects_with_replacement_primary.__wrapped__(
        image,
        primary_labels,
        method=SecondaryMethod.DISTANCE_N,
        distance_to_dilate=1,
        fill_holes=False,
        discard_edge_objects=True,
    )

    (
        _output_image,
        _measurements,
        primary_secondary,
        primary_replacement,
        replacement_secondary,
        secondary_output,
        replacement_primary_output,
    ) = result
    assert isinstance(secondary_output, ObjectLabelValue)
    assert isinstance(replacement_primary_output, ObjectLabelValue)
    assert (
        primary_secondary.source_ids,
        primary_secondary.target_ids,
    ) == ((1,), (1,))
    assert (
        primary_replacement.source_ids,
        primary_replacement.target_ids,
    ) == ((1,), (1,))
    assert (
        replacement_secondary.source_ids,
        replacement_secondary.target_ids,
    ) == ((1,), (1,))

    replacement_array = object_label_dense_array(
        replacement_primary_output,
        dtype=np.int32,
    )
    assert replacement_array[4, 4] == 1
    assert replacement_array[0, 0] == 0
    np.testing.assert_array_equal(
        replacement_primary_output.variant_data.labels_for_variant("unedited"),
        primary_array,
    )
    np.testing.assert_array_equal(
        replacement_primary_output.variant_data.labels_for_variant("small_removed"),
        primary_array,
    )

    primary_without_variants = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=primary_array,
        declared_object_ids=(1, 2),
    ).payload()
    variantless_result = (
        identify_secondary_objects_with_replacement_primary.__wrapped__(
            image,
            primary_without_variants,
            method=SecondaryMethod.DISTANCE_N,
            distance_to_dilate=1,
            fill_holes=False,
            discard_edge_objects=True,
        )
    )
    variantless_replacement = variantless_result[-1]
    assert isinstance(variantless_replacement, ObjectLabelValue)
    assert variantless_replacement.variant_data.unedited_labels is None
    assert variantless_replacement.variant_data.small_removed_labels is None
    assert (
        variantless_replacement.source_provenance
        == primary_without_variants.source_provenance
    )
    assert (
        variantless_replacement.source_spatial_domain
        == primary_without_variants.source_spatial_domain
    )
    assert (
        variantless_replacement.parent_image_source_voxel_spacing
        == primary_without_variants.parent_image_source_voxel_spacing
    )


def test_public_replacement_callable_reconstructs_topology_without_hidden_kwarg() -> (
    None
):
    callable_contract = CallableContract.from_callable(
        identify_secondary_objects_with_replacement_primary
    )
    invocation = NormalizedFunctionItem(
        key=FunctionInvocationKey.from_contract(
            callable_contract,
            DEFAULT_GROUP_KEY,
            0,
        ),
        contract=callable_contract,
    )
    image = ArtifactSpec.output("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    step_context = ArtifactDeclarationStepContext(
        step_name="IdentifySecondaryObjects",
        step_index=0,
        available_artifacts=ArtifactSpecCollection((image, objects)),
        main_flow_artifacts=ArtifactSpecCollection(
            (image.for_plan_type(ArtifactInputPlan),)
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            (objects,),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "identify_primary_objects",
                    invocation.key.group_key,
                    0,
                ),
            ),
        ),
    )
    blocks, block_consumed = (
        IdentifySecondaryObjectsModule.module_blocks_for_invocation(
            invocation=invocation,
            step_context=step_context,
        )
    )
    (numbered_blocks,), _next_module_num = (
        IdentifySecondaryObjectsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract, consumed = IdentifySecondaryObjectsModule.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=block_consumed,
        step_context=step_context,
    )

    assert len(blocks) == 1
    assert block_consumed == ()
    assert consumed == ()
    replacement_outputs = tuple(
        spec
        for spec, _relation in contract.artifact_outputs.relation_refs(
            IdentifySecondaryObjectsReplacementPrimarySourceRelation
        )
    )
    assert len(replacement_outputs) == 1
    assert (
        IdentifySecondaryObjectsModule.resolve_function(
            blocks[0],
            contract=contract,
            source_bindings=StepSourceBindingsConfig(),
        )
        is identify_secondary_objects_with_replacement_primary
    )
