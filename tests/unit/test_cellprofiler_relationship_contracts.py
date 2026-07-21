"""Focused compile and runtime tests for CellProfiler relationships."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.equivalence.policy import RuntimeMeasurementDialect
from openhcs.core.equivalence.relationships import (
    GenericRelationshipAggregateFeatureSemantics,
    RelationshipAggregateFeatureContext,
    RelationshipAggregateFeatureSemantics,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_measurements import (
    ObjectReferenceFeatureMarker,
    RuntimeMeasurementFeatureDeclaration,
)
from openhcs.core.runtime_relationships import DirectedObjectRelationshipPayload
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
)
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.setting_names import setting_values
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    DirectParentReferenceFeatureDeclaration,
    DirectParentReferenceMeasurementFeature,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    MeasureObjectNeighborsModule,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    RelateObjectsChildMeanFeatureDeclaration,
    RelateObjectsChildMeanMeasurementFeature,
    RelateObjectsDistanceAggregateFeatureSemantics,
    RelateObjectsModule,
    RelateObjectsRelationshipMeasurementRows,
    relate_objects,
    relate_objects_with_saved_children,
)


def _module(
    module_num: int,
    name: str,
    settings: dict[str, str],
) -> ModuleBlock:
    return ModuleBlock(
        name=name,
        module_num=module_num,
        setting_records=[
            ModuleSetting(_setting_name, _setting_value)
            for (_setting_name, _setting_value) in settings.items()
        ],
    )


def _contract(
    module_type,
    module: ModuleBlock,
    *,
    inputs: tuple[ArtifactSpec, ...],
):
    available_outputs = tuple(
        spec.for_plan_type(ArtifactOutputPlan) for spec in inputs
    )
    return module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            str(module_type.function_name),
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection(available_outputs),
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=artifact_producers_for_outputs(
                available_outputs,
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        DEFAULT_GROUP_KEY,
                        0,
                    ),
                ),
            ),
        ),
    )


def test_direct_parent_reference_declaration_owns_name_and_semantics() -> None:
    identity = DirectParentReferenceMeasurementFeature("Nuclei")
    feature_name = DirectParentReferenceFeatureDeclaration.feature_name(identity)

    assert feature_name == "Parent_Nuclei"
    assert DirectParentReferenceFeatureDeclaration.from_feature_name(feature_name) == (
        identity
    )
    assert RuntimeMeasurementFeatureDeclaration.feature_has_semantic_marker(
        feature_name,
        ObjectReferenceFeatureMarker,
    )
    assert (
        DirectParentReferenceFeatureDeclaration.from_feature_name("Parentage") is None
    )


def test_relate_objects_distance_declaration_owns_qualified_name_lookup() -> None:
    distance_feature = RelateObjectsModule.DistanceMeasurementFeature

    assert (
        distance_feature.matching_feature(
            "Distance_Centroid",
            parent_object_name="Nuclei",
        )
        is distance_feature.DISTANCE_CENTROID
    )
    assert (
        distance_feature.matching_feature(
            "Distance_Centroid_Nuclei",
            parent_object_name="Nuclei",
        )
        is distance_feature.DISTANCE_CENTROID
    )
    assert (
        distance_feature.matching_feature(
            "Distance_Unknown_Nuclei",
            parent_object_name="Nuclei",
        )
        is None
    )
    assert (
        distance_feature.DISTANCE_CENTROID.field_spec("Distance_Centroid_Nuclei").dtype
        is float
    )
    assert (
        distance_feature.DISTANCE_MINIMUM.field_spec("Distance_Minimum_Nuclei").dtype
        is float
    )


def test_relate_objects_distance_aggregation_is_dialect_owned() -> None:
    generic_context = RelationshipAggregateFeatureContext(
        source_name="Nuclei",
        target_name="Nucleoli",
        feature_name="Distance_Centroid_Nuclei",
        dialect=RuntimeMeasurementDialect(),
    )
    cellprofiler_context = RelationshipAggregateFeatureContext(
        source_name="Nuclei",
        target_name="Nucleoli",
        feature_name="Distance_Centroid_Nuclei",
        dialect=CELLPROFILER_MEASUREMENT_DIALECT,
    )

    assert (
        type(RelationshipAggregateFeatureSemantics.for_context(generic_context))
        is GenericRelationshipAggregateFeatureSemantics
    )
    cellprofiler_semantics = RelationshipAggregateFeatureSemantics.for_context(
        cellprofiler_context
    )
    assert type(cellprofiler_semantics) is (
        RelateObjectsDistanceAggregateFeatureSemantics
    )
    assert cellprofiler_semantics.required_child_feature_names(
        cellprofiler_context
    ) == ("distance_centroid", "distance_centroid_nuclei")
    assert cellprofiler_semantics.aggregate_feature_name(cellprofiler_context) == (
        "mean_nucleoli_distance_centroid"
    )


def test_parent_aggregation_query_uses_nominal_feature_semantics() -> None:
    feature_name = "Mean_Children_Distance_Centroid_Parents"
    identity = RelateObjectsChildMeanFeatureDeclaration.from_feature_name(feature_name)

    assert identity == RelateObjectsChildMeanMeasurementFeature(
        ("Children", "Distance", "Centroid", "Parents")
    )
    assert (
        RelateObjectsChildMeanFeatureDeclaration.feature_name(identity) == feature_name
    )
    assert (
        RelateObjectsChildMeanFeatureDeclaration
        in RuntimeMeasurementFeatureDeclaration.__registry__.values()
    )
    assert (
        RelateObjectsChildMeanFeatureDeclaration,
        identity,
    ) in RuntimeMeasurementFeatureDeclaration.matching_declarations(feature_name)

    eligibility = {
        feature_name: RelateObjectsModule.aggregates_child_measurement_feature(
            feature_name
        )
        for feature_name in (
            "Number_Object_Number",
            "Mean_Children_Distance_Centroid_Parents",
            "Parent_Parents",
            "Location_Center_X",
            "Intensity_MeanIntensity_DNA",
            "Parentage",
        )
    }

    assert eligibility == {
        "Number_Object_Number": True,
        "Mean_Children_Distance_Centroid_Parents": False,
        "Parent_Parents": False,
        "Location_Center_X": True,
        "Intensity_MeanIntensity_DNA": True,
        "Parentage": True,
    }


def test_recorded_relationship_contracts_own_exact_cpa_identity() -> None:
    cells = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    nuclei = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    mitochondria = ArtifactSpec.input("Mitochondria", ObjectLabelsArtifactType)
    nucleoli = ArtifactSpec.input("Nucleoli", ObjectLabelsArtifactType)
    neighbor_contracts = tuple(
        _contract(
            MeasureObjectNeighborsModule,
            _module(
                100 + module_number,
                "MeasureObjectNeighbors",
                {
                    "Select objects to measure": objects.name,
                    "Select neighboring objects to measure": objects.name,
                    "Retain the image of objects colored by numbers of neighbors?": "No",
                    "Retain the image of objects colored by percent of touching pixels?": "No",
                },
            ),
            inputs=(objects,),
        )
        for module_number, objects in (
            (18, cells),
            (19, nuclei),
            (20, mitochondria),
        )
    )
    relate_contracts = tuple(
        _contract(
            RelateObjectsModule,
            _module(
                100 + module_number,
                "RelateObjects",
                {
                    "Select the parent objects": parent.name,
                    "Select the child objects": child.name,
                    "Calculate child-parent distances?": "None",
                    "Calculate distances to other parents?": "No",
                    "Do you want to save the children with parents as a new object set?": (
                        "Yes" if module_number == 21 else "No"
                    ),
                    "Name the output object": "NucleoliChildObjects",
                },
            ),
            inputs=(parent, child),
        )
        for module_number, parent, child in (
            (21, nuclei, nucleoli),
            (22, cells, mitochondria),
        )
    )
    recorded_declarations = tuple(
        declaration
        for contract in (*neighbor_contracts, *relate_contracts)
        for _spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if _spec.artifact_type is RelationshipsArtifactType
    )
    assert tuple(
        (
            declaration.producer_module_number,
            declaration.relationship_type,
            declaration.source.name,
            declaration.target.name,
        )
        for declaration in recorded_declarations
    ) == (
        (118, "Neighbors", "Cells", "Cells"),
        (119, "Neighbors", "Nuclei", "Nuclei"),
        (120, "Neighbors", "Mitochondria", "Mitochondria"),
        (121, "Parent", "Nuclei", "Nucleoli"),
        (121, "Child", "Nucleoli", "Nuclei"),
        (122, "Parent", "Cells", "Mitochondria"),
        (122, "Child", "Mitochondria", "Cells"),
    )
    internal_lineage = relate_contracts[0].artifact_outputs.of_artifact_type(
        ObjectLineageArtifactType
    )
    assert len(internal_lineage) == 1
    ((_spec, internal_declaration),) = ArtifactSpecCollection(
        internal_lineage
    ).relation_refs(ObjectRelationshipDeclaration)
    assert internal_declaration.relationship_type == "parent_child"


def test_relationship_input_selection_uses_declared_endpoint_refs() -> None:
    parent = ArtifactSpec.output("ParentObjects", ObjectLabelsArtifactType)
    child = ArtifactSpec.output("ChildObjects", ObjectLabelsArtifactType)
    declaration = ObjectRelationshipDeclaration(
        source=parent.ref(),
        target=child.ref(),
        producer_module_number=7,
        relationship_type="parent_child",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=0,
        target_runtime_slice_offset=0,
    )
    relationship = ArtifactSpec.output(
        "opaque-artifact-identity",
        ObjectLineageArtifactType,
        relations=(declaration,),
    ).for_plan_type(ArtifactInputPlan)

    ((selected_spec, selected_declaration),) = ArtifactSpecCollection(
        (relationship,)
    ).relation_refs(ObjectRelationshipDeclaration)
    assert selected_spec == relationship
    assert (
        selected_declaration.source,
        selected_declaration.target,
    ) == (
        parent.ref().for_plan_type(ArtifactInputPlan),
        child.ref().for_plan_type(ArtifactInputPlan),
    )


def test_relate_objects_parent_means_compile_prior_child_measurement_inputs() -> None:
    parent = ArtifactSpec.input("ParentObjects", ObjectLabelsArtifactType)
    child = ArtifactSpec.input("ChildObjects", ObjectLabelsArtifactType)
    child_measurements = ArtifactSpec.output(
        "child-measurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(child.ref()),),
    )
    parent_measurements = ArtifactSpec.output(
        "parent-measurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(parent.ref()),),
    )
    contract = _contract(
        RelateObjectsModule,
        _module(
            8,
            "RelateObjects",
            {
                "Select the parent objects": parent.name,
                "Select the child objects": child.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
                "Calculate per-parent means for all child measurements?": "Yes",
            },
        ),
        inputs=(parent, child, child_measurements, parent_measurements),
    )

    assert tuple(
        spec.name
        for spec in contract.artifact_inputs.of_artifact_type(
            MeasurementsArtifactType
        )
    ) == (child_measurements.name,)

    disabled_contract = _contract(
        RelateObjectsModule,
        _module(
            8,
            "RelateObjects",
            {
                "Select the parent objects": parent.name,
                "Select the child objects": child.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
                "Calculate per-parent means for all child measurements?": "No",
            },
        ),
        inputs=(parent, child, child_measurements, parent_measurements),
    )
    assert (
        disabled_contract.artifact_inputs.of_artifact_type(
            MeasurementsArtifactType
        )
        == ()
    )


def test_relate_objects_parent_means_reuse_qualified_distance_features() -> None:
    rows = object.__new__(RelateObjectsRelationshipMeasurementRows)
    (row,) = rows.parent_mean_distance_rows(
        parent_object_name="Nuclei",
        child_object_name="Cells",
        centroid_child_feature_name="Distance_Centroid_Nuclei",
        minimum_child_feature_name="Distance_Minimum_Nuclei",
        pairs=((1, 1),),
        centroid_distances=np.asarray((2.5,)),
        minimum_distances=np.asarray((1.5,)),
        slice_index=None,
    )

    assert row["Mean_Cells_Distance_Centroid_Nuclei"] == 2.5
    assert row["Mean_Cells_Distance_Minimum_Nuclei"] == 1.5
    assert "Mean_Cells_Distance_Centroid" not in row
    assert "Mean_Cells_Distance_Minimum" not in row


def test_relationship_runtime_value_preserves_declared_producer_module_number() -> None:
    parent = ArtifactSpec.output("ParentObjects", ObjectLabelsArtifactType)
    child = ArtifactSpec.output("ChildObjects", ObjectLabelsArtifactType)
    declaration = ObjectRelationshipDeclaration(
        source=parent.ref(),
        target=child.ref(),
        producer_module_number=17,
        relationship_type="parent_child",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=0,
        target_runtime_slice_offset=0,
    )

    relationship = ObjectRelationship.from_payload(
        name=declaration.artifact_name(),
        declaration=declaration,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(1,),
            target_ids=(2,),
            slice_indices=(0,),
            slice_count=1,
        ),
    )

    assert relationship.declaration.producer_module_number == 17
    assert (
        relationship.project_runtime_slice(0).declaration.producer_module_number == 17
    )


def test_sparse_public_relate_objects_lowering_reconstructs_exact_contract(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "sparse-relate.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Cells
RelateObjects:[module_num:4|enabled:True]
    Select the parent objects:Nuclei
    Select the child objects:Cells
    Calculate child-parent distances?:None
    Calculate per-parent means for all child measurements?:No
    Calculate distances to other parents?:No
    Do you want to save the children with parents as a new object set?:No
""",
        encoding="utf-8",
    )

    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    relate_step = steps[-1]
    assert relate_step.name == "RelateObjects"
    (invocation,) = tuple(normalize_function_pattern(relate_step.func).iter_items())
    object_specs = (
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
        ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
    )
    producer_keys = (
        FunctionInvocationKey(
            "identify_primary_objects",
            DEFAULT_GROUP_KEY,
            0,
        ),
    )
    step_context = ArtifactDeclarationStepContext(
        step_index=2,
        available_artifacts=ArtifactSpecCollection(object_specs),
        available_artifact_producers=artifact_producers_for_outputs(
            object_specs,
            groups=(None,),
            invocation_keys=producer_keys,
        ),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )
    blocks, consumed_names = RelateObjectsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    (numbered_blocks,), _next_module_num = (
        RelateObjectsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=3,
        )
    )
    assert blocks[0].module_num == 0
    assert numbered_blocks[0].module_num == 3
    assert setting_values(
        numbered_blocks[0], RelateObjectsModule.other_parent_distances_setting
    ) == ("No",)
    contract, _consumed_names = RelateObjectsModule.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=step_context,
    )
    declarations = tuple(
        declaration
        for spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if spec.artifact_type is RelationshipsArtifactType
    )
    assert tuple(
        (
            declaration.producer_module_number,
            declaration.relationship_type,
            declaration.source.name,
            declaration.target.name,
        )
        for declaration in declarations
    ) == (
        (3, "Parent", "Nuclei", "Cells"),
        (3, "Child", "Cells", "Nuclei"),
    )


def test_saved_child_relate_objects_lowers_with_object_main_flow(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "saved-child-relate.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Cells
RelateObjects:[module_num:4|enabled:True]
    Select the parent objects:Nuclei
    Select the child objects:Cells
    Calculate child-parent distances?:Both
    Calculate per-parent means for all child measurements?:Yes
    Calculate distances to other parents?:No
    Do you want to save the children with parents as a new object set?:Yes
    Name the output object:SavedCells
""",
        encoding="utf-8",
    )

    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    relate_step = steps[-1]
    (invocation,) = tuple(normalize_function_pattern(relate_step.func).iter_items())

    assert invocation.contract.resolve_runtime_callable() is (
        relate_objects_with_saved_children
    )
    assert "name_the_output_object" not in invocation.kwargs_dict

    inputs = (
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
        ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_index=2,
        available_artifacts=ArtifactSpecCollection(inputs),
        available_artifact_producers=artifact_producers_for_outputs(
            inputs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "identify_primary_objects",
                    DEFAULT_GROUP_KEY,
                    0,
                ),
            ),
        ),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )
    blocks, consumed_names = RelateObjectsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        RelateObjectsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=3,
        )
    )
    contract, _consumed_names = RelateObjectsModule.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=context,
    )

    (saved_child_output,) = contract.main_flow_outputs
    assert saved_child_output.artifact_type is ObjectLabelsArtifactType
    assert saved_child_output.ref().plan_type is ArtifactOutputPlan
    assert contract.artifact_outputs[0].ref() == saved_child_output.ref()
    assert tuple(
        spec.artifact_type for spec in contract.artifact_outputs
    ) == (
        ObjectLabelsArtifactType,
        RelationshipsArtifactType,
        RelationshipsArtifactType,
        ObjectLineageArtifactType,
        MeasurementsArtifactType,
    )


def test_relate_objects_emits_native_forward_and_reverse_pairs() -> None:
    parent_labels = np.zeros((5, 5), dtype=np.int32)
    parent_labels[1:4, 1:4] = 1
    child_labels = np.zeros_like(parent_labels)
    child_labels[2, 2] = 1

    _output, parent_to_child, child_to_parent, _measurements = (
        relate_objects.__wrapped__(
            np.zeros_like(parent_labels, dtype=np.float32),
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=parent_labels)
            ),
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=child_labels)
            ),
        )
    )

    assert (parent_to_child.source_ids, parent_to_child.target_ids) == ((1,), (1,))
    assert (child_to_parent.source_ids, child_to_parent.target_ids) == ((1,), (1,))
