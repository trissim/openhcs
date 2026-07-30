from __future__ import annotations

from inspect import unwrap
from types import SimpleNamespace

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomainScope,
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import setting_values
from openhcs.processing.backends.cellprofiler.relationships import (
    RelateObjectsDistanceMethod,
    RelateObjectsModule,
    RelateObjectsRelationshipMeasurementRows,
    relate_objects,
)
from openhcs.processing.backends.cellprofiler.tracking import (
    TrackObjectsModule,
    TrackObjectsResult,
    TrackingMethod,
    track_objects,
)


def _invocation_key(function_name: str) -> FunctionInvocationKey:
    return FunctionInvocationKey(function_name, DEFAULT_GROUP_KEY, 0)


def _context(
    *artifacts: ArtifactSpec,
    step_index: int = 2,
    main_flow: tuple[ArtifactSpec, ...] = (),
) -> ArtifactDeclarationStepContext:
    produced = tuple(
        artifact for artifact in artifacts if artifact.plan_type is ArtifactOutputPlan
    )
    return ArtifactDeclarationStepContext(
        step_index=step_index,
        available_artifacts=ArtifactSpecCollection(artifacts),
        main_flow_artifacts=ArtifactSpecCollection(main_flow),
        available_artifact_producers=artifact_producers_for_outputs(
            produced,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
            ),
        ),
    )


def _relationship_artifact(
    parent: ArtifactSpec,
    child: ArtifactSpec,
    *,
    module_number: int,
) -> ArtifactSpec:
    declaration = ObjectRelationshipDeclaration(
        source=parent.ref(),
        target=child.ref(),
        producer_module_number=module_number,
        relationship_type="parent_child",
        source_role="parent",
        target_role="child",
        source_id_field="parent_id",
        target_id_field="child_id",
        source_runtime_slice_offset=0,
        target_runtime_slice_offset=0,
    )
    return ArtifactSpec.output(
        declaration.artifact_name(),
        ObjectLineageArtifactType,
        relations=(declaration,),
    )


def _relate_module(
    *,
    distance_method: str,
    calculate_other_parent_distances: str,
    other_parents: tuple[str, ...] = (),
) -> ModuleBlock:
    return ModuleBlock(
        name="RelateObjects",
        module_num=3,
        metadata={"variable_revision_number": 5},
        setting_records=[
            ModuleSetting("Select the parent objects", "Parents"),
            ModuleSetting("Select the child objects", "Children"),
            ModuleSetting("Calculate child-parent distances?", distance_method),
            ModuleSetting(
                "Calculate distances to other parents?",
                calculate_other_parent_distances,
            ),
            *(
                ModuleSetting("Parent name", parent_name)
                for parent_name in other_parents
            ),
            ModuleSetting(
                "Calculate per-parent means for all child measurements?",
                "No",
            ),
            ModuleSetting(
                "Do you want to save the children with parents as a new object set?",
                "No",
            ),
        ],
    )


def _relate_contract(
    module: ModuleBlock,
    *artifacts: ArtifactSpec,
):
    return RelateObjectsModule.callable_contract(
        module=module,
        invocation_key=_invocation_key("relate_objects"),
        step_context=_context(*artifacts),
    )


def test_relate_objects_active_repeated_parents_use_exact_relationship_inputs() -> None:
    parents = ArtifactSpec.output("Parents", ObjectLabelsArtifactType)
    children = ArtifactSpec.output("Children", ObjectLabelsArtifactType)
    grandparents = ArtifactSpec.output("Grandparents", ObjectLabelsArtifactType)
    siblings = ArtifactSpec.output("Siblings", ObjectLabelsArtifactType)
    grandparent_relationship = _relationship_artifact(
        grandparents,
        parents,
        module_number=1,
    )
    sibling_relationship = _relationship_artifact(
        parents,
        siblings,
        module_number=2,
    )

    contract = _relate_contract(
        _relate_module(
            distance_method="Both",
            calculate_other_parent_distances="Yes",
            other_parents=(grandparents.name, siblings.name),
        ),
        parents,
        children,
        grandparents,
        siblings,
        grandparent_relationship,
        sibling_relationship,
    )

    object_inputs = contract.artifact_inputs.of_artifact_type(ObjectLabelsArtifactType)
    assert tuple(spec.name for spec in object_inputs) == (
        "Parents",
        "Children",
        "Grandparents",
        "Siblings",
    )
    relationship_inputs = contract.artifact_inputs.relation_refs(
        ObjectRelationshipDeclaration
    )
    assert tuple(spec.name for spec, _declaration in relationship_inputs) == (
        grandparent_relationship.name,
        sibling_relationship.name,
    )

    relationship_declarations = contract.artifact_outputs.relation_refs(
        ObjectRelationshipDeclaration
    )
    assert tuple(
        (
            declaration.relationship_type,
            declaration.source.name,
            declaration.target.name,
        )
        for _spec, declaration in relationship_declarations
    ) == (
        ("Parent", "Parents", "Children"),
        ("Child", "Children", "Parents"),
    )
    measurement = contract.artifact_outputs.of_artifact_type(MeasurementsArtifactType)[
        0
    ]
    assert relationship_declarations[0][0].ref() in {
        relation.source for relation in measurement.relations
    }


@pytest.mark.parametrize(
    ("distance_method", "calculate_other_parent_distances"),
    (("Both", "No"), ("None", "Yes")),
)
def test_relate_objects_inactive_repeated_parent_rows_do_not_change_contract(
    distance_method: str,
    calculate_other_parent_distances: str,
) -> None:
    parents = ArtifactSpec.output("Parents", ObjectLabelsArtifactType)
    children = ArtifactSpec.output("Children", ObjectLabelsArtifactType)
    stale_parent = ArtifactSpec.output("StaleParent", ObjectLabelsArtifactType)

    contract = _relate_contract(
        _relate_module(
            distance_method=distance_method,
            calculate_other_parent_distances=calculate_other_parent_distances,
            other_parents=(stale_parent.name,),
        ),
        parents,
        children,
        stale_parent,
    )

    assert tuple(
        spec.name
        for spec in contract.artifact_inputs.of_artifact_type(ObjectLabelsArtifactType)
    ) == ("Parents", "Children")


class _DistanceRows(RelateObjectsRelationshipMeasurementRows):
    def distance_method(self) -> RelateObjectsDistanceMethod:
        return self.request.distance_method

    def per_parent_means_enabled(self) -> bool:
        return False

    def object_labels(self, spec, *, slice_index=None, slice_count=None):
        del slice_index, slice_count
        return self.request.labels[spec.name]


@pytest.mark.parametrize(
    ("distance_method", "expected_feature", "excluded_feature"),
    (
        (
            RelateObjectsDistanceMethod.CENTROID,
            "Distance_Centroid_Grandparents",
            "Distance_Minimum_Grandparents",
        ),
        (
            RelateObjectsDistanceMethod.MINIMUM,
            "Distance_Minimum_Grandparents",
            "Distance_Centroid_Grandparents",
        ),
    ),
)
def test_relate_objects_distance_rows_declare_only_selected_exact_feature(
    distance_method: RelateObjectsDistanceMethod,
    expected_feature: str,
    excluded_feature: str,
) -> None:
    parent = ArtifactSpec.input("Grandparents", ObjectLabelsArtifactType)
    child = ArtifactSpec.input("Children", ObjectLabelsArtifactType)
    parent_labels = np.zeros((5, 5), dtype=np.int32)
    parent_labels[1:4, 1:4] = 1
    child_labels = np.zeros_like(parent_labels)
    child_labels[2, 2] = 1
    projector = _DistanceRows(
        SimpleNamespace(
            distance_method=distance_method,
            labels={parent.name: parent_labels, child.name: child_labels},
        )
    )

    rows = projector.distance_rows_for_pairs(
        parent_spec=parent,
        child_spec=child,
        pairs=((1, 1),),
        slice_index=None,
    )
    (row,) = tuple(rows.iter_row_mappings())

    assert expected_feature in row
    assert excluded_feature not in row
    fields = {field.name: field for field in rows.fields}
    assert expected_feature in fields
    assert fields[expected_feature].dtype is float
    assert excluded_feature not in fields


def test_public_relate_objects_reconstructs_repeated_parent_contract() -> None:
    parents = ArtifactSpec.output("Parents", ObjectLabelsArtifactType)
    children = ArtifactSpec.output("Children", ObjectLabelsArtifactType)
    grandparents = ArtifactSpec.output("Grandparents", ObjectLabelsArtifactType)
    supporting_relationship = _relationship_artifact(
        grandparents,
        parents,
        module_number=1,
    )
    kwargs = {
        (
            RelateObjectsModule.parent_objects_binding.require_parameter_name()
        ): parents.name,
        (
            RelateObjectsModule.child_objects_binding.require_parameter_name()
        ): children.name,
        RelateObjectsModule.other_parent_objects_binding.require_parameter_name(): (
            grandparents.name,
        ),
        "calculate_distances": RelateObjectsDistanceMethod.CENTROID,
        "calculate_distances_to_other_parents": True,
    }
    step = FunctionStep(func=(relate_objects, kwargs))
    invocation = next(normalize_function_pattern(step.func).iter_items())
    context = _context(
        parents,
        children,
        grandparents,
        supporting_relationship,
    )

    blocks, consumed = RelateObjectsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    assert len(blocks) == 1
    assert set(consumed) == {
        RelateObjectsModule.parent_objects_binding.require_parameter_name(),
        RelateObjectsModule.child_objects_binding.require_parameter_name(),
        RelateObjectsModule.other_parent_objects_binding.require_parameter_name(),
    }
    assert setting_values(
        blocks[0], RelateObjectsModule.other_parent_distances_setting
    ) == ("Yes",)
    assert setting_values(
        blocks[0], RelateObjectsModule.other_parent_objects_setting
    ) == (grandparents.name,)
    (numbered_blocks,), _next_module_num = (
        RelateObjectsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=3,
        )
    )
    contract = RelateObjectsModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    RelateObjectsModule.validate_callable_artifact_abi(
        RelateObjectsModule.require_callable(),
        contract,
    )
    assert tuple(
        spec.name
        for spec in contract.artifact_inputs.of_artifact_type(ObjectLabelsArtifactType)
    ) == (parents.name, children.name, grandparents.name)


def _track_module(method: str, *, retain_image: str = "No") -> ModuleBlock:
    return ModuleBlock(
        name="TrackObjects",
        module_num=37,
        setting_records=[
            ModuleSetting("Select the objects to track", "Cells"),
            ModuleSetting("Choose a tracking method", method),
            ModuleSetting("Save color-coded image?", retain_image),
        ],
    )


@pytest.mark.parametrize("method", ("Overlap", "Distance"))
def test_track_objects_contract_declares_parent_self_relationship(
    method: str,
) -> None:
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    contract = TrackObjectsModule.callable_contract(
        module=_track_module(method),
        invocation_key=_invocation_key("track_objects"),
        step_context=_context(cells),
    )

    (relationship, declaration), = contract.artifact_outputs.relation_refs(
        ObjectRelationshipDeclaration
    )
    cells_input_ref = cells.for_plan_type(ArtifactInputPlan).ref()
    assert (
        declaration.relationship_type,
        declaration.source,
        declaration.target,
    ) == ("Parent", cells_input_ref, cells_input_ref)
    assert declaration.source_runtime_slice_offset == -1
    assert declaration.target_runtime_slice_offset == 0
    assert declaration.producer_module_number == 37
    (measurement,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert relationship.ref() not in {
        relation.source for relation in measurement.relations
    }
    TrackObjectsModule.validate_callable_artifact_abi(
        TrackObjectsModule.require_callable(),
        contract,
    )


@pytest.mark.parametrize("method", ("Measurements", "LAP"))
def test_track_objects_unsupported_methods_fail_before_contract_emission(
    method: str,
) -> None:
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    with pytest.raises(NotImplementedError, match="not supported"):
        TrackObjectsModule.callable_contract(
            module=_track_module(method),
            invocation_key=_invocation_key("track_objects"),
            step_context=_context(cells),
        )


def _timepoint_labels(labels: np.ndarray) -> ObjectLabelPayload:
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=labels.shape[0],
    )
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=PresentObjectLabelIdsDomainDeclaration(
            scope=ObjectLabelDomainScope.PLANE,
            plane_projection=projection,
        ).declared_domain(None, labels),
    )


@pytest.mark.parametrize(
    "method",
    (TrackingMethod.OVERLAP, TrackingMethod.DISTANCE),
)
def test_track_objects_runtime_payload_uses_previous_and_current_object_endpoints(
    method: TrackingMethod,
) -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 2:4] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels),
        tracking_method=method,
        pixel_radius=5,
    )
    assert isinstance(result, TrackObjectsResult)
    output, relationship, _rows = result.as_runtime_tuple()

    np.testing.assert_array_equal(output, image)
    assert relationship.source_ids == (1,)
    assert relationship.target_ids == (1,)
    assert relationship.slice_indices == (1,)
    assert relationship.slice_count == 2


def test_track_objects_overlap_relationship_keeps_all_merged_parents() -> None:
    labels = np.zeros((2, 6, 8), dtype=np.int32)
    labels[0, 1:4, 1:3] = 1
    labels[0, 1:4, 4:6] = 2
    labels[1, 1:4, 1:6] = 1

    result = unwrap(track_objects)(
        np.zeros(labels.shape, dtype=np.float32),
        labels=_timepoint_labels(labels),
        tracking_method=TrackingMethod.OVERLAP,
    )
    _output, relationship, _rows = result.as_runtime_tuple()

    assert relationship.source_ids == (1, 2)
    assert relationship.target_ids == (1, 1)
    assert relationship.slice_indices == (1, 1)


def test_public_track_objects_reconstructs_exact_relationship_without_sidecar() -> None:
    distractor = ArtifactSpec.output("Distractor", ObjectLabelsArtifactType)
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    tracked_parameter = (
        TrackObjectsModule.tracked_objects_binding.require_parameter_name()
    )
    step = FunctionStep(
        func=(
            track_objects,
            {
                tracked_parameter: cells.name,
                "tracking_method": "distance",
                "save_color_coded_image": False,
            },
        )
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    context = _context(distractor, cells)

    blocks, consumed = TrackObjectsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    assert consumed == (tracked_parameter,)
    (numbered_blocks,), _next_module_num = (
        TrackObjectsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=3,
        )
    )
    contract = TrackObjectsModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    (relationship, declaration), = contract.artifact_outputs.relation_refs(
        ObjectRelationshipDeclaration
    )

    assert setting_values(blocks[0], TrackObjectsModule.tracked_objects_setting) == (
        cells.name,
    )
    assert declaration.source.name == cells.name
    assert declaration.target.name == cells.name
    assert declaration.relationship_type == "Parent"
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == ()
    TrackObjectsModule.validate_callable_artifact_abi(track_objects, contract)


def test_relate_objects_requires_explicit_other_parent_distance_setting() -> None:
    module = ModuleBlock(
        name="RelateObjects",
        module_num=3,
        setting_records=[
            ModuleSetting("Calculate child-parent distances?", "Both"),
            ModuleSetting("Parent name", "Grandparents"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Calculate distances to other parents",
    ):
        RelateObjectsModule.other_parent_distances_enabled(module)


def test_relate_objects_requires_parent_rows_when_other_distances_are_enabled() -> None:
    parents = ArtifactSpec.output("Parents", ObjectLabelsArtifactType)
    children = ArtifactSpec.output("Children", ObjectLabelsArtifactType)
    module = ModuleBlock(
        name="RelateObjects",
        module_num=3,
        setting_records=[
            ModuleSetting("Select the parent objects", parents.name),
            ModuleSetting("Select the child objects", children.name),
            ModuleSetting("Calculate child-parent distances?", "Both"),
            ModuleSetting("Calculate distances to other parents?", "Yes"),
        ],
    )

    with pytest.raises(ValueError, match="declares no 'Parent name' setting row"):
        RelateObjectsModule.other_parent_inputs(
            module,
            ArtifactSpecCollection((parents, children)),
        )


def test_parsed_cppipe_track_objects_retained_image_matches_public_abi(
    tmp_path,
) -> None:
    cppipe_path = tmp_path / "track-objects.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
TrackObjects:[module_num:3|enabled:True]
    Choose a tracking method:Overlap
    Select the objects to track:Cells
    Save color-coded image?:Yes
    Name the output image:TrackedCells
""",
        encoding="utf-8",
    )
    (module,) = CPPipeParser(cppipe_path).parse()
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    invocation = next(normalize_function_pattern(track_objects).iter_items())
    context = _context(cells)

    contract = TrackObjectsModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=context,
    )

    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == (
        "TrackedCells",
    )
    TrackObjectsModule.validate_callable_artifact_abi(track_objects, contract)
