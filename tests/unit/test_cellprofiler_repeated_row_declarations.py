from __future__ import annotations

import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    NormalizedFunctionItem,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.cellprofiler.area_occupied import (
    AreaOccupiedRow,
    MeasureImageAreaOccupiedBinaryModule,
    OperandChoice,
    measure_image_area_occupied,
)
from openhcs.processing.backends.cellprofiler.outlines import (
    OutlineSourceKind,
    OverlayOutlinesModule,
    overlay_outlines,
)
from openhcs.processing.backends.cellprofiler.worms import (
    OverlapStyle,
    StraightenWormsModule,
    UntangleWormsModule,
    straighten_worms,
    untangle_worms,
    untangle_worms_both,
    untangle_worms_with_overlap,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting


def _invocation(function, **kwargs: object) -> NormalizedFunctionItem:
    contract = CallableContract.from_callable(function)
    key = FunctionInvocationKey.from_contract(contract, "default", 0)
    return NormalizedFunctionItem(
        key=key,
        contract=contract,
        kwargs=tuple(kwargs.items()),
    )


def _outputs(*specs: ArtifactSpec) -> ArtifactSpecCollection:
    return ArtifactSpecCollection(specs)


@pytest.mark.parametrize(
    ("overlap_style", "function", "related_object_names"),
    (
        (
            OverlapStyle.WITH_OVERLAP,
            untangle_worms_with_overlap,
            ("OverlappingWorms",),
        ),
        (
            OverlapStyle.WITHOUT_OVERLAP,
            untangle_worms,
            ("NonOverlappingWorms",),
        ),
        (
            OverlapStyle.BOTH,
            untangle_worms_both,
            ("OverlappingWorms", "NonOverlappingWorms"),
        ),
    ),
)
def test_untangle_worms_measurements_declare_exact_object_relations(
    overlap_style: OverlapStyle,
    function,
    related_object_names: tuple[str, ...],
) -> None:
    invocation = _invocation(function, overlap_style=overlap_style)
    source = ArtifactSpec.output("BinaryWorms", ImageArtifactType)
    module = ModuleBlock(
        name="UntangleWorms",
        module_num=4,
        setting_records=[
            ModuleSetting("Select the input binary image", source.name),
            ModuleSetting(
                "Name the output overlapping worm objects", "OverlappingWorms"
            ),
            ModuleSetting(
                "Name the output non-overlapping worm objects", "NonOverlappingWorms"
            ),
            ModuleSetting("Overlap style", overlap_style.value),
        ],
    )
    contract = UntangleWormsModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=ArtifactDeclarationStepContext(
            step_name="UntangleWorms",
            step_index=3,
            available_artifact_producers=artifact_producers_for_outputs(
                (source,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        invocation.key.group_key,
                        0,
                    ),
                ),
            ),
            available_artifacts=_outputs(source),
        ),
    )
    outputs = contract.artifact_outputs
    (measurements,) = outputs.of_artifact_type(MeasurementsArtifactType)
    assert (
        tuple(spec.name for spec in outputs.of_artifact_type(ObjectLabelsArtifactType))
        == related_object_names
    )
    related_object_refs = tuple(
        outputs.require_by_name_and_artifact_type(
            object_name,
            ObjectLabelsArtifactType,
        ).ref()
        for object_name in related_object_names
    )

    assert (
        tuple(
            relation.source
            for relation in measurements.relations
            if type(relation) is ArtifactSpecRelation
            and relation.source.artifact_type is ObjectLabelsArtifactType
        )
        == related_object_refs
    )


@pytest.mark.parametrize(
    ("function", "overlap_style", "kwargs", "expected_image_names"),
    (
        (
            untangle_worms,
            OverlapStyle.WITHOUT_OVERLAP,
            {
                "retain_nonoverlapping_outline": True,
                "nonoverlapping_outline_name": "NonOverlapOutline",
            },
            ("NonOverlapOutline",),
        ),
        (
            untangle_worms_with_overlap,
            OverlapStyle.WITH_OVERLAP,
            {
                "retain_overlapping_outline": True,
                "overlapping_outline_name": "OverlapOutline",
            },
            ("OverlapOutline",),
        ),
        (
            untangle_worms_both,
            OverlapStyle.BOTH,
            {
                "retain_overlapping_outline": True,
                "retain_nonoverlapping_outline": True,
                "overlapping_outline_name": "OverlapOutline",
                "nonoverlapping_outline_name": "NonOverlapOutline",
            },
            ("OverlapOutline", "NonOverlapOutline"),
        ),
    ),
)
def test_untangle_worms_public_invocation_reconstructs_exact_outline_outputs(
    function,
    overlap_style: OverlapStyle,
    kwargs: dict[str, object],
    expected_image_names: tuple[str, ...],
) -> None:
    source = ArtifactSpec.output("BinaryWorms", ImageArtifactType)
    invocation = _invocation(
        function,
        overlap_style=overlap_style,
        **kwargs,
    )
    context = ArtifactDeclarationStepContext(
        step_name="UntangleWorms",
        step_index=3,
        available_artifact_producers=artifact_producers_for_outputs(
            (source,),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_producer",
                    invocation.key.group_key,
                    0,
                ),
            ),
        ),
        available_artifacts=_outputs(source),
        main_flow_artifacts=_outputs(source.for_plan_type(ArtifactInputPlan)),
    )
    blocks, consumed_names = UntangleWormsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        UntangleWormsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=4,
        )
    )
    contract, _consumed = UntangleWormsModule.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=context,
    )

    outputs = contract.artifact_outputs
    assert (
        tuple(spec.name for spec in outputs.of_artifact_type(ImageArtifactType))
        == expected_image_names
    )


def test_area_occupied_reconstructs_every_mixed_measurement_row() -> None:
    step = FunctionStep(
        func=(
            measure_image_area_occupied,
            {
                MeasureImageAreaOccupiedBinaryModule.operand_choices_binding.require_parameter_name(): (
                    OperandChoice.OBJECTS,
                    OperandChoice.BINARY_IMAGE,
                ),
                MeasureImageAreaOccupiedBinaryModule.objects_binding.require_parameter_name(): (
                    "Nuclei",
                ),
                MeasureImageAreaOccupiedBinaryModule.binary_image_binding.require_parameter_name(): (
                    "Mask",
                ),
            },
        ),
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    available = _outputs(
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
        ArtifactSpec.output("Mask", ImageArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="repeated-row-test",
        step_index=2,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", invocation.key.group_key, 0),
            ),
        ),
        available_artifacts=available,
    )

    blocks, consumed = (
        MeasureImageAreaOccupiedBinaryModule.module_blocks_for_invocation(
            invocation=invocation,
            step_context=context,
        )
    )

    assert consumed == (
        MeasureImageAreaOccupiedBinaryModule.objects_binding.require_parameter_name(),
        MeasureImageAreaOccupiedBinaryModule.binary_image_binding.require_parameter_name(),
    )
    assert len(blocks) == 1
    rows = MeasureImageAreaOccupiedBinaryModule.measurement_rows(blocks[0])
    assert all(isinstance(row, AreaOccupiedRow) for row in rows)
    assert tuple((row.operand, row.input_name) for row in rows) == (
        (OperandChoice.OBJECTS, "Nuclei"),
        (OperandChoice.BINARY_IMAGE, "Mask"),
    )

    (numbered_blocks,), _next_module_num = (
        MeasureImageAreaOccupiedBinaryModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = MeasureImageAreaOccupiedBinaryModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    assert contract.artifact_inputs.names() == (
        "Mask",
        "Nuclei",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == ()
    assert (
        len(contract.artifact_outputs.names_of_artifact_type(MeasurementsArtifactType))
        == 1
    )
    MeasureImageAreaOccupiedBinaryModule.validate_callable_artifact_abi(
        measure_image_area_occupied,
        contract,
    )


def test_area_occupied_omitted_subset_has_no_reconstruction_candidate() -> None:
    step = FunctionStep(
        func=(
            measure_image_area_occupied,
            {
                MeasureImageAreaOccupiedBinaryModule.operand_choices_binding.require_parameter_name(): (
                    OperandChoice.OBJECTS,
                    OperandChoice.OBJECTS,
                ),
            },
        ),
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    available = _outputs(
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
        ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
        ArtifactSpec.output("Cytoplasm", ObjectLabelsArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="repeated-row-subset-test",
        step_index=2,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", invocation.key.group_key, 0),
            ),
        ),
        available_artifacts=available,
    )

    blocks, consumed = (
        MeasureImageAreaOccupiedBinaryModule.module_blocks_for_invocation(
            invocation=invocation,
            step_context=context,
        )
    )

    assert blocks == ()
    assert consumed == ()


def test_overlay_outlines_reconstructs_all_typed_rows_in_order() -> None:
    invocation = _invocation(
        overlay_outlines,
        **{
            OverlayOutlinesModule.blank_image_binding.require_parameter_name(): False,
            OverlayOutlinesModule.source_kind_binding.require_parameter_name(): (
                OutlineSourceKind.IMAGE,
                OutlineSourceKind.OBJECTS,
            ),
            OverlayOutlinesModule.color_binding.require_parameter_name(): (
                "Red",
                "Green",
            ),
            OverlayOutlinesModule.base_image_binding.require_parameter_name(): "DNA",
            OverlayOutlinesModule.outline_image_binding.require_parameter_name(): (
                "Edges",
            ),
            OverlayOutlinesModule.objects_binding.require_parameter_name(): ("Nuclei",),
            OverlayOutlinesModule.output_image_binding.require_parameter_name(): "OutlinedDNA",
        },
    )
    available = _outputs(
        ArtifactSpec.output("DNA", ImageArtifactType),
        ArtifactSpec.output("Edges", ImageArtifactType),
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="repeated-row-test",
        step_index=4,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", invocation.key.group_key, 0),
            ),
        ),
        available_artifacts=available,
    )

    blocks, consumed = OverlayOutlinesModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )

    assert consumed == (
        OverlayOutlinesModule.base_image_binding.require_parameter_name(),
        OverlayOutlinesModule.outline_image_binding.require_parameter_name(),
        OverlayOutlinesModule.objects_binding.require_parameter_name(),
        OverlayOutlinesModule.output_image_binding.require_parameter_name(),
    )
    assert len(blocks) == 1
    rows = OverlayOutlinesModule.outline_rows(blocks[0])
    assert tuple((row.source_kind, row.input_name, row.color) for row in rows) == (
        (OutlineSourceKind.IMAGE, "Edges", "Red"),
        (OutlineSourceKind.OBJECTS, "Nuclei", "Green"),
    )

    (numbered_blocks,), _next_module_num = (
        OverlayOutlinesModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = OverlayOutlinesModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    assert contract.artifact_inputs.names() == (
        "DNA",
        "Edges",
        "Nuclei",
    )
    assert contract.artifact_outputs.names() == ("OutlinedDNA",)


def test_overlay_outlines_omitted_subset_has_no_reconstruction_candidate() -> None:
    invocation = _invocation(
        overlay_outlines,
        **{
            OverlayOutlinesModule.blank_image_binding.require_parameter_name(): False,
            OverlayOutlinesModule.source_kind_binding.require_parameter_name(): (
                OutlineSourceKind.OBJECTS,
                OutlineSourceKind.OBJECTS,
            ),
            OverlayOutlinesModule.color_binding.require_parameter_name(): (
                "Red",
                "Green",
            ),
            OverlayOutlinesModule.base_image_binding.require_parameter_name(): "DNA",
            OverlayOutlinesModule.output_image_binding.require_parameter_name(): "OutlinedDNA",
        },
    )
    available = _outputs(
        ArtifactSpec.output("DNA", ImageArtifactType),
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
        ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
        ArtifactSpec.output("Cytoplasm", ObjectLabelsArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="repeated-outline-subset-test",
        step_index=4,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", invocation.key.group_key, 0),
            ),
        ),
        available_artifacts=available,
    )

    blocks, consumed = OverlayOutlinesModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )

    assert blocks == ()
    assert consumed == (
        OverlayOutlinesModule.base_image_binding.require_parameter_name(),
        OverlayOutlinesModule.output_image_binding.require_parameter_name(),
    )


def test_straighten_worms_preserves_both_image_row_pairs() -> None:
    invocation = _invocation(
        straighten_worms,
        **{
            StraightenWormsModule.input_objects_binding.require_parameter_name(): "Worms",
            StraightenWormsModule.output_objects_binding.require_parameter_name(): "StraightenedWorms",
            StraightenWormsModule.input_image_binding.require_parameter_name(): (
                "Red",
                "Green",
            ),
            StraightenWormsModule.output_image_binding.require_parameter_name(): (
                "StraightRed",
                "StraightGreen",
            ),
        },
    )
    available = _outputs(
        ArtifactSpec.output("Worms", ObjectLabelsArtifactType),
        ArtifactSpec.output("Red", ImageArtifactType),
        ArtifactSpec.output("Green", ImageArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="repeated-row-test",
        step_index=6,
        available_artifacts=available,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_producer",
                    invocation.key.group_key,
                    0,
                ),
            ),
        ),
    )

    blocks, consumed = StraightenWormsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )

    assert consumed == (
        StraightenWormsModule.input_objects_binding.require_parameter_name(),
        StraightenWormsModule.output_objects_binding.require_parameter_name(),
        StraightenWormsModule.input_image_binding.require_parameter_name(),
        StraightenWormsModule.output_image_binding.require_parameter_name(),
    )
    assert len(blocks) == 1
    assert StraightenWormsModule.image_bindings(blocks[0]) == (
        StraightenWormsModule.ImageBinding("Red", "StraightRed"),
        StraightenWormsModule.ImageBinding("Green", "StraightGreen"),
    )

    (numbered_blocks,), _next_module_num = (
        StraightenWormsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = StraightenWormsModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    assert contract.artifact_inputs.names() == (
        "Red",
        "Green",
        "Worms",
    )
    declared_outputs = contract.artifact_outputs
    assert declared_outputs.names() == (
        "StraightRed",
        "StraightGreen",
        "repeated-row-test_1_measurements",
        "StraightenedWorms",
    )
    assert tuple(spec.artifact_type for spec in declared_outputs) == (
        ImageArtifactType,
        ImageArtifactType,
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    )
    assert contract.main_flow_outputs.names() == (
        "StraightRed",
        "StraightGreen",
        "StraightenedWorms",
    )
    declared_inputs = contract.artifact_inputs
    outputs = ArtifactSpecCollection(contract.artifact_outputs)
    assert outputs.require_by_name_and_artifact_type(
        "StraightRed", ImageArtifactType
    ).source_context_sources() == (
        declared_inputs.require_by_name_and_artifact_type(
            "Red", ImageArtifactType
        ).ref(),
    )
    assert outputs.require_by_name_and_artifact_type(
        "StraightGreen", ImageArtifactType
    ).source_context_sources() == (
        declared_inputs.require_by_name_and_artifact_type(
            "Green", ImageArtifactType
        ).ref(),
    )
    assert outputs.require_by_name_and_artifact_type(
        "StraightenedWorms", ObjectLabelsArtifactType
    ).source_context_sources() == (
        declared_inputs.require_by_name_and_artifact_type(
            "Worms", ObjectLabelsArtifactType
        ).ref(),
    )
    StraightenWormsModule.validate_callable_artifact_abi(straighten_worms, contract)


def test_straighten_worms_derives_one_output_for_each_repeated_input() -> None:
    invocation = _invocation(
        straighten_worms,
        **{
            StraightenWormsModule.input_objects_binding.require_parameter_name(): "Worms",
            StraightenWormsModule.output_objects_binding.require_parameter_name(): "StraightenedWorms",
            StraightenWormsModule.input_image_binding.require_parameter_name(): (
                "Red",
                "Green",
            ),
        },
    )
    available = _outputs(
        ArtifactSpec.output("Worms", ObjectLabelsArtifactType),
        ArtifactSpec.output("Red", ImageArtifactType),
        ArtifactSpec.output("Green", ImageArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="repeated-row-test",
        step_index=6,
        available_artifacts=available,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(FunctionInvocationKey("fixture_producer", "default", 0),),
        ),
    )

    blocks, consumed = StraightenWormsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )

    assert consumed == (
        StraightenWormsModule.input_objects_binding.require_parameter_name(),
        StraightenWormsModule.output_objects_binding.require_parameter_name(),
        StraightenWormsModule.input_image_binding.require_parameter_name(),
    )
    assert StraightenWormsModule.image_bindings(blocks[0]) == (
        StraightenWormsModule.ImageBinding("Red", "StraightenWorms_7_image_1"),
        StraightenWormsModule.ImageBinding("Green", "StraightenWorms_7_image_2"),
    )


def test_straighten_worms_selects_exact_related_measurements() -> None:
    object_key = FunctionInvocationKey("identify", "default", 0)
    measurement_key = FunctionInvocationKey("measure", "default", 0)
    objects = ArtifactSpec.output("Worms", ObjectLabelsArtifactType)
    other_objects = ArtifactSpec.output("OtherWorms", ObjectLabelsArtifactType)
    measurements = ArtifactSpec.output(
        "ObjectMeasurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(source=objects.ref()),),
    )
    unrelated = ArtifactSpec.output(
        "SameInvocationMeasurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(source=other_objects.ref()),),
    )
    context = ArtifactDeclarationStepContext(
        available_artifacts=_outputs(
            measurements,
            objects,
            other_objects,
            unrelated,
        ),
        available_artifact_producers=(
            *artifact_producers_for_outputs(
                (objects, unrelated),
                groups=(None,),
                invocation_keys=(object_key,),
            ),
            *artifact_producers_for_outputs(
                (measurements, other_objects),
                groups=(None,),
                invocation_keys=(measurement_key,),
            ),
        ),
    )

    selected = StraightenWormsModule.producer_measurement_input(
        objects.for_plan_type(ArtifactInputPlan),
        step_context=context,
    )

    assert selected == measurements.for_plan_type(ArtifactInputPlan)


def test_straighten_worms_ignores_unrelated_same_invocation_measurements() -> None:
    producer_key = FunctionInvocationKey("identify", "default", 0)
    objects = ArtifactSpec.output("Worms", ObjectLabelsArtifactType)
    other_objects = ArtifactSpec.output("OtherWorms", ObjectLabelsArtifactType)
    unrelated = ArtifactSpec.output(
        "ObjectMeasurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(source=other_objects.ref()),),
    )
    context = ArtifactDeclarationStepContext(
        available_artifacts=_outputs(objects, other_objects, unrelated),
        available_artifact_producers=artifact_producers_for_outputs(
            (objects, unrelated),
            groups=(None,),
            invocation_keys=(producer_key,),
        ),
    )

    assert (
        StraightenWormsModule.producer_measurement_input(
            objects.for_plan_type(ArtifactInputPlan),
            step_context=context,
        )
        is None
    )


def test_straighten_worms_rejects_ambiguous_exact_measurement_relations() -> None:
    objects = ArtifactSpec.output("Worms", ObjectLabelsArtifactType)
    measurements = tuple(
        ArtifactSpec.output(
            name,
            MeasurementsArtifactType,
            relations=(ArtifactSpecRelation(source=objects.ref()),),
        )
        for name in ("FirstMeasurements", "SecondMeasurements")
    )
    context = ArtifactDeclarationStepContext(
        available_artifacts=_outputs(objects, *measurements),
    )

    with pytest.raises(ValueError, match="multiple measurement artifacts related"):
        StraightenWormsModule.producer_measurement_input(
            objects.for_plan_type(ArtifactInputPlan),
            step_context=context,
        )
