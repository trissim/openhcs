from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
    SpatialGridArtifactType,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import setting_name_matches
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.alignment import AlignModule, align
from openhcs.processing.backends.cellprofiler.grid import (
    DefineGridManualModule,
    define_grid_automatic,
    define_grid_manual,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    MaskImageModule,
    ResizeMethod,
    ResizeModule,
    TileModule,
    mask_image,
    resize,
    resize_volumetric,
    tile,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathModule,
    ImageMathOperation,
    image_math,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    ResizeObjectsModule,
    resize_objects,
    resize_objects_3d,
)
from openhcs.processing.backends.cellprofiler.object_images import (
    ConvertObjectsToImageModule,
    convert_objects_to_image,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    CorrectIlluminationApplyModule,
    CorrectIlluminationCalculateModule,
    correct_illumination_apply,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedMethod,
    WatershedModule,
    watershed_library,
)


def _invocation(func: Callable, kwargs: Mapping[str, object] | None = None):
    pattern = func if kwargs is None else (func, dict(kwargs))
    return next(normalize_function_pattern(pattern).iter_items())


def _artifact_state(
    *,
    image_names: tuple[str, ...] = (),
    object_names: tuple[str, ...] = (),
):
    image_specs = tuple(
        ArtifactSpec.output(name, ImageArtifactType) for name in image_names
    )
    object_specs = tuple(
        ArtifactSpec.output(name, ObjectLabelsArtifactType) for name in object_names
    )
    return (
        ArtifactSpecCollection((*image_specs, *object_specs)),
        ArtifactSpecCollection(
            spec.for_plan_type(ArtifactInputPlan) for spec in image_specs
        ),
    )


def _artifact_context(
    available: ArtifactSpecCollection,
    main_flow: ArtifactSpecCollection,
    *,
    invocation_key: FunctionInvocationKey,
) -> ArtifactDeclarationStepContext:
    return ArtifactDeclarationStepContext(
        step_index=0,
        available_artifact_producers=artifact_producers_for_outputs(
            tuple(spec for spec in available if spec.plan_type is ArtifactOutputPlan),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_producer",
                    invocation_key.group_key,
                    0,
                ),
            ),
        ),
        available_artifacts=available,
        main_flow_artifacts=main_flow,
    )


def _compile_declaration(
    module_type,
    func: Callable,
    *,
    kwargs: Mapping[str, object] | None = None,
    image_names: tuple[str, ...] = (),
    object_names: tuple[str, ...] = (),
):
    invocation = _invocation(func, kwargs)
    available, main_flow = _artifact_state(
        image_names=image_names,
        object_names=object_names,
    )
    context = _artifact_context(
        available,
        main_flow,
        invocation_key=invocation.key,
    )
    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    contracts = tuple(
        module_type.callable_contract(
            module=block,
            invocation_key=invocation.key,
            step_context=context,
        )
        for block in blocks
    )
    return blocks, contracts, consumed


def _record_values(block: ModuleBlock, setting_name) -> tuple[str, ...]:
    return tuple(
        setting.value
        for setting in block.iter_settings()
        if setting_name_matches(setting.name, setting_name)
    )


def _contract_input_names(contract) -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.name for spec in contract.artifact_inputs))


def test_align_reconstructs_fixed_and_repeated_image_roles() -> None:
    blocks, contracts, consumed = _compile_declaration(
        AlignModule,
        align,
        image_names=("First", "Second", "Additional"),
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert _record_values(blocks[0], AlignModule.first_input_setting) == ("First",)
    assert _record_values(blocks[0], AlignModule.second_input_setting) == ("Second",)
    assert _record_values(blocks[0], AlignModule.additional_input_setting) == (
        "Additional",
    )
    assert _contract_input_names(contracts[0]) == ("First", "Second", "Additional")
    outputs = contracts[0].artifact_outputs.of_artifact_type(ImageArtifactType)
    assert len(outputs) == 3
    assert tuple(
        relation.source.name
        for output in outputs
        for relation in output.relations
        if isinstance(relation, SourceStackLineageSourceRelation)
    ) == ("First", "Second", "Additional")


def test_align_explicit_repeated_role_does_not_consume_ambient_image() -> None:
    input_names = ("First", "Second", "Additional")
    output_names = ("AlignedFirst", "AlignedSecond", "AlignedAdditional")
    blocks, contracts, _consumed = _compile_declaration(
        AlignModule,
        align,
        kwargs={
            **{
                binding.require_parameter_name(): name
                for binding, name in zip(
                    AlignModule.declared_artifact_bindings(
                        plan_type=ArtifactInputPlan,
                        artifact_type=ImageArtifactType,
                    ),
                    input_names,
                    strict=True,
                )
            },
            **{
                binding.require_parameter_name(): name
                for binding, name in zip(
                    AlignModule.declared_artifact_bindings(
                        plan_type=ArtifactOutputPlan,
                        artifact_type=ImageArtifactType,
                    ),
                    output_names,
                    strict=True,
                )
            },
        },
        image_names=(*input_names, "Ambient"),
    )

    assert len(blocks) == 1
    assert _record_values(blocks[0], AlignModule.additional_input_setting) == (
        "Additional",
    )
    assert _contract_input_names(contracts[0]) == input_names
    assert (
        contracts[0].artifact_outputs.names_of_artifact_type(ImageArtifactType)
        == output_names
    )


def test_align_measurements_observe_every_exact_image_output_name() -> None:
    input_names = ("Plate", "Red", "Combined")
    output_names = ("AlignedPlate", "AlignedRed", "AlignedCombined")
    _blocks, contracts, _consumed = _compile_declaration(
        AlignModule,
        align,
        kwargs={
            **{
                binding.require_parameter_name(): name
                for binding, name in zip(
                    AlignModule.declared_artifact_bindings(
                        plan_type=ArtifactInputPlan,
                        artifact_type=ImageArtifactType,
                    ),
                    input_names,
                    strict=True,
                )
            },
            **{
                binding.require_parameter_name(): name
                for binding, name in zip(
                    AlignModule.declared_artifact_bindings(
                        plan_type=ArtifactOutputPlan,
                        artifact_type=ImageArtifactType,
                    ),
                    output_names,
                    strict=True,
                )
            },
        },
        image_names=input_names,
    )

    (measurements,) = contracts[0].artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )

    assert tuple(
        relation.source.name
        for relation in measurements.relations
        if isinstance(relation, ImageMeasurementSubjectRelation)
    ) == output_names


def test_define_grid_callable_variant_owns_object_topology() -> None:
    manual_blocks, manual_contracts, _ = _compile_declaration(
        DefineGridManualModule,
        define_grid_manual,
        image_names=("Display",),
        object_names=("Guides",),
    )
    automatic_blocks, automatic_contracts, _ = _compile_declaration(
        DefineGridManualModule,
        define_grid_automatic,
        image_names=("Display",),
        object_names=("Guides",),
    )

    assert _contract_input_names(manual_contracts[0]) == ("Display",)
    assert _contract_input_names(automatic_contracts[0]) == ("Display", "Guides")
    assert (
        _record_values(
            manual_blocks[0], DefineGridManualModule.previous_objects_setting
        )
        == ()
    )
    assert _record_values(
        automatic_blocks[0], DefineGridManualModule.previous_objects_setting
    ) == ("Guides",)
    (manual_grid,) = manual_contracts[0].artifact_outputs.of_artifact_type(
        SpatialGridArtifactType
    )
    (automatic_grid,) = automatic_contracts[0].artifact_outputs.of_artifact_type(
        SpatialGridArtifactType
    )
    assert tuple(source.name for source in manual_grid.group_scope_sources()) == (
        "Display",
    )
    assert tuple(source.name for source in automatic_grid.group_scope_sources()) == (
        "Guides",
    )


def test_define_grid_public_object_identity_disambiguates_automatic_source() -> None:
    kwargs = {
        DefineGridManualModule.display_grid_image_binding.require_parameter_name(): "Display",
        DefineGridManualModule.previous_objects_binding.require_parameter_name(): "Guides",
        DefineGridManualModule.grid_output_binding.require_parameter_name(): "Grid",
    }
    blocks, contracts, consumed = _compile_declaration(
        DefineGridManualModule,
        define_grid_automatic,
        kwargs=kwargs,
        image_names=("Display",),
        object_names=("Distractor", "Guides"),
    )

    assert set(consumed) == set(kwargs)
    assert _record_values(
        blocks[0], DefineGridManualModule.previous_objects_setting
    ) == ("Guides",)
    assert _contract_input_names(contracts[0]) == ("Display", "Guides")
    (grid,) = contracts[0].artifact_outputs.of_artifact_type(SpatialGridArtifactType)
    assert tuple(source.name for source in grid.group_scope_sources()) == ("Guides",)


def test_mask_image_contract_excludes_inactive_mask_role() -> None:
    module = ModuleBlock(
        name="MaskImage",
        module_num=1,
        setting_records=[
            ModuleSetting("Select the input image", "Input"),
            ModuleSetting("Select image for mask", "StaleMask"),
            ModuleSetting("Select object for mask", "Objects"),
            ModuleSetting("Name the output image", "Masked"),
            ModuleSetting("Use objects or an image as a mask?", "objects"),
        ],
    )
    invocation = _invocation(
        mask_image,
        {MaskImageModule.mask_source_binding.require_parameter_name(): "objects"},
    )
    available, main_flow = _artifact_state(
        image_names=("Input", "StaleMask"),
        object_names=("Objects",),
    )
    contract = MaskImageModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=_artifact_context(
            available,
            main_flow,
            invocation_key=invocation.key,
        ),
    )

    assert _contract_input_names(contract) == ("Input", "Objects")
    output = contract.artifact_outputs.of_artifact_type(ImageArtifactType)[0]
    assert len(output.relations) == 1
    assert type(output.relations[0]) is SourceStackLineageSourceRelation
    assert output.relations[0].source.name == "Input"


def test_resize_reconstructs_only_the_primary_image_role() -> None:
    blocks, contracts, _ = _compile_declaration(
        ResizeModule,
        resize,
        image_names=("Input",),
    )

    assert len(blocks) == 1
    assert _record_values(blocks[0], ResizeModule.input_image_setting) == ("Input",)
    assert (
        _record_values(blocks[0], ResizeModule.desired_dimensions_image_setting) == ()
    )
    assert _contract_input_names(contracts[0]) == ("Input",)
    output = contracts[0].artifact_outputs.of_artifact_type(ImageArtifactType)[0]
    assert output.relations == (
        SourceStackLineageSourceRelation(source=contracts[0].artifact_inputs[0].ref()),
    )


@pytest.mark.parametrize(
    ("resize_method", "resizing_factor_z", "relation_type"),
    (
        (ResizeMethod.BY_FACTOR, "1.0", SourceStackLineageSourceRelation),
        (ResizeMethod.BY_FACTOR, "0.5", GroupLineageSourceRelation),
        (ResizeMethod.TO_SIZE, "1.0", GroupLineageSourceRelation),
    ),
)
def test_resize_volumetric_declares_lineage_from_z_cardinality_transform(
    resize_method: ResizeMethod,
    resizing_factor_z: str,
    relation_type: type[SourceStackLineageSourceRelation | GroupLineageSourceRelation],
) -> None:
    module = ModuleBlock(
        name="Resize",
        module_num=1,
        setting_records=[
            ModuleSetting("Select the input image", "Input"),
            ModuleSetting("Name the output image", "Resized"),
            ModuleSetting("Resizing method", resize_method.value),
            ModuleSetting("Z Resizing factor", resizing_factor_z),
            ModuleSetting("# of planes (z) in the final image", "17"),
        ],
    )
    available, main_flow = _artifact_state(image_names=("Input",))
    contract = ResizeModule.callable_contract(
        module=module,
        invocation_key=_invocation(resize_volumetric).key,
        step_context=_artifact_context(
            available,
            main_flow,
            invocation_key=_invocation(resize_volumetric).key,
        ),
    )

    output = contract.artifact_outputs.of_artifact_type(ImageArtifactType)[0]
    assert output.relations == (
        relation_type(source=contract.artifact_inputs[0].ref()),
    )


def test_object_derived_image_preserves_object_artifact_stack_lineage() -> None:
    _blocks, contracts, _ = _compile_declaration(
        ConvertObjectsToImageModule,
        convert_objects_to_image,
        object_names=("Nuclei",),
    )

    output = contracts[0].artifact_outputs.of_artifact_type(ImageArtifactType)[0]
    assert output.relations == (
        SourceStackLineageSourceRelation(
            source=ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref()
        ),
    )


def test_tile_reconstructs_repeated_images_and_binds_typed_settings() -> None:
    blocks, contracts, consumed = _compile_declaration(
        TileModule,
        tile,
        image_names=("Original", "Outline", "Tracked"),
    )
    parsed = ModuleBlock(
        name="Tile",
        module_num=1,
        setting_records=[
            ModuleSetting("Select an input image", "Original"),
            ModuleSetting("Name the output image", "Adjacent"),
            ModuleSetting("Tile assembly method", "Within cycles"),
            ModuleSetting("Final number of rows", "1"),
            ModuleSetting("Final number of columns", "12"),
            ModuleSetting("Image corner to begin tiling", "top left"),
            ModuleSetting("Direction to begin tiling", "row"),
            ModuleSetting("Use meander mode?", "No"),
            ModuleSetting("Automatically calculate number of rows?", "No"),
            ModuleSetting("Automatically calculate number of columns?", "Yes"),
            ModuleSetting("Select an additional image to tile", "Outline"),
            ModuleSetting("Select an additional image to tile", "Tracked"),
        ],
    )
    bound = TileModule.bind_settings(parsed, binder=SettingsBinder())

    assert consumed == ()
    assert _record_values(blocks[0], TileModule.input_image_setting) == ("Original",)
    assert _record_values(blocks[0], TileModule.additional_image_setting) == (
        "Outline",
        "Tracked",
    )
    assert _contract_input_names(contracts[0]) == (
        "Original",
        "Outline",
        "Tracked",
    )
    outputs = contracts[0].artifact_outputs.of_artifact_type(ImageArtifactType)
    assert len(outputs) == 1
    assert outputs[0].relations == (
        SourceStackLineageSourceRelation(
            source=ArtifactSpec.input("Original", ImageArtifactType).ref()
        ),
    )
    assert bound.kwargs["rows"] == 1
    assert bound.kwargs["columns"] == 12
    assert bound.kwargs["auto_columns"] is True
    assert bound.kwargs[
        TileModule.additional_image_binding.require_parameter_name()
    ] == ("Outline", "Tracked")


def test_image_math_contract_uses_only_active_operands() -> None:
    add_blocks, add_contracts, _ = _compile_declaration(
        ImageMathModule,
        image_math,
        kwargs={
            ImageMathModule.operation_binding.require_parameter_name(): (
                ImageMathOperation.ADD
            )
        },
        image_names=("First", "Second"),
    )
    invert_blocks, invert_contracts, _ = _compile_declaration(
        ImageMathModule,
        image_math,
        kwargs={
            ImageMathModule.operation_binding.require_parameter_name(): (
                ImageMathOperation.INVERT
            )
        },
        image_names=("First",),
    )

    assert _record_values(add_blocks[0], ImageMathModule.first_image_setting) == (
        "First",
    )
    assert _record_values(add_blocks[0], ImageMathModule.second_image_setting) == (
        "Second",
    )
    assert _contract_input_names(add_contracts[0]) == ("First", "Second")
    assert _record_values(invert_blocks[0], ImageMathModule.second_image_setting) == ()
    assert _contract_input_names(invert_contracts[0]) == ("First",)
    for contract in (add_contracts[0], invert_contracts[0]):
        outputs = contract.artifact_outputs.of_artifact_type(ImageArtifactType)
        assert len(outputs) == 1
        relations = tuple(
            relation
            for relation in outputs[0].relations
            if isinstance(relation, SourceStackLineageSourceRelation)
        )
        assert len(relations) == 1
        assert relations[0].source.name == "First"


def test_watershed_contract_uses_active_roles_and_preserves_reuse() -> None:
    blocks, contracts, _ = _compile_declaration(
        WatershedModule,
        watershed_library,
        kwargs={
            WatershedModule.watershed_method_binding.require_parameter_name(): (
                WatershedMethod.DISTANCE
            ),
            WatershedModule.mask_binding.require_parameter_name(): "Input",
        },
        image_names=("Input",),
    )

    assert _record_values(blocks[0], WatershedModule.segmentation_image_setting) == (
        "Input",
    )
    assert _record_values(blocks[0], WatershedModule.mask_setting) == ("Input",)
    available, main_flow = _artifact_state(image_names=("Input",))
    role_inputs = WatershedModule.artifact_contract_inputs(
        blocks[0],
        invocation_key=_invocation(watershed_library).key,
        step_context=_artifact_context(
            available,
            main_flow,
            invocation_key=_invocation(watershed_library).key,
        ),
    )
    assert tuple(spec.name for spec in role_inputs) == ("Input", "Input")
    assert tuple(spec.name for spec in contracts[0].artifact_inputs) == (
        "Input",
        "Input",
    )
    assert WatershedModule.primary_image_inputs(
        watershed_library,
        contracts[0].artifact_inputs,
    ) == (contracts[0].artifact_inputs[0],)
    assert _contract_input_names(contracts[0]) == ("Input",)
    object_output = contracts[0].artifact_outputs.of_artifact_type(
        ObjectLabelsArtifactType,
    )[0]
    assert tuple(relation.source.name for relation in object_output.relations) == (
        "Input",
    )


def test_illumination_apply_contract_preserves_reused_input_role_occurrences() -> None:
    module_type = CorrectIlluminationApplyModule
    _blocks, contracts, _consumed = _compile_declaration(
        module_type,
        correct_illumination_apply,
        kwargs={
            module_type.input_image_binding.require_parameter_name(): "Input",
            module_type.illumination_function_binding.require_parameter_name(): (
                "Input"
            ),
            module_type.output_image_binding.require_parameter_name(): "Corrected",
        },
        image_names=("Input",),
    )

    assert tuple(spec.name for spec in contracts[0].artifact_inputs) == (
        "Input",
        "Input",
    )
    assert _contract_input_names(contracts[0]) == ("Input",)


def test_correct_illumination_calculate_declares_only_retained_image_ports() -> None:
    module = ModuleBlock(
        name="CorrectIlluminationCalculate",
        module_num=1,
        setting_records=[
            ModuleSetting("Select the input image", "Input"),
            ModuleSetting("Name the output image", "Illumination"),
            ModuleSetting("Retain the averaged image?", "Yes"),
            ModuleSetting("Name the averaged image", "Average"),
            ModuleSetting("Retain the dilated image?", "No"),
            ModuleSetting("Name the dilated image", "UnusedDilated"),
            ModuleSetting(
                "Calculate function for each image individually, or based on all images?",
                "Each",
            ),
        ],
    )
    available, main_flow = _artifact_state(image_names=("Input",))
    invocation = _invocation(CorrectIlluminationCalculateModule.require_callable())
    contract = CorrectIlluminationCalculateModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=_artifact_context(
            available,
            main_flow,
            invocation_key=invocation.key,
        ),
    )

    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == (
        "Illumination",
        "Average",
    )
    assert not contract.artifact_outputs.of_artifact_type(MeasurementsArtifactType)
    assert tuple(
        relation.source.name
        for output in contract.artifact_outputs.of_artifact_type(ImageArtifactType)
        for relation in output.relations
        if isinstance(relation, SourceStackLineageSourceRelation)
    ) == ("Input", "Input")


def test_correct_illumination_calculate_all_scope_does_not_preserve_input_stack() -> (
    None
):
    module = ModuleBlock(
        name="CorrectIlluminationCalculate",
        module_num=1,
        setting_records=[
            ModuleSetting("Select the input image", "Input"),
            ModuleSetting("Name the output image", "Illumination"),
            ModuleSetting("Retain the averaged image?", "No"),
            ModuleSetting("Retain the dilated image?", "No"),
            ModuleSetting(
                "Calculate function for each image individually, or based on all images?",
                "All: First cycle",
            ),
        ],
    )
    available, main_flow = _artifact_state(image_names=("Input",))
    invocation = _invocation(CorrectIlluminationCalculateModule.require_callable())
    contract = CorrectIlluminationCalculateModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=_artifact_context(
            available,
            main_flow,
            invocation_key=invocation.key,
        ),
    )

    output = contract.artifact_outputs.of_artifact_type(ImageArtifactType)[0]
    assert output.relations == (
        GroupLineageSourceRelation(source=contract.artifact_inputs[0].ref()),
    )


def test_resize_objects_declares_exact_axis_transform_for_selected_variant() -> None:
    source = ArtifactSpec.output("InputObjects", ObjectLabelsArtifactType)
    available = ArtifactSpecCollection((source,))
    context = _artifact_context(
        available,
        ArtifactSpecCollection(()),
        invocation_key=_invocation(resize_objects).key,
    )

    for func, extra_settings, relation_type in (
        (resize_objects, (), SourceStackLineageSourceRelation),
        (
            resize_objects_3d,
            (
                ModuleSetting("Method", "Factor"),
                ModuleSetting("Z Factor", "0.25"),
            ),
            GroupLineageSourceRelation,
        ),
        (
            resize_objects_3d,
            (
                ModuleSetting("Method", "Factor"),
                ModuleSetting("Z Factor", "1.0"),
            ),
            SourceStackLineageSourceRelation,
        ),
    ):
        module = ModuleBlock(
            name="ResizeObjects",
            module_num=1,
            setting_records=[
                ModuleSetting("Select the input object", "InputObjects"),
                ModuleSetting("Name the output object", "ResizedObjects"),
                *extra_settings,
            ],
        )
        invocation = _invocation(func)
        contract = ResizeObjectsModule.callable_contract(
            module=module,
            invocation_key=invocation.key,
            step_context=context,
        )

        output = contract.artifact_outputs.of_artifact_type(ObjectLabelsArtifactType)[0]
        assert output.relations == (
            relation_type(source=contract.artifact_inputs[0].ref()),
        )
