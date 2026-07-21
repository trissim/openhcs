from __future__ import annotations

import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import setting_values
from openhcs.processing.backends.cellprofiler.color import (
    GrayToColorModule,
    InvertForPrintingModule,
    InvertInputMode,
    OutputMode,
    gray_to_color,
    invert_for_printing_grayscale,
)
from openhcs.processing.backends.cellprofiler.grid import (
    IdentifyObjectsInGridModule,
    ShapeChoice,
    identify_objects_in_grid,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    CorrectIlluminationApplyModule,
    CorrectIlluminationCalculateModule,
    correct_illumination_apply,
    correct_illumination_calculate,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    ResizeDimensionSpecification,
    ResizeMethod,
    ResizeModule,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    DistanceMethod,
    MeasureObjectNeighborsModule,
    measure_object_neighbors,
)


def _reconstruct(
    module_type,
    function,
    kwargs,
    *,
    artifacts: tuple[ArtifactSpec, ...],
    step_index: int = 0,
):
    invocation = next(normalize_function_pattern((function, kwargs)).iter_items())
    available = ArtifactSpecCollection(artifacts)
    context = ArtifactDeclarationStepContext(
        step_name=str(module_type.module_name),
        step_index=step_index,
        available_artifacts=available,
        available_artifact_producers=artifact_producers_for_outputs(
            tuple(
                artifact
                for artifact in artifacts
                if artifact.plan_type is ArtifactOutputPlan
            ),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_artifact_producer",
                    invocation.key.group_key,
                    0,
                ),
            ),
        ),
        main_flow_artifacts=ArtifactSpecCollection(
            artifact.for_plan_type(ArtifactInputPlan) for artifact in artifacts
        ),
    )
    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    contract, consumed = module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed,
        step_context=context,
    )
    return numbered_blocks, contract, consumed


def test_illumination_apply_derives_both_routine_inputs_and_output() -> None:
    blocks, contract, consumed = _reconstruct(
        CorrectIlluminationApplyModule,
        correct_illumination_apply,
        {},
        artifacts=(
            ArtifactSpec.output("Raw", ImageArtifactType),
            ArtifactSpec.output("Illumination", ImageArtifactType),
        ),
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "Raw",
        "Illumination",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == (
        "CorrectIlluminationApply_1_image_1",
    )


@pytest.mark.parametrize(
    ("retain_average", "retain_dilated", "expected_output_count"),
    (
        (False, False, 1),
        (True, False, 2),
        (False, True, 2),
        (True, True, 3),
    ),
)
def test_illumination_calculate_behavior_controls_unnamed_retained_outputs(
    retain_average: bool,
    retain_dilated: bool,
    expected_output_count: int,
) -> None:
    _blocks, contract, consumed = _reconstruct(
        CorrectIlluminationCalculateModule,
        correct_illumination_calculate,
        {
            "retain_average": retain_average,
            "retain_dilated": retain_dilated,
        },
        artifacts=(ArtifactSpec.output("Raw", ImageArtifactType),),
    )

    assert consumed == ()
    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "Raw",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == tuple(
        f"CorrectIlluminationCalculate_1_image_{index}"
        for index in range(1, expected_output_count + 1)
    )


def test_gray_to_color_callable_channels_derive_omitted_image_identities() -> None:
    blocks, contract, consumed = _reconstruct(
        GrayToColorModule,
        gray_to_color,
        {
            "red_channel": -1,
            "green_channel": 0,
            "blue_channel": 1,
        },
        artifacts=(
            ArtifactSpec.output("DNA", ImageArtifactType),
            ArtifactSpec.output("Actin", ImageArtifactType),
        ),
    )

    assert consumed == ()
    assert setting_values(
        blocks[0], GrayToColorModule.rgb_channels[0].image_binding.setting_name
    ) == ("None",)
    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "DNA",
        "Actin",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == (
        "GrayToColor_1_image_1",
    )


def test_invert_for_printing_modes_derive_only_enabled_unnamed_ports() -> None:
    _blocks, contract, consumed = _reconstruct(
        InvertForPrintingModule,
        invert_for_printing_grayscale,
        {
            "input_mode": InvertInputMode.GRAYSCALE,
            "use_red_input": True,
            "use_green_input": False,
            "use_blue_input": True,
            "output_mode": OutputMode.GRAYSCALE,
            "output_red": False,
            "output_green": True,
            "output_blue": False,
        },
        artifacts=(
            ArtifactSpec.output("Red", ImageArtifactType),
            ArtifactSpec.output("Blue", ImageArtifactType),
        ),
    )

    assert consumed == ()
    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "Red",
        "Blue",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == (
        "InvertForPrinting_1_image_1",
    )


def test_resize_dimensions_image_activation_uses_behavior_not_identity() -> None:
    def active(*, resize_method: ResizeMethod, specification):
        records = [
            ModuleSetting(ResizeModule.method_setting.canonical, resize_method.value),
        ]
        if specification is not None:
            records.append(
                ModuleSetting(
                    ResizeModule.dimension_specification_setting.canonical,
                    specification.value,
                )
            )
        module = ModuleBlock(
            name="Resize",
            module_num=1,
            setting_records=records,
        )
        return ResizeModule.active_artifact_bindings(module)

    assert ResizeModule.desired_dimensions_image_binding not in active(
        resize_method=ResizeMethod.BY_FACTOR,
        specification=ResizeDimensionSpecification.IMAGE,
    )
    assert ResizeModule.desired_dimensions_image_binding not in active(
        resize_method=ResizeMethod.TO_SIZE,
        specification=ResizeDimensionSpecification.MANUAL,
    )
    assert ResizeModule.desired_dimensions_image_binding in active(
        resize_method=ResizeMethod.TO_SIZE,
        specification=ResizeDimensionSpecification.IMAGE,
    )


def test_guided_grid_derives_required_guide_without_an_identity_kwarg() -> None:
    _blocks, contract, consumed = _reconstruct(
        IdentifyObjectsInGridModule,
        identify_objects_in_grid,
        {"shape_choice": ShapeChoice.NATURAL},
        artifacts=(
            ArtifactSpec.output("Grid", SpatialGridArtifactType),
            ArtifactSpec.output("Guides", ObjectLabelsArtifactType),
        ),
    )

    assert consumed == ()
    assert tuple(
        (spec.name, spec.artifact_type) for spec in contract.artifact_inputs
    ) == (
        ("Grid", SpatialGridArtifactType),
        ("Guides", ObjectLabelsArtifactType),
    )


def test_neighbors_derives_measured_and_neighbor_occurrences_from_one_identity() -> (
    None
):
    blocks, contract, consumed = _reconstruct(
        MeasureObjectNeighborsModule,
        measure_object_neighbors,
        {
            "distance_method": DistanceMethod.EXPAND,
            "neighbor_distance": 5,
        },
        artifacts=(ArtifactSpec.output("Cells", ObjectLabelsArtifactType),),
    )

    assert consumed == ()
    assert setting_values(
        blocks[0], MeasureObjectNeighborsModule.measured_objects_setting
    ) == ("Cells",)
    assert setting_values(
        blocks[0], MeasureObjectNeighborsModule.neighbor_objects_setting
    ) == ("Cells",)
    assert contract.artifact_inputs.names_of_artifact_type(
        ObjectLabelsArtifactType
    ) == (
        "Cells",
        "Cells",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == ()
