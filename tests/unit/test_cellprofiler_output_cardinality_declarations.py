from __future__ import annotations

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
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
    ColorToGrayModule,
    GrayToColorModule,
    UnmixColorsModule,
    color_to_gray,
    gray_to_color,
    unmix_colors,
)
from openhcs.core.pipeline.function_contracts import (
    ImagePayloadConsumption,
    image_payload_consumption_from_callable,
)
from openhcs.processing.backends.cellprofiler.image_quality import (
    MeasureImageQualityModule,
    measure_image_quality,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    DistanceMethod,
    MeasureObjectNeighborsModule,
    measure_object_neighbors,
)
from openhcs.processing.backends.cellprofiler.tracking import (
    TrackObjectsModule,
    track_objects,
)


def _reconstruct(
    module_type,
    function,
    kwargs,
    *,
    available,
    main_flow,
    step_index=0,
):
    invocation = next(normalize_function_pattern((function, kwargs)).iter_items())
    available_artifacts = ArtifactSpecCollection(available)
    main_flow_artifacts = ArtifactSpecCollection(main_flow)
    step_context = ArtifactDeclarationStepContext(
        step_name=str(module_type.module_name),
        step_index=step_index,
        available_artifact_producers=artifact_producers_for_outputs(
            tuple(
                spec
                for spec in available_artifacts
                if spec.plan_type is ArtifactOutputPlan
            ),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", invocation.key.group_key, 0),
            ),
        ),
        available_artifacts=available_artifacts,
        main_flow_artifacts=ArtifactSpecCollection(
            spec.for_plan_type(ArtifactInputPlan)
            for spec in main_flow_artifacts
        ),
    )
    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    (blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    contracts = tuple(
        module_type.callable_contract(
            module=block,
            invocation_key=invocation.key,
            step_context=step_context,
        )
        for block in blocks
    )
    return blocks, contracts, consumed


def test_color_to_gray_plain_split_declares_every_callable_output() -> None:
    source = ArtifactSpec.input("OrigColor", ImageArtifactType)

    blocks, contracts, consumed = _reconstruct(
        ColorToGrayModule,
        color_to_gray,
        {},
        available=(source,),
        main_flow=(source,),
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert setting_values(
        blocks[0],
        ColorToGrayModule.output_image_setting,
    ) == ("OrigRed", "OrigGreen", "OrigBlue")
    assert contracts[0].artifact_outputs.names_of_artifact_type(
        ImageArtifactType
    ) == ("OrigRed", "OrigGreen", "OrigBlue")
    assert ColorToGrayModule.main_flow_output_specs(
        contracts[0].main_flow_outputs.specs
    ) == contracts[0].canonical_return_output_specs.specs


def test_gray_to_color_plain_rgb_uses_three_current_image_inputs() -> None:
    images = tuple(
        ArtifactSpec.output(name, ImageArtifactType)
        for name in ("Red", "Green", "Blue")
    )

    blocks, contracts, consumed = _reconstruct(
        GrayToColorModule,
        gray_to_color,
        {},
        available=images,
        main_flow=images,
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert contracts[0].artifact_inputs.names_of_artifact_type(
        ImageArtifactType
    ) == ("Red", "Green", "Blue")
    assert (
        len(contracts[0].artifact_outputs.names_of_artifact_type(ImageArtifactType))
        == 1
    )
    output = contracts[0].artifact_outputs.of_artifact_type(ImageArtifactType)[0]
    first_input = (
        contracts[0].artifact_inputs.of_artifact_type(ImageArtifactType)[0]
    )
    assert output.relations == (GroupLineageSourceRelation(source=first_input.ref()),)
    assert (
        image_payload_consumption_from_callable(gray_to_color)
        is ImagePayloadConsumption.COMPOSED
    )


def test_gray_to_color_sparse_rgb_excludes_inactive_channel_role() -> None:
    images = tuple(
        ArtifactSpec.output(name, ImageArtifactType) for name in ("DNA", "Actin")
    )
    green_binding = GrayToColorModule.rgb_channels[1].image_binding
    blue_binding = GrayToColorModule.rgb_channels[2].image_binding

    blocks, contracts, consumed = _reconstruct(
        GrayToColorModule,
        gray_to_color,
        {
            "green_channel": 0,
            "blue_channel": 1,
            green_binding.require_parameter_name(): "Actin",
            blue_binding.require_parameter_name(): "DNA",
            GrayToColorModule.output_image_binding.require_parameter_name(): "Composite",
        },
        available=images,
        main_flow=images,
    )

    assert len(blocks) == 1
    assert contracts[0].artifact_inputs.names_of_artifact_type(
        ImageArtifactType
    ) == ("Actin", "DNA")
    assert consumed == (
        green_binding.require_parameter_name(),
        blue_binding.require_parameter_name(),
        GrayToColorModule.output_image_binding.require_parameter_name(),
    )


def test_unmix_colors_contract_preserves_every_declared_output_row() -> None:
    source = ArtifactSpec.input("OrigColor", ImageArtifactType)
    invocation = next(normalize_function_pattern(unmix_colors).iter_items())
    available = ArtifactSpecCollection((source,))
    step_context = ArtifactDeclarationStepContext(
        step_name="UnmixColors",
        step_index=0,
        available_artifacts=available,
        main_flow_artifacts=available,
    )
    module = ModuleBlock(
        name="UnmixColors",
        module_num=1,
        setting_records=[
            ModuleSetting("Select the input color image", "OrigColor"),
            ModuleSetting("Name the output image", "Hematoxylin"),
            ModuleSetting("Stain", "Hematoxylin"),
            ModuleSetting("Name the output image", "Eosin"),
            ModuleSetting("Stain", "Eosin"),
            ModuleSetting("Stain count", "2"),
        ],
    )

    contract = UnmixColorsModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=step_context,
    )

    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "OrigColor",
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == (
        "Hematoxylin",
        "Eosin",
    )


def test_neighbors_inherits_neighbor_identity_and_preserves_fixed_output_slots() -> (
    None
):
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    blocks, contracts, consumed = _reconstruct(
        MeasureObjectNeighborsModule,
        measure_object_neighbors,
        {
            "distance_method": DistanceMethod.EXPAND,
            "neighbor_distance": 5,
            "retain_percent_touching_image": True,
        },
        available=(objects,),
        main_flow=(objects,),
        step_index=3,
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert setting_values(
        blocks[0],
        MeasureObjectNeighborsModule.measured_objects_setting,
    ) == ("Nuclei",)
    assert setting_values(
        blocks[0],
        MeasureObjectNeighborsModule.neighbor_objects_setting,
    ) == ("Nuclei",)
    output_names = setting_values(
        blocks[0],
        MeasureObjectNeighborsModule.output_image_setting,
    )
    assert len(output_names) == 2
    assert contracts[0].artifact_outputs.names_of_artifact_type(
        ImageArtifactType
    ) == (output_names[1],)
    retained_image = (
        contracts[0]
        .artifact_outputs
        .require_by_name_and_artifact_type(
            output_names[1],
            ImageArtifactType,
        )
    )
    assert retained_image.relations == (
        SourceStackLineageSourceRelation(
            source=ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref()
        ),
    )
    assert (
        len(
            contracts[0]
            .artifact_outputs
            .names_of_artifact_type(MeasurementsArtifactType)
        )
        == 1
    )


def test_track_objects_plain_invocation_uses_cp_no_retained_image_default() -> None:
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    blocks, contracts, consumed = _reconstruct(
        TrackObjectsModule,
        track_objects,
        {},
        available=(objects,),
        main_flow=(objects,),
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert setting_values(blocks[0], TrackObjectsModule.tracked_objects_setting) == (
        "Nuclei",
    )
    assert setting_values(blocks[0], TrackObjectsModule.retain_image_setting) == ("No",)
    assert setting_values(blocks[0], TrackObjectsModule.output_image_setting) == ()
    assert (
        contracts[0].artifact_outputs.names_of_artifact_type(ImageArtifactType) == ()
    )


def test_track_objects_public_retained_image_kwargs_define_exact_output() -> None:
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    blocks, contracts, consumed = _reconstruct(
        TrackObjectsModule,
        track_objects,
        {
            "save_color_coded_image": True,
            "name_the_output_image": "TrackedNuclei",
        },
        available=(objects,),
        main_flow=(objects,),
    )

    assert consumed == ("name_the_output_image",)
    assert setting_values(blocks[0], TrackObjectsModule.retain_image_setting) == (
        "Yes",
    )
    assert setting_values(blocks[0], TrackObjectsModule.output_image_setting) == (
        "TrackedNuclei",
    )
    assert contracts[0].artifact_outputs.names_of_artifact_type(
        ImageArtifactType
    ) == ("TrackedNuclei",)
    TrackObjectsModule.validate_callable_artifact_abi(track_objects, contracts[0])


def test_track_objects_explicitly_disabled_retained_image_remains_public() -> None:
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    blocks, contracts, consumed = _reconstruct(
        TrackObjectsModule,
        track_objects,
        {"save_color_coded_image": False},
        available=(objects,),
        main_flow=(objects,),
    )

    assert consumed == ()
    assert setting_values(blocks[0], TrackObjectsModule.retain_image_setting) == ("No",)
    assert (
        contracts[0].artifact_outputs.names_of_artifact_type(ImageArtifactType) == ()
    )


def test_measure_image_quality_consumes_explicit_repeated_selection() -> None:
    images = (
        ArtifactSpec.output("DNA", ImageArtifactType),
        ArtifactSpec.output("RNA", ImageArtifactType),
    )

    blocks, contracts, consumed = _reconstruct(
        MeasureImageQualityModule,
        measure_image_quality,
        {"select_images_to_measure": ("RNA", "DNA")},
        available=images,
        main_flow=images,
    )

    assert consumed == ("select_images_to_measure",)
    assert setting_values(
        blocks[0],
        MeasureImageQualityModule.image_measurement_binding.setting_name,
    ) == ("RNA", "DNA")
    assert contracts[0].artifact_inputs.names_of_artifact_type(
        ImageArtifactType
    ) == ("RNA", "DNA")
