from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect

import numpy as np

from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import ArtifactProducer
from openhcs.core.runtime_image_values import image_payload_data, image_payload_metadata
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.color import (
    InvertForPrintingModule,
    invert_for_printing,
    invert_for_printing_grayscale,
    invert_for_printing_without_output,
)
from openhcs.processing.backends.cellprofiler.imagej_macro import (
    RunImagejMacroModule,
    run_imagej_macro,
)
from openhcs.runtime.fiji_macro_runtime import FijiMacroExecutionRequest
from openhcs.processing.backends.cellprofiler.skeleton import (
    MeasureObjectSkeletonModule,
    ObjectSkeletonMeasurement,
    ObjectSkeletonSliceMeasurement,
    measure_object_skeleton,
    measure_object_skeleton_with_branchpoint_image,
)


def _invocation(func: Callable, kwargs: Mapping[str, object] | None = None):
    pattern = func if kwargs is None else (func, dict(kwargs))
    return next(normalize_function_pattern(pattern).iter_items())


def _artifact_context(
    *,
    image_names: tuple[str, ...],
    object_names: tuple[str, ...] = (),
) -> ArtifactDeclarationStepContext:
    images = tuple(ArtifactSpec.output(name, ImageArtifactType) for name in image_names)
    main_flow_images = tuple(
        ArtifactSpec.input(name, ImageArtifactType) for name in image_names
    )
    objects = tuple(
        ArtifactSpec.output(name, ObjectLabelsArtifactType) for name in object_names
    )
    return ArtifactDeclarationStepContext(
        step_index=0,
        available_artifacts=ArtifactSpecCollection((*images, *objects)),
        available_artifact_producers=tuple(
            ArtifactProducer(
                spec=spec,
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        function_name="fixture_object_producer",
                        group_key=DEFAULT_GROUP_KEY,
                        position=position,
                    ),
                ),
            )
            for position, spec in enumerate(objects)
        ),
        main_flow_artifacts=ArtifactSpecCollection(main_flow_images),
    )


def _parsed_contract(
    module_type,
    records: tuple[tuple[str, str], ...],
    *,
    image_names: tuple[str, ...],
    object_names: tuple[str, ...] = (),
):
    module = ModuleBlock(
        name=str(module_type.module_name),
        module_num=1,
        setting_records=[ModuleSetting(name, value) for name, value in records],
    )
    context = _artifact_context(
        image_names=image_names,
        object_names=object_names,
    )
    primary = module_type.require_callable()
    contract = module_type.callable_contract(
        module=module,
        invocation_key=_invocation(primary).key,
        step_context=context,
    )
    resolved = module_type.resolve_function(
        module,
        contract=contract,
        source_bindings=context.source_bindings,
    )
    module_type.validate_callable_artifact_abi(resolved, contract)
    return module, contract, resolved


def _public_function_step_contract(
    module_type,
    func: Callable,
    *,
    kwargs: Mapping[str, object] | None = None,
    image_names: tuple[str, ...],
    object_names: tuple[str, ...] = (),
):
    pattern = func if kwargs is None else (func, dict(kwargs))
    step = FunctionStep(func=pattern, name=str(module_type.module_name))
    invocation = next(normalize_function_pattern(step.func).iter_items())
    context = _artifact_context(
        image_names=image_names,
        object_names=object_names,
    )
    blocks, consumed_names = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        module_type.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract, consumed = module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=context,
    )
    return contract, consumed


def _outputs(contract, artifact_type):
    return contract.artifact_outputs.for_artifact_type(artifact_type).specs


def _invert_records(
    *,
    output_mode: str,
    output_flags: tuple[bool, bool, bool],
    input_mode: str = "Color",
) -> tuple[tuple[str, str], ...]:
    return (
        ("Input image type", input_mode),
        ("Use a red image?", "Yes"),
        ("Select the red image", "RedInput"),
        ("Use a green image?", "No"),
        ("Select the green image", "UnusedGreenInput"),
        ("Use a blue image?", "Yes"),
        ("Select the blue image", "BlueInput"),
        ("Select the color image", "ColorInput"),
        ("Output image type", output_mode),
        (
            'Select "*Yes*" to produce a red image.',
            "Yes" if output_flags[0] else "No",
        ),
        ("Name the red image", "RedOutput"),
        (
            'Select "*Yes*" to produce a green image.',
            "Yes" if output_flags[1] else "No",
        ),
        ("Name the green image", "GreenOutput"),
        (
            'Select "*Yes*" to produce a blue image.',
            "Yes" if output_flags[2] else "No",
        ),
        ("Name the blue image", "BlueOutput"),
        ("Name the inverted color image", "ColorOutput"),
    )


def test_invert_for_printing_color_declares_one_exact_named_output() -> None:
    _module, contract, resolved = _parsed_contract(
        InvertForPrintingModule,
        _invert_records(output_mode="Color", output_flags=(False, False, False)),
        image_names=("ColorInput",),
    )

    (output,) = _outputs(contract, ImageArtifactType)
    assert output.name == "ColorOutput"
    assert output.relations == (
        GroupLineageSourceRelation(source=contract.artifact_inputs[0].ref()),
    )
    assert resolved is InvertForPrintingModule.require_callable()


def test_invert_for_printing_declares_every_grayscale_output_subset() -> None:
    ordered_names = ("RedOutput", "GreenOutput", "BlueOutput")
    for mask in range(8):
        flags = tuple(bool(mask & (1 << index)) for index in range(3))
        records = _invert_records(
            output_mode="Grayscale",
            output_flags=flags,
        )
        _module, contract, resolved = _parsed_contract(
            InvertForPrintingModule,
            records,
            image_names=("ColorInput",),
        )

        assert tuple(
            output.name for output in _outputs(contract, ImageArtifactType)
        ) == tuple(
            name for name, enabled in zip(ordered_names, flags, strict=True) if enabled
        )
        expected_function = (
            invert_for_printing_grayscale
            if any(flags)
            else invert_for_printing_without_output
        )
        assert resolved is expected_function


def test_invert_for_printing_grayscale_input_declares_only_enabled_sources() -> None:
    _module, contract, _resolved = _parsed_contract(
        InvertForPrintingModule,
        _invert_records(
            input_mode="Grayscale",
            output_mode="Color",
            output_flags=(True, True, True),
        ),
        image_names=("RedInput", "BlueInput"),
    )

    assert tuple(spec.name for spec in contract.artifact_inputs) == (
        "RedInput",
        "BlueInput",
    )


def test_invert_for_printing_runtime_returns_enabled_channels_in_rgb_order() -> None:
    image = np.empty((2, 3, 3), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.4
    image[..., 2] = 0.6
    raw = inspect.unwrap(invert_for_printing)
    expected = (
        np.full((2, 3), 0.24, dtype=np.float32),
        np.full((2, 3), 0.32, dtype=np.float32),
        np.full((2, 3), 0.48, dtype=np.float32),
    )

    for mask in range(8):
        flags = tuple(bool(mask & (1 << index)) for index in range(3))
        result = raw(
            image,
            output_mode="grayscale",
            output_red=flags[0],
            output_green=flags[1],
            output_blue=flags[2],
        )
        if not any(flags):
            assert result is None
            continue
        values = result.slices if isinstance(result, AlignedImageStack) else (result,)
        selected_expected = tuple(
            value for value, enabled in zip(expected, flags, strict=True) if enabled
        )
        assert len(values) == len(selected_expected)
        for value, expected_value in zip(values, selected_expected, strict=True):
            assert image_payload_metadata(value).source_channel_axis is None
            np.testing.assert_allclose(image_payload_data(value), expected_value)


def test_invert_for_printing_runtime_declares_color_channel_axis() -> None:
    image = np.empty((2, 3, 3), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.4
    image[..., 2] = 0.6

    output = inspect.unwrap(invert_for_printing)(image, output_mode="color")

    assert image_payload_data(output).shape == (2, 3, 3)
    assert image_payload_metadata(output).source_channel_axis == -1


def test_invert_for_printing_public_variants_reconstruct_without_sidecars() -> None:
    grayscale_contract, _ = _public_function_step_contract(
        InvertForPrintingModule,
        invert_for_printing_grayscale,
        kwargs={
            "output_red": False,
            "output_green": True,
            "output_blue": True,
            "green_output_name": "OnlyGreen",
            "blue_output_name": "OnlyBlue",
        },
        image_names=("ColorInput",),
    )
    empty_contract, _ = _public_function_step_contract(
        InvertForPrintingModule,
        invert_for_printing_without_output,
        image_names=("ColorInput",),
    )

    assert tuple(
        output.name for output in _outputs(grayscale_contract, ImageArtifactType)
    ) == ("OnlyGreen", "OnlyBlue")
    assert _outputs(empty_contract, ImageArtifactType) == ()


def _skeleton_records(retain: bool) -> tuple[tuple[str, str], ...]:
    return (
        ("Select the seed objects", "Seeds"),
        ("Select the skeletonized image", "Skeleton"),
        ("Retain the branchpoint image?", "Yes" if retain else "No"),
        ("Name the branchpoint image", "Branches"),
        ("Fill small holes?", "Yes"),
        ("Maximum hole size", "10"),
        ("Export the skeleton graph relationships?", "No"),
        ("Intensity image", "None"),
        ("File output directory", "Default Output Folder|"),
        ("Vertex file name", "vertices.csv"),
        ("Edge file name", "edges.csv"),
    )


def test_measure_object_skeleton_declares_branchpoint_image_conditionally() -> None:
    for retain in (False, True):
        _module, contract, resolved = _parsed_contract(
            MeasureObjectSkeletonModule,
            _skeleton_records(retain),
            image_names=("Skeleton",),
            object_names=("Seeds",),
        )
        image_outputs = _outputs(contract, ImageArtifactType)
        measurement_outputs = _outputs(contract, MeasurementsArtifactType)

        assert tuple(output.name for output in image_outputs) == (
            ("Branches",) if retain else ()
        )
        assert len(measurement_outputs) == 1
        assert resolved is (
            measure_object_skeleton_with_branchpoint_image
            if retain
            else measure_object_skeleton
        )
        if retain:
            assert image_outputs[0].relations == (
                SourceStackLineageSourceRelation(
                    source=contract.artifact_inputs[0].ref()
                ),
            )


def test_measure_object_skeleton_retained_runtime_image_is_exact_analysis_image() -> (
    None
):
    skeleton = np.zeros((9, 9), dtype=bool)
    skeleton[4, 4:8] = True
    skeleton[2:7, 6] = True
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[3:6, 3:5] = 1
    payload = ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels))
    raw = inspect.unwrap(measure_object_skeleton_with_branchpoint_image)

    branchpoint_image, rows = raw(
        skeleton,
        payload,
        branchpoint_image_name="NamedBranches",
    )
    expected = ObjectSkeletonSliceMeasurement(
        skeleton=skeleton,
        seed_labels=labels,
        slice_index=0,
        fill_small_holes=True,
        maximum_hole_size=10,
    ).analyze()

    np.testing.assert_array_equal(
        image_payload_data(branchpoint_image),
        expected.branchpoint_image,
    )
    assert rows.row_type is ObjectSkeletonMeasurement
    assert rows.row_mappings() == tuple(
        {
            field: getattr(measurement, field)
            for field in measurement.__dataclass_fields__
        }
        for measurement in expected.measurements
    )


def test_measure_object_skeleton_public_variants_reconstruct_retain_condition() -> None:
    plain_contract, _ = _public_function_step_contract(
        MeasureObjectSkeletonModule,
        measure_object_skeleton,
        image_names=("Skeleton",),
        object_names=("Seeds",),
    )
    retained_contract, _ = _public_function_step_contract(
        MeasureObjectSkeletonModule,
        measure_object_skeleton_with_branchpoint_image,
        kwargs={"branchpoint_image_name": "PublicBranches"},
        image_names=("Skeleton",),
        object_names=("Seeds",),
    )

    assert _outputs(plain_contract, ImageArtifactType) == ()
    assert tuple(
        output.name for output in _outputs(retained_contract, ImageArtifactType)
    ) == ("PublicBranches",)


def _run_imagej_records() -> tuple[tuple[str, str], ...]:
    return (
        ("Hidden", "2"),
        ("Hidden", "3"),
        ("Hidden", "1"),
        (
            "What variable in your macro defines the folder ImageJ should use?",
            "Directory",
        ),
        ("Select an image to send to your macro", "FirstInput"),
        ("What should this image temporarily saved as?", "first-input.tif"),
        ("Select an image to send to your macro", "SecondInput"),
        ("What should this image temporarily saved as?", "second-input.tif"),
        ("What is the image filename CellProfiler should load?", "third.tif"),
        ("What should CellProfiler call the loaded image?", "ThirdOutput"),
        ("What is the image filename CellProfiler should load?", "first.tif"),
        ("What should CellProfiler call the loaded image?", "FirstOutput"),
        ("What is the image filename CellProfiler should load?", "second.tif"),
        ("What should CellProfiler call the loaded image?", "SecondOutput"),
        ("What variable name is your macro expecting?", "Threshold"),
        ("What value should this variable have?", "0.25"),
    )


def test_run_imagej_macro_output_groups_own_names_order_and_relations() -> None:
    module, contract, _resolved = _parsed_contract(
        RunImagejMacroModule,
        _run_imagej_records(),
        image_names=("FirstInput", "SecondInput"),
    )
    bound = RunImagejMacroModule.bind_settings(module, binder=SettingsBinder())
    outputs = _outputs(contract, ImageArtifactType)

    assert tuple(output.name for output in outputs) == (
        "ThirdOutput",
        "FirstOutput",
        "SecondOutput",
    )
    assert bound.kwargs["output_filenames"] == (
        "third.tif",
        "first.tif",
        "second.tif",
    )
    assert bound.kwargs["output_image_names"] == (
        "ThirdOutput",
        "FirstOutput",
        "SecondOutput",
    )
    assert all(
        output.relations
        == (
            GroupLineageSourceRelation(source=contract.artifact_inputs[0].ref()),
        )
        for output in outputs
    )


def test_run_imagej_macro_runtime_returns_declared_group_order(
    monkeypatch,
) -> None:
    output_filenames = ("third.tif", "first.tif", "second.tif")
    output_values = (3.0, 1.0, 2.0)

    def fake_send(self, _config, *, timeout=300.0):
        assert timeout == 300.0
        assert self.output_filenames == output_filenames
        return tuple(
            np.full((5, 6), value, dtype=np.float32) for value in output_values
        )

    monkeypatch.setattr(FijiMacroExecutionRequest, "send", fake_send)
    raw = inspect.unwrap(run_imagej_macro)
    result = raw(
        np.stack(
            (
                np.zeros((5, 6), dtype=np.uint8),
                np.ones((5, 6), dtype=np.uint8),
            )
        ),
        input_filenames=("first-input.tif", "second-input.tif"),
        output_filenames=output_filenames,
        output_image_names=("ThirdOutput", "FirstOutput", "SecondOutput"),
    )

    assert isinstance(result, AlignedImageStack)
    assert tuple(context.output_key for context in result.slice_contexts) == (
        "ThirdOutput",
        "FirstOutput",
        "SecondOutput",
    )
    assert tuple(
        float(np.mean(image_payload_data(value))) for value in result.slices
    ) == (
        3.0,
        1.0,
        2.0,
    )


def test_run_imagej_macro_public_function_step_reconstructs_named_groups() -> None:
    contract, _ = _public_function_step_contract(
        RunImagejMacroModule,
        run_imagej_macro,
        kwargs={
            "input_filenames": ("first-input.tif", "second-input.tif"),
            "output_filenames": ("one.tif", "two.tif"),
            "output_image_names": ("OneOutput", "TwoOutput"),
        },
        image_names=("FirstInput", "SecondInput"),
    )

    assert tuple(output.name for output in _outputs(contract, ImageArtifactType)) == (
        "OneOutput",
        "TwoOutput",
    )
