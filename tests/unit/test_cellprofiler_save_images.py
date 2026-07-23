"""Focused contracts for executable CellProfiler SaveImages support."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MaterializationSourceIdentityRelation,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractPlan,
)
from openhcs.core.pipeline.function_contracts import (
    special_input_names_from_callable,
)
from openhcs.core.pipeline.artifact_planning import (
    artifact_producers_for_outputs,
    extract_artifact_declarations,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.steps.function_runtime import (
    FunctionOutputContextStrategy,
    project_declared_source_identity,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    PlaneRuntimeArtifactModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.save_images import (
    SaveImagesBitDepth,
    SaveImagesFileFormat,
    SaveImagesFilenameMethod,
    SaveImagesModule,
    save_images,
    save_images_with_measurements,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)
from openhcs.processing.materialization import (
    ImageFileOptions,
    MaterializationSpec,
    MaterializedFilenameIdentity,
    materialize,
)
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend


def _module(**settings: str) -> ModuleBlock:
    records = [ModuleSetting(name, value) for name, value in settings.items()]
    return ModuleBlock(
        name="SaveImages",
        module_num=9,
        setting_records=records,
    )


def _contract(
    module: ModuleBlock,
    image_name: str = "ImageToSave",
    *,
    source_names: tuple[str, ...] = (),
):
    invocation_key = FunctionInvocationKey("save_images", "default", 0)
    image = ArtifactSpec.output(image_name, ImageArtifactType)
    return SaveImagesModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_index=8,
            source_bindings=StepSourceBindingsConfig(
                enabled=bool(source_names),
                bindings=tuple(
                    NamedSourceBinding(
                        alias=name,
                        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    )
                    for name in source_names
                ),
            ),
            available_artifacts=ArtifactSpecCollection(
                (
                    *(
                        ArtifactSpec.input(name, ImageArtifactType)
                        for name in source_names
                    ),
                    image,
                )
            ),
            main_flow_artifacts=ArtifactSpecCollection(
                (image.for_plan_type(ArtifactInputPlan),)
            ),
            available_artifact_producers=artifact_producers_for_outputs(
                (image,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        invocation_key.group_key,
                        0,
                    ),
                ),
            ),
        ),
    )


def _single_name_module(
    *,
    file_format: str,
    bit_depth: str,
    single_name: str = "Saved",
    output_location: str = "Default Output Folder sub-folder|exports",
) -> ModuleBlock:
    return _module(
        **{
            "Select the type of image to save": "Image",
            "Select the image to save": "ImageToSave",
            "Select method for constructing file names": "Single name",
            "Enter single file name": single_name,
            "Number of digits": "4",
            "Append a suffix to the image file name?": "Yes",
            "Text to append to the image name": "_result",
            "Saved file format": file_format,
            "Output file location": output_location,
            "Image bit depth": bit_depth,
            "Overwrite existing files without warning?": "Yes",
            "When to save": "Every cycle",
            "Record the file and path information to the saved image?": "No",
            "Create subfolders in the output folder?": "No",
            "Base image folder": "Elsewhere...|",
            "How to save the series": "T (Time)",
        }
    )


def _image_file_options(contract) -> ImageFileOptions:
    materialization = contract.artifact_outputs[0].materialization
    assert isinstance(materialization, MaterializationSpec)
    assert len(materialization.outputs) == 1
    options = materialization.outputs[0]
    assert isinstance(options, ImageFileOptions)
    return options


def test_save_images_is_an_adapter_free_executable_axis_module() -> None:
    assert issubclass(SaveImagesModule, PlaneRuntimeArtifactModule)
    assert issubclass(SaveImagesModule, CellProfilerModule)
    image_inputs = SaveImagesModule.artifact_bindings_for(
        None,
        plan_type=ArtifactInputPlan,
        artifact_type=ImageArtifactType,
    )
    assert tuple(binding.require_parameter_name() for binding in image_inputs) == (
        "select_image_name_for_file_prefix",
        "select_the_image_to_save",
    )
    assert tuple(binding.runtime_parameter_name for binding in image_inputs) == (
        None,
        "image_to_save",
    )
    assert all(
        binding.require_artifact_plan_type() is ArtifactInputPlan
        for binding in image_inputs
    )
    image_outputs = SaveImagesModule.artifact_bindings_for(
        None,
        plan_type=ArtifactOutputPlan,
        artifact_type=ImageArtifactType,
    )
    assert tuple(binding.require_parameter_name() for binding in image_outputs) == (
        "materialized_image_artifact_name",
    )
    assert SaveImagesModule.emits_function_step()
    assert not SaveImagesModule.uses_cellprofiler_runtime_adapter()
    assert SaveImagesModule.main_flow_output_specs(()) == ()

    callable_contract = CallableContract.from_callable(save_images)
    assert callable_contract.processing_contract is ProcessingContract.PURE_3D
    assert callable_contract.runtime_adapter is None
    assert callable_contract.resolve_runtime_callable() is save_images
    assert special_input_names_from_callable(save_images) == ("image_to_save",)


def test_save_images_contract_consumes_runtime_image_and_declares_export_only() -> None:
    module = _single_name_module(
        file_format="png",
        bit_depth="8-bit integer",
    )

    contract = _contract(module)

    assert contract.artifact_inputs.names() == (
        "ImageToSave",
    )
    assert contract.artifact_outputs.names() == (
        "SaveImages_9_image_1",
    )
    output = contract.artifact_outputs[0]
    assert output.artifact_type is ImageArtifactType
    assert output.sidecar_role is ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY
    assert not output.participates_in_main_flow
    assert contract.preserves_input_main_flow()
    assert contract.canonical_return_output_specs.names() == ()
    assert contract.trailing_return_output_specs.names() == (
        "SaveImages_9_image_1",
    )
    assert output.relations == (
        SourceStackLineageSourceRelation(
            source=contract.artifact_inputs[0].ref()
        ),
    )
    assert isinstance(output.materialization, MaterializationSpec)


def test_materialized_image_copy_role_derives_from_generic_sidecar_authority() -> None:
    assert ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY.name_for("SavedImage") == (
        "SavedImage__materialized_image_copy"
    )


def test_save_images_selected_image_binding_is_runtime_special_identity() -> None:
    parameter_names = tuple(inspect.signature(save_images).parameters)
    assert "select_the_image_to_save" not in parameter_names
    assert "image_to_save" in parameter_names
    assert "filename_source_image" not in parameter_names
    assert "materialized_image_artifact_name" not in parameter_names

    module = _single_name_module(
        file_format="png",
        bit_depth="8-bit integer",
    )
    bound = SaveImagesModule.bind_settings(module, binder=SettingsBinder())

    assert bound.kwargs["select_the_image_to_save"] == "ImageToSave"
    assert "image_to_save" not in bound.kwargs
    assert bound.kwargs["file_format"] is SaveImagesFileFormat.PNG
    assert bound.kwargs["bit_depth"] is SaveImagesBitDepth.UINT8
    assert bound.kwargs["filename_method"] is SaveImagesFilenameMethod.SINGLE_NAME
    assert special_input_names_from_callable(save_images) == ("image_to_save",)


def test_save_images_public_contract_selects_upstream_runtime_image() -> None:
    source = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    selected = ArtifactSpec.output("RGBImage", ImageArtifactType)
    invocation = next(
        normalize_function_pattern(
            (
                save_images,
                {"select_image_name_for_file_prefix": "OrigBlue"},
            )
        ).iter_items()
    )
    step_context = ArtifactDeclarationStepContext(
        step_name="SaveImages",
        step_index=8,
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigBlue",
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                ),
            ),
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            (selected,),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_producer",
                    invocation.key.group_key,
                    0,
                ),
            ),
        ),
        available_artifacts=ArtifactSpecCollection((source, selected)),
        main_flow_artifacts=ArtifactSpecCollection(
            (selected.for_plan_type(ArtifactInputPlan),)
        ),
    )
    blocks, consumed_names = SaveImagesModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    (numbered_blocks,), _next_module_num = (
        SaveImagesModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=9,
        )
    )
    contract, consumed = SaveImagesModule.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=step_context,
    )

    assert contract.artifact_inputs.names() == (
        "OrigBlue",
        "RGBImage",
    )
    assert consumed == ("select_image_name_for_file_prefix",)


def test_grouped_save_images_materializations_keep_numbered_module_identity() -> None:
    images = (
        ArtifactSpec.output("IllumActin", ImageArtifactType),
        ArtifactSpec.output("IllumDAPI", ImageArtifactType),
    )
    pattern = normalize_function_pattern(
        {
            "2": [
                (
                    save_images_with_measurements,
                    {
                        "image_to_save": "IllumActin",
                        "saved_image_name": "IllumActin",
                        "filename_method": SaveImagesFilenameMethod.SINGLE_NAME,
                        "single_file_name": r"\g<folder>_IllumActin",
                        "file_format": SaveImagesFileFormat.NPY,
                    },
                )
            ],
            "1": [
                (
                    save_images_with_measurements,
                    {
                        "image_to_save": "IllumDAPI",
                        "saved_image_name": "IllumDAPI",
                        "filename_method": SaveImagesFilenameMethod.SINGLE_NAME,
                        "single_file_name": r"\g<folder>_IllumDAPI",
                        "file_format": SaveImagesFileFormat.NPY,
                    },
                )
            ],
        }
    )
    context = ArtifactDeclarationStepContext(
        step_name="SaveImages",
        step_index=1,
        available_artifacts=ArtifactSpecCollection(images),
        main_flow_artifacts=ArtifactSpecCollection(
            image.for_plan_type(ArtifactInputPlan) for image in images
        ),
        available_artifact_producers=tuple(
            producer
            for image, group_key in zip(images, ("2", "1"), strict=True)
            for producer in artifact_producers_for_outputs(
                (image,),
                groups=(group_key,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", group_key, 0),
                ),
            )
        ),
    )
    invocations = tuple(pattern.iter_items())
    blocks_and_consumed = tuple(
        SaveImagesModule.module_blocks_for_invocation(
            invocation=invocation,
            step_context=context,
        )
        for invocation in invocations
    )
    numbered_blocks, _next_module_num = (
        SaveImagesModule.number_step_invocation_blocks(
            tuple(blocks for blocks, _consumed in blocks_and_consumed),
            first_module_num=8,
        )
    )
    contracts = {}
    for invocation, blocks, (_raw_blocks, consumed_names) in zip(
        invocations,
        numbered_blocks,
        blocks_and_consumed,
        strict=True,
    ):
        contract, _consumed = SaveImagesModule.invocation_callable_contract(
            invocation=invocation,
            numbered_module_blocks=blocks,
            consumed_kwarg_names=consumed_names,
            step_context=context,
        )
        contracts[invocation.key] = contract

    graph = extract_artifact_declarations(
        pattern,
        declaration_provider=lambda invocation, _context: contracts[invocation.key],
        step_context=context,
    )

    image_outputs = tuple(
        spec
        for spec in graph.outputs.values()
        if spec.artifact_type is ImageArtifactType
    )
    assert tuple(spec.name for spec in image_outputs) == (
        "SaveImages_8_image_1",
        "SaveImages_9_image_1",
    )
    assert tuple(
        spec.materialization.outputs[0].relative_path_template
        for spec in image_outputs
    ) == (
        r"\g<folder>_IllumActin.npy",
        r"\g<folder>_IllumDAPI.npy",
    )


@pytest.mark.parametrize(
    ("format_literal", "depth_literal", "depth", "input_dtype", "expected_dtype"),
    (
        ("png", "8-bit integer", SaveImagesBitDepth.UINT8, np.float32, np.uint8),
        (
            "tiff",
            "16-bit integer",
            SaveImagesBitDepth.UINT16,
            np.float32,
            np.uint16,
        ),
        (
            "npy",
            "32-bit floating point",
            SaveImagesBitDepth.FLOAT32,
            np.float64,
            np.float32,
        ),
        ("tiff", "Raw", SaveImagesBitDepth.NATIVE, np.int32, np.int32),
    ),
)
def test_save_images_converts_and_materializes_through_registered_image_formats(
    format_literal: str,
    depth_literal: str,
    depth: SaveImagesBitDepth,
    input_dtype,
    expected_dtype,
) -> None:
    module = _single_name_module(
        file_format=format_literal,
        bit_depth=depth_literal,
    )
    contract = _contract(module)
    materialization = contract.artifact_outputs[0].materialization
    assert isinstance(materialization, MaterializationSpec)

    main = np.arange(4, dtype=np.float32).reshape(2, 2)
    selected = np.asarray(((0.0, 0.5), (1.0, 0.25)), dtype=input_dtype)
    returned_main, converted = save_images(
        main,
        image_to_save=selected,
        bit_depth=depth,
    )

    np.testing.assert_array_equal(returned_main, main)
    assert np.shares_memory(returned_main, main)
    assert np.asarray(image_payload_data(converted)).dtype == np.dtype(expected_dtype)

    filemanager = FileManager({"memory": MemoryStorageBackend()})
    primary_path = materialize(
        materialization,
        converted,
        "/analysis/SaveImages.pkl",
        filemanager,
        ("memory",),
    )
    expected_suffix = SaveImagesFileFormat[
        "TIFF" if format_literal == "tiff" else format_literal.upper()
    ].value
    assert primary_path == f"/analysis/exports/Saved_result{expected_suffix}"
    stored = filemanager.load(primary_path, "memory")
    assert np.asarray(stored).dtype == np.dtype(expected_dtype)


def test_save_images_projects_declared_singleton_runtime_slice() -> None:
    contract = _contract(
        _single_name_module(
            file_format="png",
            bit_depth="8-bit integer",
        )
    )
    materialization = contract.artifact_outputs[0].materialization
    assert isinstance(materialization, MaterializationSpec)
    payload = ImageMetadataPayload(
        np.ones((1, 3, 4), dtype=np.uint8),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    primary_path = materialize(
        materialization,
        payload,
        "/analysis/SaveImages.pkl",
        filemanager,
        ("memory",),
    )

    assert primary_path == "/analysis/exports/Saved_result.png"
    np.testing.assert_array_equal(
        filemanager.load(primary_path, "memory"),
        np.ones((3, 4), dtype=np.uint8),
    )


def test_save_images_preserves_metadata_backreferences_in_relative_paths() -> None:
    module = _single_name_module(
        file_format="png",
        bit_depth="8-bit integer",
        single_name=r"\g<Specimen>-\g<FrameNumber>",
        output_location=r"Default Output Folder sub-folder|\g<Run>",
    )
    contract = _contract(module)
    options = _image_file_options(contract)
    assert options.relative_path_template == (
        r"\g<Run>/\g<Specimen>-\g<FrameNumber>_result.png"
    )

    payload = ImagePayloadMetadata(
        source_component_metadata={
            "Run": "Sequence1",
            "Specimen": "GFPHistone",
            "FrameNumber": "0007",
        }
    ).payload_with(np.ones((3, 4), dtype=np.uint8), None)
    materialization = contract.artifact_outputs[0].materialization
    assert isinstance(materialization, MaterializationSpec)
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    primary_path = materialize(
        materialization,
        payload,
        "/analysis/SaveImages.pkl",
        filemanager,
        ("memory",),
    )

    assert primary_path == ("/analysis/Sequence1/GFPHistone-0007_result.png")
    np.testing.assert_array_equal(
        filemanager.load(primary_path, "memory"),
        np.ones((3, 4), dtype=np.uint8),
    )


def test_save_images_projects_dynamic_path_metadata_for_each_runtime_slice() -> None:
    contract = _contract(
        _single_name_module(
            file_format="png",
            bit_depth="8-bit integer",
            single_name=r"\g<Specimen>-\g<FrameNumber>",
            output_location=r"Default Output Folder sub-folder|\g<Run>",
        )
    )
    payload = RuntimeSliceAlignedValues(
        tuple(
            ImagePayloadMetadata(
                source_component_metadata={
                    "Run": "Sequence1",
                    "Specimen": "GFPHistone",
                    "FrameNumber": f"{frame:04d}",
                }
            ).payload_with(np.full((3, 4), frame, dtype=np.uint8), None)
            for frame in (1, 2)
        )
    )
    materialization = contract.artifact_outputs[0].materialization
    assert isinstance(materialization, MaterializationSpec)
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    primary_path = materialize(
        materialization,
        payload,
        "/analysis/SaveImages.pkl",
        filemanager,
        ("memory",),
    )

    assert primary_path == "/analysis/Sequence1/GFPHistone-0001_result.png"
    for frame in (1, 2):
        np.testing.assert_array_equal(
            filemanager.load(
                f"/analysis/Sequence1/GFPHistone-{frame:04d}_result.png",
                "memory",
            ),
            np.full((3, 4), frame, dtype=np.uint8),
        )


def test_save_images_projects_sequential_numbers_from_materialization_order() -> None:
    module = _module(
        **{
            "Select the type of image to save": "Image",
            "Select the image to save": "ImageToSave",
            "Select method for constructing file names": "Sequential numbers",
            "Enter single file name": "CroppedFlyImage",
            "Number of digits": "4",
            "Append a suffix to the image file name?": "Yes",
            "Text to append to the image name": "RGB",
            "Saved file format": "tiff",
            "Output file location": "Default Output Folder|",
            "Image bit depth": "8-bit integer",
            "Create subfolders in the output folder?": "No",
        }
    )
    contract = _contract(module)
    options = _image_file_options(contract)
    assert options.relative_path_template == "CroppedFlyImage{index:04d}RGB.tiff"
    payload = RuntimeSliceAlignedValues(
        tuple(
            ImagePayloadMetadata().payload_with(
                np.full((3, 4), item, dtype=np.uint8), None
            )
            for item in (1, 2, 3)
        )
    )
    materialization = contract.artifact_outputs[0].materialization
    assert isinstance(materialization, MaterializationSpec)
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    primary_path = materialize(
        materialization,
        payload,
        "/analysis/SaveImages.pkl",
        filemanager,
        ("memory",),
    )

    assert primary_path == "/analysis/CroppedFlyImage0001RGB.tiff"
    for item in (1, 2, 3):
        np.testing.assert_array_equal(
            filemanager.load(
                f"/analysis/CroppedFlyImage{item:04d}RGB.tiff",
                "memory",
            ),
            np.full((3, 4), item, dtype=np.uint8),
        )


def test_save_images_from_image_filename_uses_source_identity_and_suffix() -> None:
    module = _module(
        **{
            "Select the image to save": "ImageToSave",
            "Select method for constructing file names": "From image filename",
            "Select image name for file prefix": "ImageToSave",
            "Append a suffix to the image file name?": "Yes",
            "Text to append to the image name": "_Overlay",
            "Saved file format": "png",
            "Output file location": "Default Output Folder|",
            "Image bit depth": "8-bit integer",
        }
    )

    options = _image_file_options(_contract(module))

    assert options.filename_identity is MaterializedFilenameIdentity.SOURCE_IDENTITY
    assert options.filename_suffix == "_Overlay.png"
    assert options.relative_path_template is None


def test_save_images_filename_source_is_a_source_identity_contract_input() -> None:
    module = _module(
        **{
            "Select the image to save": "RGBImage",
            "Select method for constructing file names": "From image filename",
            "Select image name for file prefix": "OrigBlue",
            "Append a suffix to the image file name?": "Yes",
            "Text to append to the image name": "RGB",
            "Saved file format": "tiff",
            "Image bit depth": "8-bit integer",
        }
    )

    contract = _contract(
        module,
        image_name="RGBImage",
        source_names=("OrigBlue",),
    )

    assert contract.artifact_inputs.names() == (
        "OrigBlue",
        "RGBImage",
    )
    assert contract.artifact_outputs[0].relations == (
        MaterializationSourceIdentityRelation(
            source=contract.artifact_inputs[0].ref(),
        ),
        SourceStackLineageSourceRelation(
            source=contract.artifact_inputs[1].ref(),
        ),
    )
    SaveImagesModule.validate_callable_artifact_abi(save_images, contract)


def test_save_images_missing_selection_preserves_all_declared_image_choices() -> None:
    invocation = next(normalize_function_pattern(save_images).iter_items())
    produced = ArtifactSpec.output("Produced", ImageArtifactType)
    step_context = ArtifactDeclarationStepContext(
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(NamedSourceBinding(alias="OrigBlue"),),
        ),
        available_artifacts=ArtifactSpecCollection(
            (
                ArtifactSpec.input("OrigBlue", ImageArtifactType),
                produced,
            )
        ),
        main_flow_artifacts=ArtifactSpecCollection(
            (ArtifactSpec.input("OrigBlue", ImageArtifactType),)
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            (produced,),
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
    blocks, consumed = SaveImagesModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )

    assert consumed == ()
    assert tuple(
        SaveImagesModule.artifact_names_for_binding(
            block,
            SaveImagesModule.selected_image_binding,
        )
        for block in blocks
    ) == (("OrigBlue",), ("Produced",))

    explicit_invocation = next(
        normalize_function_pattern(
            (save_images, {"select_the_image_to_save": "OrigBlue"})
        ).iter_items()
    )
    explicit_blocks, explicit_consumed = SaveImagesModule.module_blocks_for_invocation(
        invocation=explicit_invocation,
        step_context=step_context,
    )

    assert explicit_consumed == ("select_the_image_to_save",)
    assert tuple(
        SaveImagesModule.artifact_names_for_binding(
            block,
            SaveImagesModule.selected_image_binding,
        )
        for block in explicit_blocks
    ) == (("OrigBlue",),)


def test_save_images_adapter_free_compile_binds_selected_runtime_image() -> None:
    module_contract = _contract(
        _single_name_module(file_format="png", bit_depth="8-bit integer")
    )
    selected = module_contract.artifact_inputs[0]
    materialized = module_contract.artifact_outputs[0]

    compiled = compile_function_pattern(
        save_images,
        {plan.ref(): plan for plan in (ArtifactInputPlan(
                selected.name,
                "/tmp/save-images-selected.pkl",
                artifact_type=selected.artifact_type,
            ),)},
        {plan.ref(): plan for plan in (ArtifactOutputPlan(
                materialized.name,
                "/tmp/save-images-materialized.pkl",
                artifact_type=materialized.artifact_type,
            ),)},
        invocation_contract_provider=lambda _invocation, _context: (
            InvocationContractPlan(module_contract)
        ),
    )
    invocation = next(compiled.iter_invocations())

    assert invocation.contract is module_contract
    assert invocation.contract.artifact_inputs.specs == (selected,)
    assert selected.parameter_name == "image_to_save"
    assert invocation.artifact_output_plans[0].ref() == materialized.ref()


def test_image_output_context_projects_declared_group_lineage_source() -> None:
    source_slices = RuntimeSliceAlignedValues(
        tuple(
            ImagePayloadMetadata(
                source_channel_axis=-1,
                source_image_names=("OrigRed", "OrigGreen", "OrigBlue"),
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=tuple(
                            f"/source/site{site}_{channel}.tif"
                            for channel in ("R", "F", "D")
                        ),
                        component_metadata=tuple(
                            {
                                "site": str(site),
                                "source_alias": alias,
                            }
                            for alias in (
                                "OrigRed",
                                "OrigGreen",
                                "OrigBlue",
                            )
                        ),
                    )
                ),
            ).payload_with(np.zeros((2, 2, 3), dtype=np.uint8), None)
            for site in range(1, 4)
        )
    )
    output_plan = ArtifactOutputPlan(
        name="SavedRGB",
        path="/memory/SavedRGB.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input("OrigBlue", ImageArtifactType).ref(),
            ),
        ),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        project_declared_source_identity(
            source_slices,
            output_plan.source_context_source(),
        ),
        np.zeros((3, 2, 2, 3), dtype=np.uint8),
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        ),
    )

    assert image_payload_metadata(
        contextualized
    ).source_image_provenance_planes.paths == tuple(
        f"/source/site{site}_D.tif" for site in range(1, 4)
    )


def test_declared_group_lineage_source_overrides_complete_output_identity() -> None:
    aliases = ("OrigRed", "OrigGreen", "OrigBlue")
    source_slices = RuntimeSliceAlignedValues(
        tuple(
            ImagePayloadMetadata(
                source_channel_axis=-1,
                source_image_names=aliases,
                source_image_provenance_planes=(
                    SourceImageProvenancePlanes.from_components(
                        paths=tuple(
                            f"/source/site{site}_{channel}.tif"
                            for channel in ("R", "G", "B")
                        ),
                        component_metadata=tuple(
                            {
                                "site": str(site),
                                "source_alias": alias,
                            }
                            for alias in aliases
                        ),
                    )
                ),
            ).payload_with(np.zeros((2, 2, 3), dtype=np.uint8), None)
            for site in range(1, 4)
        )
    )
    complete_output = ImagePayloadMetadata(
        source_channel_axis=-1,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=tuple(alias for _site in range(1, 4) for alias in aliases),
        source_image_provenance_planes=(
            SourceImageProvenancePlanes.from_components(
                paths=tuple(
                    f"/source/site{site}_{channel}.tif"
                    for site in range(1, 4)
                    for channel in ("R", "G", "B")
                ),
                component_metadata=tuple(
                    {
                        "site": str(site),
                        "source_alias": alias,
                    }
                    for site in range(1, 4)
                    for alias in aliases
                ),
            )
        ),
    ).payload_with(np.zeros((3, 2, 2, 3), dtype=np.uint8), None)
    output_plan = ArtifactOutputPlan(
        name="SavedRGB",
        path="/memory/SavedRGB.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input("OrigBlue", ImageArtifactType).ref(),
            ),
        ),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        project_declared_source_identity(
            source_slices,
            output_plan.source_context_source(),
        ),
        complete_output,
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        ),
    )

    assert image_payload_metadata(
        contextualized
    ).source_image_provenance_planes.paths == tuple(
        f"/source/site{site}_B.tif" for site in range(1, 4)
    )


def test_planned_image_output_restores_axis_without_duplicate_provenance() -> None:
    plane_count = 60
    metadata = ImagePayloadMetadata(
        source_image_names=("CellsImage",) * plane_count,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/source/z{index:03d}.tif" for index in range(plane_count)),
            component_metadata=tuple(
                {"z_index": str(index), "source_alias": "CellsImage"}
                for index in range(plane_count)
            ),
        ),
    )
    source_payload = metadata.payload_with(
        np.zeros((plane_count, 2, 2), dtype=np.uint16),
        None,
    )
    output_plan = ArtifactOutputPlan(
        name="SavedCells",
        path="/memory/SavedCells.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input("CellsImage", ImageArtifactType).ref(),
            ),
        ),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source_payload,
        source_payload,
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=plane_count,
        ),
    )

    contextualized_metadata = image_payload_metadata(contextualized)
    assert contextualized_metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert contextualized_metadata.source_image_provenance_planes.paths == (
        metadata.source_image_provenance_planes.paths
    )
    assert contextualized_metadata.source_image_provenance_planes.component_metadata == (
        metadata.source_image_provenance_planes.component_metadata
    )
    assert contextualized_metadata.source_provenance.source_plane_count == plane_count


def test_image_output_context_removes_selected_source_runtime_axis() -> None:
    source_payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_names=("OrigRed", "OrigGreen", "OrigBlue"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/source/red.tif", "/source/green.tif", "/source/blue.tif"),
            component_metadata=tuple(
                {"channel": str(index), "source_alias": alias}
                for index, alias in enumerate(
                    ("OrigRed", "OrigGreen", "OrigBlue"),
                    start=1,
                )
            ),
        ),
    ).payload_with(np.zeros((3, 2, 2), dtype=np.uint8), None)
    output_plan = ArtifactOutputPlan(
        name="SavedRGB",
        path="/memory/SavedRGB.pkl",
        artifact_type=ImageArtifactType,
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input("OrigBlue", ImageArtifactType).ref(),
            ),
        ),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        project_declared_source_identity(
            source_payload,
            output_plan.source_context_source(),
        ),
        np.zeros((2, 2, 3), dtype=np.uint8),
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        ),
    )

    metadata = image_payload_metadata(contextualized)
    assert metadata.plane_axis is None
    assert metadata.source_path == "/source/blue.tif"


def test_image_output_context_preserves_repeated_declared_source_planes() -> None:
    source_payload = ImagePayloadMetadata(
        source_image_names=("Original", "Original"),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/source/site1.tif", "/source/site2.tif"),
            component_metadata=(
                {"site": "1", "source_alias": "Original"},
                {"site": "2", "source_alias": "Original"},
            ),
        ),
    ).payload_with(np.zeros((2, 2, 2), dtype=np.uint8), None)
    output_plan = ArtifactOutputPlan(
        name="SavedImage",
        path="/memory/SavedImage.pkl",
        artifact_type=ImageArtifactType,
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input("Original", ImageArtifactType).ref(),
            ),
        ),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        project_declared_source_identity(
            source_payload,
            output_plan.source_context_source(),
        ),
        np.zeros((2, 2, 2), dtype=np.uint8),
        output_plan,
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert image_payload_metadata(
        contextualized
    ).source_image_provenance_planes.paths == (
        "/source/site1.tif",
        "/source/site2.tif",
    )


def test_image_output_context_accepts_exact_loaded_derived_artifact() -> None:
    selected_artifact = ArtifactSpec.output(
        "SaveImages_17_image_1",
        ImageArtifactType,
    )
    source_payload = ImagePayloadMetadata(
        source_image_names=("OutlinedNatural",),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/source/original.tif",),
            component_metadata=({"source_alias": "OutlinedNatural", "site": "1"},),
        ),
    ).payload_with(np.zeros((2, 2), dtype=np.uint8), None)
    output_plan = ArtifactOutputPlan(
        name="SaveImages_18_image_1",
        path="/memory/SaveImages_18_image_1.pkl",
        artifact_type=ImageArtifactType,
        relations=(SourceStackLineageSourceRelation(source=selected_artifact.ref()),),
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source_payload,
        np.ones((2, 2), dtype=np.uint8),
        output_plan,
        None,
    )

    metadata = image_payload_metadata(contextualized)
    assert metadata.source_image_provenance_planes.paths == ("/source/original.tif",)
    assert metadata.source_image_names == ("OutlinedNatural",)


@pytest.mark.parametrize("file_format", ("hdf5", "definitely-unknown"))
def test_save_images_rejects_unregistered_image_formats_at_compilation(
    file_format: str,
) -> None:
    module = _single_name_module(
        file_format=file_format,
        bit_depth="8-bit integer",
    )

    with pytest.raises(ValueError, match="serialization format|coerced"):
        _contract(module)


def test_save_images_bit_depth_conversion_preserves_image_metadata() -> None:
    metadata = ImagePayloadMetadata(
        source_path="/source/A01_s001_w1_z001_t001.tif",
        source_component_metadata={"well": "A01", "channel": "1"},
    )
    payload = metadata.payload_with(
        np.asarray(((0.0, 1.0), (0.25, 0.75)), dtype=np.float32), None
    )

    converted = SaveImagesBitDepth.UINT16.convert(payload)

    assert image_payload_metadata(converted) == metadata
    assert np.asarray(image_payload_data(converted)).dtype == np.uint16
    np.testing.assert_array_equal(
        image_payload_data(converted),
        np.asarray(((0, 65535), (16384, 49151)), dtype=np.uint16),
    )
