"""Architectural guards for public CellProfiler/OpenHCS pipeline boundaries."""

from pathlib import Path

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ImageArtifactType
from openhcs.core.pipeline_image_schema import ImageAssignment, PipelineImageSchema
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceBindingOrigin,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerCompileTimeArtifactFlow,
    CellProfilerCompileTimeSettingsRequest,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.classification import (
    ClassifyObjectsSingleMeasurementModule,
)
from openhcs.processing.backends.cellprofiler.alignment import AlignModule
from openhcs.processing.backends.cellprofiler.color import (
    ColorToGrayMode,
    ColorToGrayModule,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    MeasureColocalizationModule,
)
from openhcs.processing.backends.cellprofiler.crop import CropModule
from openhcs.processing.backends.cellprofiler.illumination import (
    CorrectIlluminationApplyModule,
    CorrectIlluminationCalculateModule,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathModule,
    ImageMathOperation,
)
from openhcs.processing.backends.cellprofiler.image_quality import (
    MeasureImageQualityModule,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    MaskImageModule,
    MaskSource,
    ResizeMethod,
    ResizeModule,
    TileModule,
)
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathInvocationOptions,
    CalculateMathModule,
    calculate_math_object_dependencies,
)
from openhcs.processing.backends.cellprofiler.morphology import MaskObjectsModule
from openhcs.processing.backends.cellprofiler.shape import MeasureObjectSizeShapeModule
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedDeclumpMethod,
    WatershedMethod,
    WatershedModule,
)
from openhcs.processing.backends.cellprofiler.worms import StraightenWormsModule


PROJECT_ROOT = Path(__file__).parents[2]


def _source(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_cellprofiler_compiler_does_not_use_selected_cppipe_contract_fallback() -> None:
    """Converted pipelines must compile from public steps/config, not .cppipe state."""

    source = _source("openhcs/interop/cellprofiler/compile_time_contracts.py")

    assert "_runtime_contracts_from_selected_cppipe" not in source
    assert "_runtime_contracts_from_cppipe_path" not in source
    assert "selected_pipeline_path" not in source
    assert "input_workspace_preparation_result" not in source


def test_cellprofiler_import_and_ui_paths_do_not_rebind_runtime_contracts() -> None:
    """UI/import code may load public pipelines, but must not attach contracts."""

    checked_paths = (
        "openhcs/interop/cellprofiler/runtime_pipeline.py",
        "openhcs/pyqt_gui/widgets/shared/services/cellprofiler_pipeline_rebinding.py",
    )
    forbidden_terms = (
        "CellProfilerPipelineRuntimeRebinder",
        "GeneratedPipelineRuntimeBindings",
        ".rebind(",
        ".apply()",
        "invocation_contracts",
    )

    for path in checked_paths:
        source = _source(path)
        for term in forbidden_terms:
            assert term not in source, f"{path} still uses {term}"


def test_generated_import_module_has_no_execution_contract_sidecar() -> None:
    """Generated import modules should not smuggle runtime contracts into steps."""

    source = _source("openhcs/interop/cellprofiler/runtime/generated_pipeline.py")

    forbidden_terms = (
        "_openhcs_cp_contract_values",
        "GeneratedPipelineRuntimeBindings(",
        "bind_generated_pipeline_runtime(",
        "CellProfilerPipelineRuntimeRebinder",
    )

    for term in forbidden_terms:
        assert term not in source


def _source_image_schema(*aliases: str) -> PipelineImageSchema:
    return PipelineImageSchema(
        assignments_by_alias={
            alias: ImageAssignment(
                alias=alias,
                selector=SourceSelector(),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(),
                image_type="grayscale image",
            )
            for alias in aliases
        }
    )


def test_correct_illumination_calculate_keeps_only_ambiguous_public_identity() -> None:
    """Derived-image selectors stay public, source-bound canonical rows stay implicit."""

    derived_module = ModuleBlock(
        "CorrectIlluminationCalculate",
        2,
        settings={
            "Select the input image": "OrigRed",
            "Name the output image": "IllumRed",
        },
        setting_records=[
            ModuleSetting("Select the input image", "OrigRed"),
            ModuleSetting("Name the output image", "IllumRed"),
        ],
    )
    source_bound_module = ModuleBlock(
        "CorrectIlluminationCalculate",
        3,
        settings={
            "Select the input image": "OrigStain1",
            "Name the output image": "IllumStain1",
        },
        setting_records=[
            ModuleSetting("Select the input image", "OrigStain1"),
            ModuleSetting("Name the output image", "IllumStain1"),
        ],
    )

    assert CorrectIlluminationCalculateModule.compile_time_public_setting_records(
        derived_module,
        _source_image_schema("OrigColor"),
    ) == (ModuleSetting("Select the input image", "OrigRed"),)
    assert CorrectIlluminationCalculateModule.compile_time_public_setting_records(
        source_bound_module,
        _source_image_schema("OrigStain1"),
    ) == ()


def test_correct_illumination_apply_keeps_noncanonical_public_identity() -> None:
    """CorrectIlluminationApply derives canonical rows but preserves real CP names."""

    module = ModuleBlock(
        "CorrectIlluminationApply",
        9,
        settings={
            "Select the input image": "OrigRed",
            "Select the illumination function": "IllumRed",
            "Name the output image": "CorrRed",
        },
        setting_records=[
            ModuleSetting("Select the input image", "OrigRed"),
            ModuleSetting("Select the illumination function", "IllumRed"),
            ModuleSetting("Name the output image", "CorrRed"),
        ],
    )

    assert CorrectIlluminationApplyModule.compile_time_public_setting_records(
        module,
        _source_image_schema("OrigColor"),
    ) == (
        ModuleSetting("Select the input image", "OrigRed"),
        ModuleSetting("Name the output image", "CorrRed"),
    )


def test_align_keeps_derived_image_role_selectors_public() -> None:
    """Align source references can inherit while derived role inputs stay explicit."""

    module = ModuleBlock(
        "Align",
        11,
        settings={
            "Select the first input image": "PlateTemplate",
            "Name the first output image": "AlignedPlate",
            "Select the second input image": "CorrRed",
            "Name the second output image": "AlignedRed",
            "Select the additional image": "CombinedImage",
            "Name the output image": "AlignedCombined",
        },
        setting_records=[
            ModuleSetting("Select the first input image", "PlateTemplate"),
            ModuleSetting("Name the first output image", "AlignedPlate"),
            ModuleSetting("Select the second input image", "CorrRed"),
            ModuleSetting("Name the second output image", "AlignedRed"),
            ModuleSetting("Select the additional image", "CombinedImage"),
            ModuleSetting("Name the output image", "AlignedCombined"),
        ],
    )

    records = AlignModule.compile_time_public_setting_records(
        module,
        _source_image_schema("PlateTemplate"),
    )

    assert (
        ModuleSetting("Select the first input image", "PlateTemplate") not in records
    )
    assert records == (
        ModuleSetting("Select the second input image", "CorrRed"),
        ModuleSetting("Select the additional image", "CombinedImage"),
        ModuleSetting("Name the first output image", "AlignedPlate"),
        ModuleSetting("Name the second output image", "AlignedRed"),
        ModuleSetting("Name the output image", "AlignedCombined"),
    )


def test_measurement_output_identity_is_public_nominal_compile_kwarg() -> None:
    """Standard measurement identity belongs to the measurement-output module family."""

    module = ModuleBlock("ClassifyObjects", 18)
    public_kwargs = ClassifyObjectsSingleMeasurementModule.compile_time_public_kwargs(
        module
    )

    assert public_kwargs == {
        ClassifyObjectsSingleMeasurementModule.compile_time_measurement_artifact_name_kwarg: (
            "ClassifyObjects_18_measurements"
        )
    }
    assert (
        ClassifyObjectsSingleMeasurementModule.compile_time_measurement_artifact_name_kwarg
        in ClassifyObjectsSingleMeasurementModule.compile_time_consumed_kwarg_names()
    )

    metadata = (
        ClassifyObjectsSingleMeasurementModule.compile_time_module_metadata_for_invocation(
            CellProfilerCompileTimeSettingsRequest(
                module_name="ClassifyObjectsSingleMeasurement",
                module_num=16,
                kwargs=public_kwargs,
            )
        )
    )
    reconstructed = ModuleBlock(
        "ClassifyObjectsSingleMeasurement",
        16,
        metadata=dict(metadata),
    )

    assert ClassifyObjectsSingleMeasurementModule.measurement_artifact_name(
        reconstructed
    ) == "ClassifyObjects_18_measurements"


def test_tile_reconstructs_repeated_inputs_from_public_flow() -> None:
    """Repeated CP image inputs come from public source bindings and artifact flow."""

    flow = (
        CellProfilerCompileTimeArtifactFlow.empty()
        .with_image_names("default", ("OutlineImage",))
        .with_available_image_names("default", ("TrackedCells",))
    )

    records = TileModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="Tile",
            module_num=7,
            kwargs={},
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(NamedSourceBinding(alias="OrigColor"),),
            ),
            artifact_flow=flow,
        )
    )

    image_input_records = tuple(
        (record.name, record.value)
        for record in records
        if record.name
        in {
            "Select an input image",
            "Select an additional image to tile",
        }
    )

    assert image_input_records == (
        ("Select an input image", "OrigColor"),
        ("Select an additional image to tile", "OutlineImage"),
        ("Select an additional image to tile", "TrackedCells"),
    )


def test_calculate_math_reconstructs_count_object_dependencies_from_public_kwargs() -> None:
    """Count_* operands must compile object-label inputs from public declarations."""

    records = CalculateMathModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="CalculateMath",
            module_num=9,
            kwargs={
                "operation": ImageMathOperation.DIVIDE,
                "operand1_feature": "Count_PH3PosNuclei",
                "operand2_feature": "Count_Nuclei",
            },
            invocation_options=CalculateMathInvocationOptions(
                output_name="PercentPositive",
            ),
        )
    )
    module = ModuleBlock(
        name="CalculateMath",
        module_num=9,
        settings={record.name: record.value for record in records},
        setting_records=list(records),
    )

    assert calculate_math_object_dependencies(module) == (
        "PH3PosNuclei",
        "Nuclei",
    )


def test_mask_image_object_mask_infers_only_primary_image_from_public_flow() -> None:
    """Object-mask mode gets the mask from object labels, not a second image input."""

    records = MaskImageModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MaskImage",
            module_num=3,
            kwargs={
                "mask_source": MaskSource.OBJECTS,
                "select_object_for_mask": "Nuclei",
                "name_the_output_image": "MaskedGreen",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("EnhancedGreen",),
            ),
        )
    )

    assert tuple(
        (record.name, record.value)
        for record in records
        if record.name in {"Select the input image", "Select image for mask"}
    ) == (("Select the input image", "EnhancedGreen"),)


def test_mask_image_image_mask_uses_public_image_selectors() -> None:
    """Image-mask mode keeps explicit primary and mask image selectors public."""

    records = MaskImageModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MaskImage",
            module_num=10,
            kwargs={
                "mask_source": MaskSource.IMAGE,
                "select_the_input_image": "CombinedImage",
                "select_image_for_mask": "AlignedPlate",
                "name_the_output_image": "MaskedCombined",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("AlignedPlate", "AlignedRed", "AlignedCombined"),
            ),
        )
    )

    assert tuple(
        (record.name, record.value)
        for record in records
        if record.name in {"Select the input image", "Select image for mask"}
    ) == (
        ("Select the input image", "CombinedImage"),
        ("Select image for mask", "AlignedPlate"),
    )


def test_mask_image_image_mask_public_records_include_image_selectors() -> None:
    """Generated MaskImage image-mask source must include both image selectors."""

    module = ModuleBlock(
        name="MaskImage",
        module_num=12,
        settings={
            "Select the input image": "CombinedImage",
            "Name the output image": "MaskedCombined",
            "Use objects or an image as a mask?": "Image",
            "Select object for mask": "None",
            "Select image for mask": "AlignedPlate",
            "Invert the mask?": "No",
        },
        setting_records=[
            ModuleSetting("Select the input image", "CombinedImage"),
            ModuleSetting("Name the output image", "MaskedCombined"),
            ModuleSetting("Use objects or an image as a mask?", "Image"),
            ModuleSetting("Select object for mask", "None"),
            ModuleSetting("Select image for mask", "AlignedPlate"),
            ModuleSetting("Invert the mask?", "No"),
        ],
    )

    public_records = MaskImageModule.compile_time_public_setting_records(module)

    assert tuple(
        (record.name, record.value)
        for record in public_records
        if record.name in {"Select the input image", "Select image for mask"}
    ) == (
        ("Select the input image", "CombinedImage"),
        ("Select image for mask", "AlignedPlate"),
    )


def test_crop_image_mask_public_records_include_primary_and_mask_images() -> None:
    """Crop image-mask mode owns both primary-image and mask-image selectors."""

    module = ModuleBlock(
        name="Crop",
        module_num=5,
        settings={},
        setting_records=[
            ModuleSetting("Select the input image", "Worms"),
            ModuleSetting("Name the output image", "WormsCropped"),
            ModuleSetting("Select the cropping shape", CropModule.Shape.IMAGE.value),
            ModuleSetting("Select the masking image", "ErodedWellEdge"),
        ],
    )

    public_records = CropModule.compile_time_public_setting_records(module)

    assert tuple(
        (record.name, record.value)
        for record in public_records
        if record.name
        in {
            "Select the input image",
            "Select the masking image",
        }
    ) == (
        ("Select the input image", "Worms"),
        ("Select the masking image", "ErodedWellEdge"),
    )


def test_crop_image_mask_reconstructs_primary_and_mask_contract_inputs() -> None:
    """Compiler reconstruction must not treat the mask image as the primary image."""

    from openhcs.interop.cellprofiler.artifact_semantics import (
        DeclaredArtifactSymbolCollector,
    )
    from openhcs.interop.cellprofiler.symbol_table import (
        CellProfilerContractAssemblyMixin,
    )

    records = CropModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="Crop",
            module_num=5,
            kwargs={
                "crop_shape": CropModule.Shape.IMAGE,
                "select_the_input_image": "Worms",
                "name_the_output_image": "WormsCropped",
                "select_the_masking_image": "ErodedWellEdge",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("ErodedWellEdge",),
            ),
        )
    )
    module = ModuleBlock(
        name="Crop",
        module_num=5,
        settings={record.name: record.value for record in records},
        setting_records=list(records),
    )

    assert tuple(
        (record.name, record.value)
        for record in records
        if record.name
        in {
            "Select the input image",
            "Select the masking image",
        }
    ) == (
        ("Select the input image", "Worms"),
        ("Select the masking image", "ErodedWellEdge"),
    )
    assert CropModule.input_image_name(module) == "Worms"
    assert tuple(input_spec.name for input_spec in CropModule.mask_inputs(module)) == (
        "ErodedWellEdge",
    )
    contract = CropModule.artifact_contract(
        CellProfilerContractAssemblyMixin(),
        DeclaredArtifactSymbolCollector(),
        module,
    )

    assert tuple(input_spec.name for input_spec in contract.inputs) == (
        "Worms",
        "ErodedWellEdge",
    )


def test_resize_by_factor_requires_only_primary_image_input() -> None:
    """Resize's optional dimensions image must not be inferred for factor resize."""

    records = ResizeModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="Resize",
            module_num=2,
            kwargs={
                "resize_method": ResizeMethod.BY_FACTOR,
                "name_the_output_image": "ResizedDNA",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("RescaledDNA",),
            ),
        )
    )

    assert tuple(
        (record.name, record.value)
        for record in records
        if record.name
        in {
            "Select the input image",
            "Select the image with the desired dimensions",
        }
    ) == (("Select the input image", "RescaledDNA"),)


def test_watershed_distance_shape_requires_only_segmentation_image() -> None:
    """Inactive marker/mask/reference image settings must not be required."""

    records = WatershedModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="Watershed",
            module_num=10,
            kwargs={
                "watershed_method": WatershedMethod.DISTANCE,
                "declump_method": WatershedDeclumpMethod.SHAPE,
                "name_the_output_object": "downsizedNuclei",
                "measurement_artifact_name": "Watershed_10_measurements",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("noHolesMaskDNA",),
            ),
        )
    )

    assert tuple(
        (record.name, record.value)
        for record in records
        if record.name
        in {
            "Select the input image",
            "Markers",
            "Reference Image",
            "Mask",
        }
    ) == (("Select the input image", "noHolesMaskDNA"),)


def test_watershed_marker_mask_preserves_primary_image_mask_role() -> None:
    """Marker mode needs an explicit mask role even when it equals the primary image."""

    from openhcs.interop.cellprofiler.artifact_semantics import (
        DeclaredArtifactSymbolCollector,
    )
    from openhcs.interop.cellprofiler.symbol_table import (
        CellProfilerContractAssemblyMixin,
    )

    module = ModuleBlock(
        name="Watershed",
        module_num=25,
        settings={},
        setting_records=[
            ModuleSetting("Select the input image", "MembFinal"),
            ModuleSetting("Generate from", WatershedMethod.MARKERS.value),
            ModuleSetting("Markers", "cellSeeds"),
            ModuleSetting("Mask", "MembFinal"),
            ModuleSetting("Declump method", WatershedDeclumpMethod.SHAPE.value),
            ModuleSetting("Name the output object", "Cells"),
        ],
    )
    artifact_flow = CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
        "default",
        ("MembFinal", "cellSeeds"),
    )

    public_records = WatershedModule.compile_time_public_setting_records_for_generation(
        module,
        artifact_flow=artifact_flow,
        group_key="default",
    )

    assert ("Mask", "MembFinal") in tuple(
        (record.name, record.value) for record in public_records
    )

    records = WatershedModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="Watershed",
            module_num=25,
            kwargs={
                "watershed_method": WatershedMethod.MARKERS,
                "declump_method": WatershedDeclumpMethod.SHAPE,
                "markers": "cellSeeds",
                "mask": "MembFinal",
                "name_the_output_object": "Cells",
                "measurement_artifact_name": "Watershed_25_measurements",
            },
            artifact_flow=artifact_flow,
        )
    )
    reconstructed = ModuleBlock(
        name="Watershed",
        module_num=25,
        settings={record.name: record.value for record in records},
        setting_records=list(records),
    )
    contract = WatershedModule.artifact_contract(
        CellProfilerContractAssemblyMixin(),
        DeclaredArtifactSymbolCollector(),
        reconstructed,
    )

    assert tuple(input_spec.name for input_spec in contract.inputs) == (
        "MembFinal",
        "cellSeeds",
        "MembFinal",
    )


def test_color_to_gray_split_reads_sparse_public_output_rows() -> None:
    """Split-mode public source stores selected output names, not CP offset rows."""

    records = [
        ModuleSetting("Select the input image", "OrigColor"),
        ModuleSetting("Conversion method", ColorToGrayModule.ConversionMethod.SPLIT.value),
        ModuleSetting("Image type", ColorToGrayModule.ImageType.RGB.value),
        ModuleSetting("Convert red to gray?", "Yes"),
        ModuleSetting("Convert green to gray?", "No"),
        ModuleSetting("Convert blue to gray?", "Yes"),
        ModuleSetting("Name the output image", "OrigRed"),
        ModuleSetting("Name the output image", "OrigBlue"),
    ]
    module = ModuleBlock(
        name="ColorToGray",
        module_num=1,
        settings={record.name: record.value for record in records},
        setting_records=records,
    )

    assert ColorToGrayModule.output_image_names(module) == ("OrigRed", "OrigBlue")


def test_color_to_gray_public_records_emit_sparse_selected_output_identity() -> None:
    """Generated public kwargs preserve non-canonical outputs without CP offsets."""

    records = [
        ModuleSetting("Select the input image", "ColorFluor"),
        ModuleSetting("Conversion method", ColorToGrayModule.ConversionMethod.SPLIT.value),
        ModuleSetting("Image type", ColorToGrayModule.ImageType.RGB.value),
        ModuleSetting("Name the output image", "OrigGray"),
        ModuleSetting("Convert red to gray?", "No"),
        ModuleSetting("Name the output image", "OrigRed"),
        ModuleSetting("Convert green to gray?", "Yes"),
        ModuleSetting("Name the output image", "GrayTumor"),
        ModuleSetting("Convert blue to gray?", "No"),
        ModuleSetting("Name the output image", "OrigBlue"),
    ]
    module = ModuleBlock(
        name="ColorToGray",
        module_num=6,
        settings={record.name: record.value for record in records},
        setting_records=records,
    )

    public_records = ColorToGrayModule.compile_time_public_setting_records(module)

    assert tuple(
        (record.name, record.value)
        for record in public_records
        if record.name == "Name the output image"
    ) == (("Name the output image", "GrayTumor"),)


def test_color_to_gray_compile_time_uses_callable_defaults_for_clean_source() -> None:
    """Compiler rebuilds CP setting rows even when pycodify omits default kwargs."""

    records = ColorToGrayModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="ColorToGray",
            module_num=1,
            kwargs={
                "mode": ColorToGrayMode.COMBINE,
                "name_the_output_image": "OrigGray",
            },
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(NamedSourceBinding(alias="OrigColor"),),
            ),
        )
    )

    assert ("Image type", "rgb") in tuple(
        (record.name, record.value) for record in records
    )


def test_color_to_gray_compile_time_output_advances_artifact_flow() -> None:
    """ColorToGray owns its mode-dependent output names for public-step compilation."""

    records = [
        ModuleSetting("Conversion method", ColorToGrayModule.ConversionMethod.COMBINE.value),
        ModuleSetting("Image type", ColorToGrayModule.ImageType.RGB.value),
        ModuleSetting("Name the output image", "OrigGray"),
    ]
    module = ModuleBlock(
        name="ColorToGray",
        module_num=1,
        settings={record.name: record.value for record in records},
        setting_records=records,
    )

    assert ColorToGrayModule.compile_time_image_output_names(module) == ("OrigGray",)
    updated_flow = ColorToGrayModule.compile_time_artifact_flow_after_invocation(
        CellProfilerCompileTimeArtifactFlow.empty(),
        group_key="default",
        module=module,
    )

    assert updated_flow.image_names_for_group("default") == ("OrigGray",)


def test_image_measurement_input_infers_single_main_flow_image() -> None:
    """Image measurement modules derive hidden image settings from artifact flow."""

    records = MeasureObjectIntensityModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MeasureObjectIntensity",
            module_num=4,
            kwargs={
                "select_object_sets_to_measure": "Embryos",
                "measurement_artifact_name": "MeasureObjectIntensity_4_measurements",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("OrigGray",),
            ),
        )
    )

    assert ("Select images to measure", "OrigGray") in tuple(
        (record.name, record.value) for record in records
    )


def test_image_measurement_public_records_preserve_non_current_measurement_image() -> None:
    """Public generated kwargs must carry image selectors when current flow differs."""

    public_records = MeasureObjectIntensityModule.compile_time_public_setting_records(
        ModuleBlock(
            name="MeasureObjectIntensity",
            module_num=12,
            settings={
                "Select images to measure": "OrigRed",
                "Select objects to measure": "Cells",
            },
            setting_records=[
                ModuleSetting("Select images to measure", "OrigRed"),
                ModuleSetting("Select objects to measure", "Cells"),
            ],
        )
    )

    assert ("Select images to measure", "OrigRed") in tuple(
        (record.name, record.value) for record in public_records
    )

    records = MeasureObjectIntensityModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MeasureObjectIntensity",
            module_num=12,
            kwargs={
                "select_images_to_measure": "OrigRed",
                "select_object_sets_to_measure": "Cells",
                "measurement_artifact_name": "MeasureObjectIntensity_12_measurements",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("InvertedRed",),
            ),
        )
    )

    assert ("Select images to measure", "OrigRed") in tuple(
        (record.name, record.value) for record in records
    )
    assert ("Select images to measure", "InvertedRed") not in tuple(
        (record.name, record.value) for record in records
    )


def test_generated_public_repeated_settings_expand_list_values_to_rows() -> None:
    """Generated list literals for repeated CP rows remain repeated settings."""

    records = MeasureObjectSizeShapeModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MeasureObjectSizeShape",
            module_num=11,
            kwargs={
                "select_object_sets_to_measure": [
                    "Comet",
                    "CometHead",
                    "CometTail",
                ],
                "measurement_artifact_name": "MeasureObjectSizeShape_11_measurements",
            },
        )
    )

    assert tuple(
        record.value
        for record in records
        if record.name == "Select object sets to measure"
    ) == ("Comet", "CometHead", "CometTail")


def test_mask_objects_public_settings_preserve_masking_object_owner() -> None:
    """MaskObjects carries its masking-object selector as public compile data."""

    public_records = MaskObjectsModule.compile_time_public_setting_records(
        ModuleBlock(
            name="MaskObjects",
            module_num=10,
            settings={},
            setting_records=[
                ModuleSetting("Select objects to be masked", "Comet"),
                ModuleSetting("Name the masked objects", "CometTail"),
                ModuleSetting(
                    "Mask using a region defined by other objects or by binary image?",
                    "Objects",
                ),
                ModuleSetting("Select the masking object", "CometHead"),
                ModuleSetting("Select the masking image", "None"),
            ],
        )
    )

    assert ("Select the masking object", "CometHead") in tuple(
        (record.name, record.value) for record in public_records
    )

    records = MaskObjectsModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MaskObjects",
            module_num=10,
            kwargs={
                "select_the_input_objects": "Comet",
                "name_the_output_objects": "CometTail",
                "select_the_masking_object": "CometHead",
                "measurement_artifact_name": "MaskObjects_10_measurements",
            },
        )
    )

    assert ("Select the masking object", "CometHead") in tuple(
        (record.name, record.value) for record in records
    )


def test_image_math_public_settings_preserve_active_image_operands() -> None:
    """ImageMath operand identities are module-owned compile-time selectors."""

    public_records = ImageMathModule.compile_time_public_setting_records(
        ModuleBlock(
            name="ImageMath",
            module_num=14,
            settings={},
            setting_records=[
                ModuleSetting("Operation", "Subtract"),
                ModuleSetting("Name the output image", "SubtractedRed"),
                ModuleSetting("Image or measurement?", "Image"),
                ModuleSetting("Select the first image", "MaskedRedPlate"),
                ModuleSetting("Multiply the first image by", "1.0"),
                ModuleSetting("Measurement", ""),
                ModuleSetting("Image or measurement?", "Image"),
                ModuleSetting("Select the second image", "MaskedCombined"),
                ModuleSetting("Multiply the second image by", "1.0"),
                ModuleSetting("Measurement", ""),
            ],
        )
    )

    assert (
        ("Select the first image", "MaskedRedPlate") in tuple(
            (record.name, record.value) for record in public_records
        )
    )
    assert (
        ("Select the second image", "MaskedCombined") in tuple(
            (record.name, record.value) for record in public_records
        )
    )

    records = ImageMathModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="ImageMath",
            module_num=14,
            kwargs={
                "operation": ImageMathOperation.SUBTRACT,
                "select_the_first_image": "MaskedRedPlate",
                "select_the_second_image": "MaskedCombined",
                "name_the_output_image": "SubtractedRed",
            },
        )
    )

    assert ("Select the first image", "MaskedRedPlate") in tuple(
        (record.name, record.value) for record in records
    )
    assert ("Select the second image", "MaskedCombined") in tuple(
        (record.name, record.value) for record in records
    )


def test_image_quality_public_settings_preserve_selected_images() -> None:
    """MeasureImageQuality Select mode carries selected source image names."""

    public_records = MeasureImageQualityModule.compile_time_public_setting_records(
        ModuleBlock(
            name="MeasureImageQuality",
            module_num=17,
            settings={},
            setting_records=[
                ModuleSetting("Calculate metrics for which images?", "Select..."),
                ModuleSetting("Select the images to measure", "OrigBlue"),
                ModuleSetting("Select the images to measure", "OrigGreen"),
                ModuleSetting("Select the images to measure", "OrigRed"),
            ],
        )
    )

    assert tuple(
        record.value
        for record in public_records
        if record.name == "Select the images to measure"
    ) == ("OrigBlue", "OrigGreen", "OrigRed")

    records = MeasureImageQualityModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MeasureImageQuality",
            module_num=17,
            kwargs={
                "calculate_metrics_for_which_images": "Select...",
                "select_the_images_to_measure": [
                    "OrigBlue",
                    "OrigGreen",
                    "OrigRed",
                ],
                "measurement_artifact_name": "MeasureImageQuality_17_measurements",
            },
        )
    )

    assert tuple(
        record.value
        for record in records
        if record.name == "Select the images to measure"
    ) == ("OrigBlue", "OrigGreen", "OrigRed")


def test_image_measurement_input_infers_multi_image_setting_from_flow() -> None:
    """A single CP image-measurement row can represent multiple OpenHCS images."""

    records = MeasureColocalizationModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="MeasureColocalization",
            module_num=4,
            kwargs={},
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("Stain1", "Stain2"),
            ),
        )
    )

    assert ("Select images to measure", "Stain1, Stain2") in tuple(
        (record.name, record.value) for record in records
    )


def test_image_measurement_input_joins_source_bound_images_without_group_split() -> None:
    """Source-bound measurement images stay one CP measurement-image setting."""

    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                artifact_kind=ImageArtifactType,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="PH3",
                artifact_kind=ImageArtifactType,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
    )
    artifact_flow = (
        CellProfilerCompileTimeArtifactFlow.empty()
        .with_image_names("1", ("DNA",))
        .with_image_names("2", ("PH3",))
    )
    request = CellProfilerCompileTimeSettingsRequest(
        module_name="MeasureObjectIntensity",
        module_num=10,
        kwargs={
            "select_object_sets_to_measure": ("Nuclei", "Cells", "Cytoplasm"),
            "measurement_artifact_name": "MeasureObjectIntensity_10_measurements",
        },
        source_bindings=source_bindings,
        artifact_flow=artifact_flow,
    )

    assert (
        MeasureObjectIntensityModule.compile_time_source_binding_group_keys_for_invocation(
            request
        )
        == ()
    )
    records = MeasureObjectIntensityModule.compile_time_setting_records_for_invocation(
        request
    )

    assert ("Select images to measure", "DNA, PH3") in tuple(
        (record.name, record.value) for record in records
    )


def test_image_math_unary_operation_infers_only_first_main_flow_operand() -> None:
    """Unary ImageMath operations should not require all optional operand rows."""

    records = ImageMathModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="ImageMath",
            module_num=3,
            kwargs={
                "operation": ImageMathOperation.INVERT,
                "name_the_output_image": "WormsInverted",
            },
            artifact_flow=CellProfilerCompileTimeArtifactFlow.empty().with_image_names(
                "default",
                ("CorrectedWorms",),
            ),
        )
    )

    assert tuple(
        (record.name, record.value)
        for record in records
        if record.name in ImageMathModule.image_operand_settings
    ) == (("Select the first image", "CorrectedWorms"),)


def test_straighten_worms_reconstructs_paired_identity_kwargs() -> None:
    """StraightenWorms repeated input/output image identities stay paired."""

    records = StraightenWormsModule.compile_time_setting_records_for_invocation(
        CellProfilerCompileTimeSettingsRequest(
            module_name="StraightenWorms",
            module_num=7,
            kwargs={
                "select_the_input_untangled_worm_objects": "NonOverlappingWorms",
                "name_the_output_straightened_worm_objects": "StraightenedWorms",
                "select_an_input_image_to_straighten": ("mCherry", "GFP"),
                "name_the_output_straightened_image": (
                    "Straightened_mCherry",
                    "Straightened_GFP",
                ),
            },
        )
    )
    module = ModuleBlock(
        name="StraightenWorms",
        module_num=7,
        settings={record.name: record.value for record in records},
        setting_records=list(records),
    )

    assert StraightenWormsModule.input_objects_name(module) == "NonOverlappingWorms"
    assert StraightenWormsModule.output_objects_name(module) == "StraightenedWorms"
    assert tuple(
        (binding.input_image_name, binding.output_image_name)
        for binding in StraightenWormsModule.image_bindings(module)
    ) == (
        ("mCherry", "Straightened_mCherry"),
        ("GFP", "Straightened_GFP"),
    )
