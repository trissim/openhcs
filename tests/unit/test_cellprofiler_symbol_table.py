from pathlib import Path
from types import SimpleNamespace
import pytest
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.module_processing_components import (
    ModuleProcessingComponents,
    source_binding_variable_component_literals,
)
from openhcs.interop.cellprofiler.pipeline_generator import (
    PipelineGenerator,
    python_literal,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope,
)
from openhcs.interop.cellprofiler.runtime_pipeline import partition_cppipe_modules
from openhcs.processing.backends.cellprofiler.outlines import OverlayOutlinesModule
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbolTable
from openhcs.core.artifacts import (
    ArtifactSpecRef,
    ArtifactSidecarRole,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.runtime_semantics import parent_child_relationship_artifact_name
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
    StepSourceBindingsConfig,
)
from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.processing.backends.cellprofiler.tracking import TrackObjectsModule
from benchmark.cellprofiler_library.functions.rescaleintensity import RescaleMethod


def test_pipeline_generator_queries_module_declarations_for_module_semantics():
    generator_source = (
        Path(__file__).parents[2]
        / "openhcs"
        / "interop"
        / "cellprofiler"
        / "pipeline_generator.py"
    ).read_text()
    assert "_ModuleSettingsBindingStrategy" not in generator_source
    assert "_ModuleFunctionResolutionStrategy" not in generator_source
    assert "default_module_processing_components" not in generator_source


def test_module_declarations_do_not_define_settings_binding_strategy():
    module_declarations_source = (
        Path(__file__).parents[2]
        / "openhcs"
        / "interop"
        / "cellprofiler"
        / "module_declarations.py"
    ).read_text()
    assert "settings_binding_strategy" not in module_declarations_source


def test_cellprofiler_artifact_capabilities_are_registered_product_terms():
    from openhcs.processing.backends.cellprofiler.intensity import (
        RescaleIntensityModule,
    )
    from openhcs.interop.cellprofiler.module_declarations import (
        CellProfilerArtifactCapability,
        ImageArtifactInputCapability,
        ImageArtifactOutputCapability,
        MeasurementArtifactOutputCapability,
        RelationshipArtifactInputCapability,
        RelationshipArtifactOutputCapability,
    )
    from openhcs.processing.backends.cellprofiler.morphology import ResizeObjectsModule
    from openhcs.processing.backends.cellprofiler.object_filtering import (
        FilterObjectsModule,
    )

    assert (
        CellProfilerArtifactCapability.__registry__["image_input"]
        is ImageArtifactInputCapability
    )
    assert (
        CellProfilerArtifactCapability.__registry__["image_output"]
        is ImageArtifactOutputCapability
    )
    assert issubclass(RescaleIntensityModule, ImageArtifactInputCapability)
    assert issubclass(RescaleIntensityModule, ImageArtifactOutputCapability)
    assert issubclass(ResizeObjectsModule, MeasurementArtifactOutputCapability)
    assert issubclass(ResizeObjectsModule, RelationshipArtifactOutputCapability)
    assert issubclass(FilterObjectsModule, RelationshipArtifactInputCapability)
    assert issubclass(FilterObjectsModule, RelationshipArtifactOutputCapability)
    with pytest.raises(TypeError, match="without inheriting"):
        RelationshipArtifactOutputCapability.bind_artifact(
            RescaleIntensityModule,
            None,
            None,
            RelationshipArtifactOutputCapability.spec("unexpected_relationship"),
        )


def _module(module_num: int, name: str, settings: dict[str, str]) -> ModuleBlock:
    return ModuleBlock(name=name, module_num=module_num, settings=settings)


def _module_with_records(
    module_num: int, name: str, setting_pairs: list[tuple[str, str]]
) -> ModuleBlock:
    records = [
        ModuleSetting(setting_name, value) for setting_name, value in setting_pairs
    ]
    return ModuleBlock(
        name=name,
        module_num=module_num,
        settings={setting.name: setting.value for setting in records},
        setting_records=records,
    )


def _step_config_universe_for_step(step):
    from openhcs.core.config import ProcessingConfig, StepMaterializationConfig
    from openhcs.core.pipeline.step_config_universe import (
        StepConfigRoot,
        StepConfigUniverse,
        step_config_declarations,
    )

    declarations = step_config_declarations()
    roots = []
    for config in (
        step.source_bindings,
        ProcessingConfig(),
        StepMaterializationConfig(enabled=False),
    ):
        declaration = next(
            declaration
            for declaration in declarations
            if isinstance(config, declaration.config_type)
        )
        roots.append(StepConfigRoot(declaration=declaration, value=config))
    return StepConfigUniverse(tuple(roots))


def test_source_binding_variable_components_derive_timepoint_from_metadata() -> None:
    source_bindings = StepSourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern="^(?P<Specimen>.*)_(?P<Stain>.*)_(?P<FrameNumber>[0-9]*)",
            ),
        )
    )
    assert source_binding_variable_component_literals(source_bindings) == (
        "VariableComponents.SITE",
        "VariableComponents.TIMEPOINT",
    )


def test_generated_kwarg_literal_serializes_enum_members_nominally():
    imports = set()

    assert (
        python_literal(RescaleMethod.STRETCH, import_collector=imports)
        == "RescaleMethod.STRETCH"
    )
    assert (
        python_literal(
            CellProfilerMeasurementTargetScope.BOTH,
            import_collector=imports,
        )
        == "CellProfilerMeasurementTargetScope.BOTH"
    )
    assert (
        python_literal(
            CellProfilerBackendProvider.LEGACY_FAST,
            import_collector=imports,
        )
        == "CellProfilerBackendProvider.LEGACY_FAST"
    )
    assert (
        "openhcs.processing.backends.cellprofiler.intensity",
        "RescaleMethod",
    ) in imports
    assert (
        "openhcs.interop.cellprofiler.measurement_scope",
        "CellProfilerMeasurementTargetScope",
    ) in imports
    assert (
        "openhcs.processing.backends.cellprofiler._backend",
        "CellProfilerBackendProvider",
    ) in imports


def _identify_primary(module_num: int = 1) -> ModuleBlock:
    return _module(
        module_num,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": "OrigBlue",
            "Name the primary objects to be identified": "Nuclei",
        },
    )


def _identify_secondary(module_num: int = 2) -> ModuleBlock:
    return _module(
        module_num,
        "IdentifySecondaryObjects",
        {
            "Select the input objects": "Nuclei",
            "Select the input image": "OrigGreen",
            "Name the objects to be identified": "Cells",
            "Name the new primary objects": "FilteredNuclei",
        },
    )


def _identify_tertiary(module_num: int = 3) -> ModuleBlock:
    return _module(
        module_num,
        "IdentifyTertiaryObjects",
        {
            "Select the larger identified objects": "Cells",
            "Select the smaller identified objects": "Nuclei",
            "Name the tertiary objects to be identified": "Cytoplasm",
        },
    )


def test_cellprofiler_symbol_table_compiles_object_measurement_graph():
    modules = [
        _identify_primary(),
        _identify_secondary(),
        _identify_tertiary(),
        _module(
            4,
            "MeasureObjectIntensity",
            {
                "Select images to measure": "OrigBlue, OrigGreen",
                "Select objects to measure": "Nuclei, Cells, Cytoplasm",
            },
        ),
        _module(
            5,
            "MeasureImageIntensity",
            {"Select images to measure": "OrigBlue", "Select input object sets": ""},
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    assert (
        table.symbol_for("OrigBlue", ImageArtifactType).artifact_type
        is ImageArtifactType
    )
    assert table.symbol_for("OrigBlue", ImageArtifactType).producer_module_num is None
    assert (
        table.symbol_for("Nuclei", ObjectLabelsArtifactType).artifact_type
        is ObjectLabelsArtifactType
    )
    assert table.symbol_for("Nuclei", ObjectLabelsArtifactType).producer_module_num == 1
    assert (
        table.symbol_for("Cytoplasm", ObjectLabelsArtifactType).artifact_type
        is ObjectLabelsArtifactType
    )
    assert (
        table.symbol_for(
            "MeasureObjectIntensity_4_measurements", MeasurementsArtifactType
        ).artifact_type
        is MeasurementsArtifactType
    )
    primary_contract = table.contracts_by_module_num[1]
    assert [spec.artifact_type for spec in primary_contract.inputs] == [
        ImageArtifactType
    ]
    assert tuple(
        (binding.alias for binding in primary_contract.source_bindings.bindings)
    ) == ("OrigBlue",)
    assert primary_contract.runtime_artifact_inputs == ()
    assert [spec.artifact_type for spec in primary_contract.outputs] == [
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    ]
    assert isinstance(primary_contract.module_contract, ModuleArtifactContract)
    secondary_contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in secondary_contract.outputs] == [
        "IdentifySecondaryObjects_2_measurements",
        "Nuclei_Cells_relationships",
        "Cells",
    ]
    assert [spec.artifact_type for spec in secondary_contract.outputs] == [
        MeasurementsArtifactType,
        RelationshipsArtifactType,
        ObjectLabelsArtifactType,
    ]
    measure_contract = table.contracts_by_module_num[4]
    assert tuple(
        (binding.alias for binding in measure_contract.source_bindings.bindings)
    ) == ("OrigBlue", "OrigGreen")
    assert [spec.name for spec in measure_contract.runtime_artifact_inputs] == [
        "Nuclei",
        "Cells",
        "Cytoplasm",
    ]
    assert measure_contract.outputs[0].artifact_type is MeasurementsArtifactType


def test_simple_image_filter_modules_infer_image_artifact_contracts():
    from openhcs.processing.backends.cellprofiler.gaussian_filter import (
        GaussianFilterModule,
    )
    from openhcs.processing.backends.cellprofiler.smoothing import ReducenoiseModule

    assert GaussianFilterModule.module_name == "GaussianFilter"
    assert ReducenoiseModule.module_name == "Reducenoise"

    modules = [
        _module(
            1,
            "GaussianFilter",
            {
                "Select the input image": "Nuclei",
                "Name the output image": "GaussianFilter",
                "Sigma": "1",
            },
        ),
        _module(
            2,
            "ReduceNoise",
            {
                "Select the input image": "GaussianFilter",
                "Name the output image": "ReduceNoise",
                "Size": "5",
                "Distance": "2",
                "Cut-off distance": "0.2",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)

    gaussian_contract = table.contracts_by_module_num[1]
    assert [spec.name for spec in gaussian_contract.inputs] == ["Nuclei"]
    assert [spec.artifact_type for spec in gaussian_contract.inputs] == [
        ImageArtifactType
    ]
    assert [spec.name for spec in gaussian_contract.outputs] == ["GaussianFilter"]
    assert [spec.artifact_type for spec in gaussian_contract.outputs] == [
        ImageArtifactType
    ]

    reduce_noise_contract = table.contracts_by_module_num[2]
    assert reduce_noise_contract.module_name == "Reducenoise"
    assert [spec.name for spec in reduce_noise_contract.inputs] == ["GaussianFilter"]
    assert [spec.artifact_type for spec in reduce_noise_contract.inputs] == [
        ImageArtifactType
    ]
    assert [spec.name for spec in reduce_noise_contract.outputs] == ["ReduceNoise"]
    assert [spec.artifact_type for spec in reduce_noise_contract.outputs] == [
        ImageArtifactType
    ]


def test_dilate_objects_infers_singular_object_artifact_settings():
    from openhcs.processing.backends.cellprofiler.morphology import DilateObjectsModule
    from openhcs.processing.backends.cellprofiler.primary_objects import (
        IdentifyPrimaryObjectsModule,
    )

    assert IdentifyPrimaryObjectsModule.module_name == "IdentifyPrimaryObjects"
    assert DilateObjectsModule.module_name == "DilateObjects"

    modules = [
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "Nuclei",
                "Name the primary objects to be identified": "ReduceNoiseObjects",
            },
        ),
        _module(
            2,
            "DilateObjects",
            {
                "Select the input object": "ReduceNoiseObjects",
                "Name the output object": "DilatedObjects",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)

    dilate_objects_contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in dilate_objects_contract.inputs] == [
        "ReduceNoiseObjects"
    ]
    assert [spec.artifact_type for spec in dilate_objects_contract.inputs] == [
        ObjectLabelsArtifactType
    ]
    assert [spec.name for spec in dilate_objects_contract.outputs] == [
        "DilateObjects_2_measurements",
        "DilatedObjects",
    ]
    assert [spec.artifact_type for spec in dilate_objects_contract.outputs] == [
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    ]


def test_watershed_contract_preserves_marker_image_dependency():
    modules = [
        _identify_primary(),
        _module(
            2,
            "ConvertObjectsToImage",
            {
                "Select the input objects": "Nuclei",
                "Name the output image": "cellSeeds",
                "Select the color format": "Binary",
            },
        ),
        _module(
            3,
            "Watershed",
            {
                "Select the input image": "MembFinal",
                "Generate from": "Markers",
                "Markers": "cellSeeds",
                "Mask": "MembFinal",
                "Name the output object": "Cells",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    convert_contract = table.contracts_by_module_num[2]
    watershed_contract = table.contracts_by_module_num[3]
    assert [spec.name for spec in convert_contract.inputs] == ["Nuclei"]
    assert [spec.name for spec in convert_contract.outputs] == ["cellSeeds"]
    assert [spec.name for spec in watershed_contract.inputs] == [
        "MembFinal",
        "cellSeeds",
        "MembFinal",
    ]
    assert [spec.name for spec in watershed_contract.runtime_artifact_inputs] == [
        "cellSeeds"
    ]
    assert [spec.name for spec in watershed_contract.outputs] == [
        "Watershed_3_measurements",
        "Cells",
    ]


def test_watershed_contract_preserves_same_runtime_image_as_mask_role():
    modules = [
        _module(
            1,
            "Threshold",
            {
                "Select the input image": "OrigMembrane",
                "Name the output image": "MembFinal",
            },
        ),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigDNA",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
        _module(
            3,
            "ConvertObjectsToImage",
            {
                "Select the input objects": "Nuclei",
                "Name the output image": "cellSeeds",
                "Select the color format": "uint16",
            },
        ),
        _module(
            4,
            "Watershed",
            {
                "Select the input image": "MembFinal",
                "Generate from": "Markers",
                "Markers": "cellSeeds",
                "Mask": "MembFinal",
                "Name the output object": "Cells",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    watershed_contract = table.contracts_by_module_num[4]
    assert [spec.name for spec in watershed_contract.inputs] == [
        "MembFinal",
        "cellSeeds",
        "MembFinal",
    ]
    assert [spec.name for spec in watershed_contract.runtime_artifact_inputs] == [
        "MembFinal",
        "cellSeeds",
    ]


def test_watershed_contract_preserves_reference_image_dependency():
    modules = [
        _module(
            1,
            "Threshold",
            {
                "Select the input image": "Input",
                "Name the output image": "Threshold",
            },
        ),
        _module(
            2,
            "GaussianFilter",
            {
                "Select the input image": "Input",
                "Name the output image": "GaussianFilter",
            },
        ),
        _module(
            3,
            "Watershed",
            {
                "Select the input image": "Threshold",
                "Generate from": "Distance",
                "Declump method": "Intensity",
                "Reference Image": "GaussianFilter",
                "Name the output object": "Watershed",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    watershed_contract = table.contracts_by_module_num[3]
    assert [spec.name for spec in watershed_contract.inputs] == [
        "Threshold",
        "GaussianFilter",
    ]
    assert [spec.name for spec in watershed_contract.runtime_artifact_inputs] == [
        "Threshold",
        "GaussianFilter",
    ]


def test_cellprofiler_symbol_table_fails_for_unknown_object_input():
    modules = [
        _module(
            1,
            "MeasureObjectSizeShape",
            {"Select object sets to measure": "MissingObjects"},
        )
    ]
    with pytest.raises(
        ValueError, match="unknown object_labels symbol 'MissingObjects'"
    ):
        CellProfilerSymbolTable.compile(modules)


def test_measure_object_neighbors_declares_retained_image_outputs():
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "MeasureObjectNeighbors",
            [
                ("Select objects to measure", "Nuclei"),
                ("Select neighboring objects to measure", "Nuclei"),
                ("Retain the image of objects colored by numbers of neighbors?", "Yes"),
                ("Name the output image", "ColorNeighbors"),
                ("Select colormap", "hot"),
                (
                    "Retain the image of objects colored by percent of touching pixels?",
                    "No",
                ),
                ("Name the output image", "PercentTouching"),
                ("Select colormap", "Default"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in contract.outputs] == [
        "ColorNeighbors",
        "MeasureObjectNeighbors_2_measurements",
    ]
    assert [spec.artifact_type for spec in contract.outputs] == [
        ImageArtifactType,
        MeasurementsArtifactType,
    ]


def test_cellprofiler_symbol_table_accepts_declared_source_object_inputs():
    setup_module = _module_with_records(
        1,
        "NamesAndTypes",
        [
            ("Assignments count", "1"),
            ("Select the rule criteria", 'and (metadata does channel "3")'),
            ("Name to assign these images", "IgnoredImageAlias"),
            ("Name to assign these objects", "LoadedNuclei"),
            ("Select the image type", "Objects"),
        ],
    )
    measurement_module = _module(
        2, "MeasureObjectSizeShape", {"Select object sets to measure": "LoadedNuclei"}
    )
    table = CellProfilerSymbolTable.compile([setup_module, measurement_module])
    contract = table.contracts_by_module_num[2]
    assert (
        table.symbol_for("LoadedNuclei", ObjectLabelsArtifactType).source_bound is True
    )
    assert contract.runtime_artifact_inputs == ()
    assert (
        contract.source_bindings.bindings[0].artifact_kind is ObjectLabelsArtifactType
    )
    assert [spec.name for spec in contract.inputs] == ["LoadedNuclei"]
    assert [spec.artifact_type for spec in contract.inputs] == [
        ObjectLabelsArtifactType
    ]


def test_cellprofiler_symbol_table_infers_bare_objects_subscriber_input():
    modules = [
        _identify_primary(),
        _module(
            2,
            "OverlayObjects",
            {
                "Input": "OrigBlue",
                "Objects": "Nuclei",
                "Name the output image": "NucleiOverlay",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in contract.inputs] == ["OrigBlue", "Nuclei"]
    assert [spec.artifact_type for spec in contract.inputs] == [
        ImageArtifactType,
        ObjectLabelsArtifactType,
    ]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == ["Nuclei"]
    assert [spec.artifact_type for spec in contract.runtime_artifact_inputs] == [
        ObjectLabelsArtifactType
    ]
    assert [spec.name for spec in contract.outputs] == ["NucleiOverlay"]


def test_cellprofiler_symbol_table_ignores_object_method_choice_values():
    modules = [
        _identify_primary(),
        _identify_secondary(),
        _module(
            3,
            "CombineObjects",
            {
                "Select initial object set": "Nuclei",
                "Select object set to combine": "Cells",
                "Select how to handle overlapping objects": "Merge",
                "Name the combined object set": "CombinedObjects",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == [
        "Nuclei",
        "Cells",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "CombineObjects_3_measurements",
        "CombinedObjects",
    ]
    assert "Merge" not in {
        spec.name for spec in (*contract.inputs, *contract.runtime_artifact_inputs)
    }


def test_maskimage_output_group_lineage_follows_primary_image_input():
    modules = [
        _module(
            1,
            "Threshold",
            {
                "Select the input image": "OrigMemb",
                "Name the output image": "MembThreshold",
            },
        ),
        _module(
            2,
            "ImageMath",
            {
                "Operation": "Invert",
                "Select the first image": "MembThreshold",
                "Name the output image": "MembInvert",
            },
        ),
        _module(
            3,
            "RemoveHoles",
            {
                "Select the input image": "MembInvert",
                "Name the output image": "MembInvertRemoveHoles",
            },
        ),
        _module(
            4,
            "Threshold",
            {
                "Select the input image": "OrigDAPI",
                "Name the output image": "MonolayerMask",
            },
        ),
        _module(
            5,
            "MaskImage",
            {
                "Select the input image": "MembInvertRemoveHoles",
                "Select image for mask": "MonolayerMask",
                "Use objects or an image as a mask?": "Image",
                "Name the output image": "MembMasked",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[5]

    assert contract.outputs[0].relations == (
        GroupLineageSourceRelation(
            source=ArtifactSpecRef.input(
                "MembInvertRemoveHoles",
                ImageArtifactType,
            )
        ),
    )


def test_single_image_transform_output_group_lineage_follows_image_input():
    modules = [
        _module(
            1,
            "Threshold",
            {
                "Select the input image": "OrigMemb",
                "Name the output image": "MembMasked",
            },
        ),
        _module(
            2,
            "ErodeImage",
            {
                "Select the input image": "MembMasked",
                "Name the output image": "MembFinal",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)
    source_bound_contract = table.contracts_by_module_num[1]
    assert source_bound_contract.outputs[0].relations == (
        GroupLineageSourceRelation(
            source=ArtifactSpecRef.input(
                "OrigMemb",
                ImageArtifactType,
            )
        ),
    )
    contract = table.contracts_by_module_num[2]

    assert contract.outputs[0].relations == (
        GroupLineageSourceRelation(
            source=ArtifactSpecRef.input(
                "MembMasked",
                ImageArtifactType,
            )
        ),
    )


def test_watershed_object_output_group_lineage_follows_segmentation_image():
    modules = [
        _module(
            1,
            "Threshold",
            {
                "Select the input image": "OrigMemb",
                "Name the output image": "MembFinal",
            },
        ),
        _module(
            2,
            "Threshold",
            {
                "Select the input image": "OrigSeeds",
                "Name the output image": "cellSeeds",
            },
        ),
        _module(
            3,
            "Watershed",
            {
                "Select the input image": "MembFinal",
                "Markers": "cellSeeds",
                "Mask": "MembFinal",
                "Name the output object": "Cells",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]
    object_outputs = [
        spec for spec in contract.outputs if spec.artifact_type is ObjectLabelsArtifactType
    ]

    assert object_outputs[0].relations == (
        GroupLineageSourceRelation(
            source=ArtifactSpecRef.input(
                "MembFinal",
                ImageArtifactType,
            )
        ),
    )


def test_cellprofiler_symbol_table_compiles_filterobjects_relabel_rows():
    modules = [
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "MyObjects",
            },
        ),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Cells",
            },
        ),
        _module(
            3,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Cytoplasm",
            },
        ),
        _module_with_records(
            4,
            "FilterObjects",
            [
                ("Name the output objects", "MyFilteredObjects"),
                ("Select the object to filter", "MyObjects"),
                ("Filter using classifier rules or measurements?", "Measurements"),
                ("Select the filtering method", "Limits"),
                ("Select additional object to relabel", "Cells"),
                ("Name the relabeled objects", "FilteredCells"),
                ("Save outlines of relabeled objects?", "No"),
                ("Name the outline image", "OutlinesFilteredCells"),
                ("Select additional object to relabel", "Cytoplasm"),
                ("Name the relabeled objects", "FilteredCytoplasm"),
                ("Save outlines of relabeled objects?", "No"),
                ("Name the outline image", "OutlinesFilteredCytoplasm"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[4]
    assert [spec.name for spec in contract.inputs] == [
        "MyObjects",
        "Cells",
        "Cytoplasm",
        "IdentifyPrimaryObjects_1_measurements",
        "IdentifyPrimaryObjects_2_measurements",
        "IdentifyPrimaryObjects_3_measurements",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "FilterObjects_4_measurements",
        "MyFilteredObjects",
        "FilteredCells",
        "FilteredCytoplasm",
        parent_child_relationship_artifact_name("MyObjects", "MyFilteredObjects"),
        parent_child_relationship_artifact_name("Cells", "FilteredCells"),
        parent_child_relationship_artifact_name("Cytoplasm", "FilteredCytoplasm"),
    ]
    assert [spec.artifact_type for spec in contract.outputs] == [
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
        ObjectLabelsArtifactType,
        ObjectLabelsArtifactType,
        RelationshipsArtifactType,
        RelationshipsArtifactType,
        RelationshipsArtifactType,
    ]


def test_cellprofiler_symbol_table_compiles_filterobjects_outline_outputs():
    modules = [
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "MyObjects",
            },
        ),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Cells",
            },
        ),
        _module_with_records(
            3,
            "FilterObjects",
            [
                ("Name the output objects", "MyFilteredObjects"),
                ("Select the object to filter", "MyObjects"),
                (
                    "Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?",
                    "Yes",
                ),
                ("Name the outline image", "FilteredObjects"),
                ("Select additional object to relabel", "Cells"),
                ("Name the relabeled objects", "FilteredCells"),
                ("Save outlines of relabeled objects?", "Yes"),
                ("Name the outline image", "OutlinesFilteredCells"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]
    assert [spec.name for spec in contract.outputs] == [
        "FilterObjects_3_measurements",
        "MyFilteredObjects",
        "FilteredCells",
        parent_child_relationship_artifact_name("MyObjects", "MyFilteredObjects"),
        parent_child_relationship_artifact_name("Cells", "FilteredCells"),
        "FilteredObjects",
        "OutlinesFilteredCells",
    ]
    assert [spec.artifact_type for spec in contract.outputs] == [
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
        ObjectLabelsArtifactType,
        RelationshipsArtifactType,
        RelationshipsArtifactType,
        ImageArtifactType,
        ImageArtifactType,
    ]


def test_cellprofiler_symbol_table_compiles_filterobjects_enclosing_input():
    modules = [
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Cells",
            },
        ),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Tiles",
            },
        ),
        _module(
            3,
            "FilterObjects",
            {
                "Select the objects to filter": "Cells",
                "Name the output objects": "OneCellPerTile",
                "Select the filtering mode": "Measurements",
                "Select the filtering method": "Maximal per object",
                "Select the objects that contain the filtered objects": "Tiles",
                "Assign overlapping child to": "Both parents",
                "Select the measurement to filter by": "AreaShape_Area",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == [
        "Cells",
        "Tiles",
        "IdentifyPrimaryObjects_1_measurements",
        "IdentifyPrimaryObjects_2_measurements",
    ]
    assert [spec.name for spec in contract.inputs] == [
        "Cells",
        "Tiles",
        "IdentifyPrimaryObjects_1_measurements",
        "IdentifyPrimaryObjects_2_measurements",
    ]


def test_filterobjects_uses_prior_enclosing_relationship_when_available():
    modules = [
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Cells",
            },
        ),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Tiles",
            },
        ),
        _module(
            3, "RelateObjects", {"Parent objects": "Tiles", "Child objects": "Cells"}
        ),
        _module(
            4,
            "FilterObjects",
            {
                "Select the objects to filter": "Cells",
                "Name the output objects": "OneCellPerTile",
                "Select the filtering mode": "Measurements",
                "Select the filtering method": "Maximal per object",
                "Select the objects that contain the filtered objects": "Tiles",
                "Assign overlapping child to": "Both parents",
                "Select the measurement to filter by": "AreaShape_Area",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[4]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == [
        "Cells",
        "Tiles",
        "IdentifyPrimaryObjects_1_measurements",
        "IdentifyPrimaryObjects_2_measurements",
        "RelateObjects_3_measurements",
        parent_child_relationship_artifact_name("Tiles", "Cells"),
    ]
    assert [spec.artifact_type for spec in contract.runtime_artifact_inputs] == [
        ObjectLabelsArtifactType,
        ObjectLabelsArtifactType,
        MeasurementsArtifactType,
        MeasurementsArtifactType,
        MeasurementsArtifactType,
        RelationshipsArtifactType,
    ]


def test_filterobjects_children_count_rule_uses_prior_relationship_input():
    modules = [
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigGreen",
                "Name the primary objects to be identified": "PH3",
            },
        ),
        _module(
            3, "RelateObjects", {"Parent objects": "Nuclei", "Child objects": "PH3"}
        ),
        _module(
            4,
            "FilterObjects",
            {
                "Select the objects to filter": "Nuclei",
                "Name the output objects": "PH3PosNuclei",
                "Select the filtering mode": "Measurements",
                "Select the filtering method": "Limits",
                "Select the measurement to filter by": "Children_PH3_Count",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[4]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == [
        "Nuclei",
        "IdentifyPrimaryObjects_1_measurements",
        "IdentifyPrimaryObjects_2_measurements",
        "RelateObjects_3_measurements",
        parent_child_relationship_artifact_name("Nuclei", "PH3"),
    ]
    assert [spec.artifact_type for spec in contract.runtime_artifact_inputs] == [
        ObjectLabelsArtifactType,
        MeasurementsArtifactType,
        MeasurementsArtifactType,
        MeasurementsArtifactType,
        RelationshipsArtifactType,
    ]


def test_cellprofiler_symbol_table_fails_for_kind_conflict():
    modules = [
        _identify_primary(),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "Nuclei",
                "Name the primary objects to be identified": "OtherObjects",
            },
        ),
    ]
    with pytest.raises(ValueError, match="expects 'Nuclei' as image"):
        CellProfilerSymbolTable.compile(modules)


def test_cellprofiler_symbol_table_updates_current_binding_for_reused_names():
    modules = [
        _identify_primary(),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigGreen",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
        _module(
            3, "MeasureObjectSizeShape", {"Select object sets to measure": "Nuclei"}
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    assert table.symbol_for("Nuclei", ObjectLabelsArtifactType).producer_module_num == 2
    assert table.contracts_by_module_num[1].output_symbols[0].producer_module_num == 1
    assert table.contracts_by_module_num[2].output_symbols[0].producer_module_num == 2
    assert table.contracts_by_module_num[3].input_symbols[0].producer_module_num == 2


def test_cellprofiler_symbol_table_allows_declared_image_object_name_overlap():
    setup_module = _module_with_records(
        1,
        "NamesAndTypes",
        [
            ("Assignments count", "1"),
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "PH3"),
            ("Name to assign these objects", "Cell"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (file does contain "d1.tif")'),
        ],
    )
    identify_module = _module(
        2,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": "PH3",
            "Name the primary objects to be identified": "PH3",
        },
    )
    table = CellProfilerSymbolTable.compile([setup_module, identify_module])
    image_symbol = table.symbol_for("PH3", ImageArtifactType)
    object_symbol = table.symbol_for("PH3", ObjectLabelsArtifactType)
    assert image_symbol.source_bound is True
    assert object_symbol.producer_module_num == 2


def test_cellprofiler_symbol_table_accepts_relate_objects_schema_aliases():
    modules = [
        _identify_primary(),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigGreen",
                "Name the primary objects to be identified": "PH3",
            },
        ),
        _module(
            3, "RelateObjects", {"Parent objects": "Nuclei", "Child objects": "PH3"}
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]
    assert [symbol.name for symbol in contract.input_symbols] == ["Nuclei", "PH3"]
    assert [spec.artifact_type for spec in contract.outputs] == [
        RelationshipsArtifactType,
        MeasurementsArtifactType,
    ]


def test_relateobjects_saved_children_declares_lineage_relationship() -> None:
    modules = [
        _identify_primary(),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigGreen",
                "Name the primary objects to be identified": "Nucleoli",
            },
        ),
        _module(
            3,
            "RelateObjects",
            {
                "Parent objects": "Nuclei",
                "Child objects": "Nucleoli",
                "Do you want to save the children with parents as a new object set?": "Yes",
                "Name the output object": "NucleoliChildObjects",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]
    assert [spec.name for spec in contract.outputs] == [
        "NucleoliChildObjects",
        parent_child_relationship_artifact_name("Nuclei", "Nucleoli"),
        parent_child_relationship_artifact_name("Nucleoli", "NucleoliChildObjects"),
        "RelateObjects_3_measurements",
    ]
    assert [spec.artifact_type for spec in contract.outputs] == [
        ObjectLabelsArtifactType,
        RelationshipsArtifactType,
        RelationshipsArtifactType,
        MeasurementsArtifactType,
    ]


def test_pipeline_generator_emits_compiled_artifact_contracts():
    generator = PipelineGenerator()
    modules = [_identify_primary(), _identify_secondary()]
    generated = generator.generate_from_registry(
        pipeline_name="cp_graph", source_cppipe=Path("source.cppipe"), modules=modules
    )
    assert len(generated.artifact_contracts) == 2
    assert len(generated.runtime_module_contracts) == 2
    assert "CELLPROFILER_MODULE_CONTRACTS" not in generated.code
    assert "benchmark.cellprofiler_library" not in generated.code
    assert "benchmark.cellprofiler_compat" not in generated.code
    assert (
        "from openhcs.interop.cellprofiler.runtime.generated_pipeline import"
        not in generated.code
    )
    assert "require_cellprofiler_function" not in generated.code
    assert "attach_callable_contract_metadata" not in generated.code
    assert "CellProfilerAbsorbedFunctionBinding" not in generated.code
    assert "CellProfilerModuleRuntimeBinding" not in generated.code
    assert "CellProfilerModuleContractBinding" not in generated.code
    assert "_CELLPROFILER_RUNTIME_CONTRACTS_BY_MODULE_NUM = {" not in generated.code
    assert "ModuleArtifactContract(" not in generated.code
    assert "CellProfilerModuleSettingsKwarg" not in generated.code
    assert "CellProfilerModuleSettingsPayload" not in generated.code
    assert "FunctionInvocationKey" not in generated.code
    assert "compile_time_kwargs=" not in generated.code
    assert "source_bindings=LazyStepSourceBindingsConfig(" in generated.code
    assert "enabled=True" in generated.code
    assert "source_bindings=StepSourceBindingsConfig(" not in generated.code
    assert (
        generated.runtime_module_contracts_by_module_num[2]
        .runtime_artifact_inputs[0]
        .name
        == "Nuclei"
    )
    assert "identify_primary_objects," in generated.code
    assert "identify_secondary_objects," in generated.code
    assert "cellprofiler_module_callable" not in generated.code
    assert "CellProfilerModuleExecutor" not in generated.code
    assert "cellprofiler_runtime_adapter_factory" not in generated.code
    assert "@module_artifact_contract" not in generated.code
    assert "@artifact_outputs(*CELLPROFILER_MODULE_CONTRACTS" not in generated.code
    assert "@artifact_inputs(*CELLPROFILER_MODULE_CONTRACTS" not in generated.code
    assert '@runtime_adapter("cellprofiler_runtime"' not in generated.code
    assert "identify_primary_objects_1_runtime.input_memory_type" not in generated.code
    assert "func=identify_primary_objects_1_runtime" not in generated.code
    assert "func=identify_secondary_objects_2_runtime" not in generated.code


def test_compiler_ignores_step_invocation_contracts_for_cellprofiler_modules():
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        normalize_function_pattern,
    )
    from openhcs.core.function_step_invocation_contracts import (
        FunctionStepInvocationContractBinding,
        FunctionStepInvocationContracts,
    )
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import (
        GlobalPipelineConfig,
        PipelineConfig,
    )
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.invocation_artifacts import (
        ArtifactDeclarationStepContext,
        PipelineInvocationContractProviderAuthority,
    )
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.backends.cellprofiler import (
        identify_primary_objects,
        identify_secondary_objects,
    )

    hidden_wrong_contract = ModuleArtifactContract(
        module_name="WrongHiddenContract",
    )

    steps = [
        FunctionStep(
            func=(
                identify_primary_objects,
                {"name_the_primary_objects_to_be_identified": "Nuclei"},
            ),
            name="IdentifyPrimaryObjects",
            invocation_contracts=FunctionStepInvocationContracts(
                (
                    FunctionStepInvocationContractBinding(
                        FunctionInvocationKey.from_callable(
                            identify_primary_objects,
                            "default",
                            0,
                        ),
                        hidden_wrong_contract,
                    ),
                )
            ),
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(
                    NamedSourceBinding(
                        alias="OrigBlue",
                        artifact_kind=ImageArtifactType,
                    ),
                ),
            ),
        ),
        FunctionStep(
            func=(
                identify_secondary_objects,
                {
                    "select_the_input_objects": "Nuclei",
                    "name_the_objects_to_be_identified": "Cells",
                },
            ),
            name="IdentifySecondaryObjects",
            invocation_contracts=FunctionStepInvocationContracts(
                (
                    FunctionStepInvocationContractBinding(
                        FunctionInvocationKey.from_callable(
                            identify_secondary_objects,
                            "default",
                            0,
                        ),
                        hidden_wrong_contract,
                    ),
                )
            ),
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(
                    NamedSourceBinding(
                        alias="OrigGreen",
                        artifact_kind=ImageArtifactType,
                    ),
                ),
            ),
        ),
    ]

    snapshots = tuple(
        StepSnapshot(
            index=index,
            scope_id=f"test::functionstep_{index}",
            name=step.name,
            step_type=step.__class__.__name__,
            enabled=bool(step.enabled),
            is_function_step=True,
            func=step.func,
            invocation_contracts=step.invocation_contracts,
            configs=_step_config_universe_for_step(step),
        )
        for index, step in enumerate(steps)
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=step.name,
                    step_type=step.__class__.__name__,
                    axis_id="A01",
                )
                for index, step in enumerate(steps)
            },
            axis_id="A01",
        ),
        steps=steps,
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={index: object() for index in range(len(steps))},
        snapshots=snapshots,
    )
    provider = PipelineInvocationContractProviderAuthority.provider_for_session(
        session,
    )
    secondary_item = next(normalize_function_pattern(steps[1].func).iter_items())

    plan = provider(
        secondary_item,
        ArtifactDeclarationStepContext(
            step_index=1,
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(NamedSourceBinding(alias="OrigGreen"),),
            ),
        ),
    )

    assert plan is not None
    contract = plan.contract.module_artifact_contract
    assert contract is not None
    assert contract.module_name == "IdentifySecondaryObjects"
    assert contract.module_name != "WrongHiddenContract"
    assert (
        contract.runtime_artifact_inputs[0].name
        == "Nuclei"
    )


def test_compiler_ignores_selected_cppipe_for_cellprofiler_contracts(tmp_path):
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.function_patterns import normalize_function_pattern
    from openhcs.core.invocation_artifacts import (
        ArtifactDeclarationStepContext,
        PipelineInvocationContractProviderAuthority,
    )
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    from openhcs.core.source_bindings import (
        NamedSourceBinding,
        StepSourceBindingsConfig,
    )
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.backends.cellprofiler import color_to_gray
    from openhcs.processing.backends.cellprofiler.color import ColorToGrayMode

    cppipe_path = tmp_path / "selected.cppipe"
    cppipe_path.write_text("", encoding="utf-8")

    step = FunctionStep(
        func=(
            color_to_gray,
            {
                "mode": ColorToGrayMode.COMBINE,
                "image_type": "rgb",
                "name_the_output_image": "OrigGray",
            },
        ),
        name="ColorToGray",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigColor",
                    artifact_kind=ImageArtifactType,
                ),
            ),
        ),
    )
    snapshots = (
        StepSnapshot(
            index=0,
            scope_id="selected-cppipe::functionstep_0",
            name=step.name,
            step_type=step.__class__.__name__,
            enabled=bool(step.enabled),
            is_function_step=True,
            func=step.func,
            invocation_contracts=step.invocation_contracts,
            configs=_step_config_universe_for_step(step),
        ),
    )
    plate_path = tmp_path / "plate"
    plate_path.mkdir()
    orchestrator = PipelineOrchestrator(
        plate_path=plate_path,
        selected_pipeline_path=cppipe_path,
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name=step.name,
                    step_type=step.__class__.__name__,
                    axis_id="A01",
                )
            },
            axis_id="A01",
        ),
        steps=(step,),
        orchestrator=orchestrator,
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=snapshots,
    )
    provider = PipelineInvocationContractProviderAuthority.provider_for_session(
        session,
    )
    item = next(normalize_function_pattern(step.func).iter_items())

    plan = provider(
        item,
        ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=step.source_bindings,
        ),
    )

    assert plan is not None
    contract = plan.contract.module_artifact_contract
    assert contract is not None
    assert contract.module_name == "ColorToGray"
    assert [spec.name for spec in contract.inputs] == [
        "OrigColor",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "OrigGray",
    ]


def test_generated_cellprofiler_public_steps_compile_expected_contracts_without_sidecars():
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.function_patterns import normalize_function_pattern
    from openhcs.core.invocation_artifacts import (
        ArtifactDeclarationStepContext,
        PipelineInvocationContractProviderAuthority,
    )
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.core.pipeline.step_snapshot import StepSnapshot
    modules = (_identify_primary(), _identify_secondary())
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="raw_generated_contract_parity",
        source_cppipe=Path("source.cppipe"),
        modules=list(modules),
    )
    expected_contracts = generated.runtime_module_contracts_by_module_num

    namespace: dict = {"__name__": "public_generated_contract_parity"}
    exec(compile(generated.code, "<generated-cellprofiler-pipeline>", "exec"), namespace)
    generated_steps = tuple(namespace["pipeline_steps"])
    assert generated_steps
    assert all(not step.invocation_contracts.bindings for step in generated_steps)

    def compile_module_contracts(steps) -> tuple[ModuleArtifactContract, ...]:
        snapshots = tuple(
            StepSnapshot(
                index=index,
                scope_id=f"contract-parity::functionstep_{index}",
                name=step.name,
                step_type=step.__class__.__name__,
                enabled=bool(step.enabled),
                is_function_step=True,
                func=step.func,
                invocation_contracts=step.invocation_contracts,
                configs=_step_config_universe_for_step(step),
            )
            for index, step in enumerate(steps)
        )
        session = CompilationSession.from_context(
            context=ProcessingContext(
                step_plans={
                    index: CompiledStepPlan(
                        step_index=index,
                        step_name=step.name,
                        step_type=step.__class__.__name__,
                        axis_id="A01",
                    )
                    for index, step in enumerate(steps)
                },
                axis_id="A01",
            ),
            steps=tuple(steps),
            orchestrator=SimpleNamespace(
                pipeline_config=generated.pipeline_config or PipelineConfig(),
            ),
            global_config=GlobalPipelineConfig(),
            step_state_map={index: object() for index in range(len(steps))},
            snapshots=snapshots,
        )
        provider = PipelineInvocationContractProviderAuthority.provider_for_session(
            session,
        )
        resolved_contracts: list[ModuleArtifactContract] = []
        for index, step in enumerate(steps):
            item = next(normalize_function_pattern(step.func).iter_items())
            plan = provider(
                item,
                ArtifactDeclarationStepContext(
                    step_index=index,
                    source_bindings=step.source_bindings,
                ),
            )
            assert plan is not None
            contract = plan.contract.module_artifact_contract
            assert contract is not None
            resolved_contracts.append(contract)
        return tuple(resolved_contracts)

    generated_contracts = compile_module_contracts(generated_steps)

    assert generated_contracts == (expected_contracts[1], expected_contracts[2])
    assert [contract.module_name for contract in generated_contracts] == [
        "IdentifyPrimaryObjects",
        "IdentifySecondaryObjects",
    ]
    assert [spec.name for spec in generated_contracts[1].runtime_artifact_inputs] == [
        "Nuclei",
    ]


def test_pipeline_generator_resolves_object_measurement_function_variants():
    generator = PipelineGenerator()
    modules = [
        _identify_primary(),
        _module(
            2,
            "MeasureTexture",
            {
                "Select images to measure": "OrigBlue",
                "Select objects to measure": "Nuclei",
                "Enter how many gray levels to measure the texture at": "256",
                "Measure images or objects?": "Objects",
                "Texture scale to measure": "3",
            },
        ),
        _module(
            3,
            "MeasureColocalization",
            {
                "Select images to measure": "OrigBlue, OrigGreen",
                "Select where to measure correlation": "Both",
                "Select objects to measure": "Nuclei",
                "Set threshold as percentage of maximum intensity for the images": "15.0",
            },
        ),
        _module(
            4,
            "MeasureGranularity",
            {
                "Select images to measure": "OrigBlue",
                "Select objects to measure": "Nuclei",
                "Subsampling factor for granularity measurements": "0.25",
            },
        ),
    ]
    generated = generator.generate_from_registry(
        pipeline_name="cp_measurement_variants",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    assert "measure_texture_objects," in generated.code
    assert "measure_colocalization_objects," in generated.code
    assert "measure_granularity_objects," in generated.code
    assert "CellProfilerMeasurementTargetScope.BOTH" in generated.code


def test_pipeline_generator_uses_module_class_required_variable_components():
    assert TrackObjectsModule.required_variable_components == (
        VariableComponents.TIMEPOINT,
    )
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="track_objects_components",
        source_cppipe=Path("source.cppipe"),
        modules=[
            _identify_primary(),
            _module(
                2,
                "TrackObjects",
                {
                    "Choose a tracking method": "Overlap",
                    "Select the objects to track": "Nuclei",
                    "Maximum pixel distance to consider matches": "50",
                },
            ),
        ],
    )
    assert "track_objects," in generated.code
    assert (
        "variable_components=[VariableComponents.SITE, VariableComponents.TIMEPOINT],"
        in generated.code
    )
    assert generated.runtime_module_contracts_by_module_num[
        2
    ].required_variable_components == (VariableComponents.TIMEPOINT,)


def test_module_processing_components_validate_required_variable_components():
    processing_components = ModuleProcessingComponents((), None)
    with pytest.raises(ValueError, match="TrackObjects requires variable_components"):
        processing_components.validate_required_variable_components(
            (VariableComponents.TIMEPOINT,), module_name="TrackObjects"
        )


def test_pipeline_generator_canonicalizes_legacy_measure_correlation_module():
    generator = PipelineGenerator()
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "MeasureCorrelation",
            [
                ("Select an image to measure", "OrigBlue"),
                ("Select an image to measure", "OrigGreen"),
                ("Select where to measure correlation", "Within objects"),
                ("Select an object to measure", "Nuclei"),
                (
                    "Set threshold as percentage of maximum intensity for the images",
                    "15.0",
                ),
            ],
        ),
    ]
    generated = generator.generate_from_registry(
        pipeline_name="legacy_measure_correlation",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    contract = generated.artifact_contracts[1]
    assert generator.has_module("MeasureCorrelation")
    assert contract.module_name == "MeasureColocalization"
    assert [spec.name for spec in contract.inputs] == [
        "OrigBlue",
        "OrigGreen",
        "Nuclei",
    ]
    assert "measure_colocalization_objects," in generated.code
    assert (
        generated.runtime_module_contracts_by_module_num[2].module_name
        == "MeasureColocalization"
    )


def test_measure_image_area_occupied_alias_compiles_binary_contract():
    module = _module_with_records(
        1,
        "MeasureImageAreaOccupied",
        [
            (
                "Measure the area occupied in a binary image, or in objects?",
                "Binary Image",
            ),
            ("Select objects to measure", "None"),
            ("Retain a binary image of the object regions?", "Yes"),
            ("Name the output binary image", "Foreground"),
            ("Select a binary image to measure", "DNA"),
        ],
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="area_occupied_binary",
        source_cppipe=Path("source.pipeline"),
        modules=[module],
    )
    assert PipelineGenerator().has_module("MeasureImageAreaOccupied")
    assert [spec.name for spec in contract.inputs] == ["DNA"]
    assert [spec.artifact_type for spec in contract.outputs] == [
        ImageArtifactType,
        MeasurementsArtifactType,
    ]
    assert [spec.name for spec in contract.outputs] == [
        "Foreground",
        "MeasureImageAreaOccupied_1_measurements",
    ]
    assert "measure_image_area_occupied," in generated.code


def test_measure_image_area_occupied_resolves_object_variant():
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "MeasureImageAreaOccupied",
            [
                (
                    "Measure the area occupied in a binary image, or in objects?",
                    "Objects",
                ),
                ("Select objects to measure", "Nuclei"),
                ("Retain a binary image of the object regions?", "Yes"),
                ("Name the output binary image", "OccupiedNuclei"),
                ("Select a binary image to measure", "None"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="area_occupied_objects",
        source_cppipe=Path("source.pipeline"),
        modules=modules,
    )
    assert [spec.name for spec in contract.inputs] == ["Nuclei"]
    assert [spec.artifact_type for spec in contract.outputs] == [
        ImageArtifactType,
        MeasurementsArtifactType,
    ]
    assert [spec.name for spec in contract.outputs] == [
        "OccupiedNuclei",
        "MeasureImageAreaOccupied_2_measurements",
    ]
    assert "measure_image_area_occupied," in generated.code


def test_measure_image_area_occupied_compiles_mixed_rows():
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "MeasureImageAreaOccupied",
            [
                (
                    "Measure the area occupied in a binary image, or in objects?",
                    "Binary Image",
                ),
                ("Select objects to measure", "None"),
                ("Retain a binary image of the object regions?", "No"),
                ("Name the output binary image", "Ignored"),
                ("Select a binary image to measure", "DNA"),
                (
                    "Measure the area occupied in a binary image, or in objects?",
                    "Objects",
                ),
                ("Select objects to measure", "Nuclei"),
                ("Retain a binary image of the object regions?", "Yes"),
                ("Name the output binary image", "OccupiedNuclei"),
                ("Select a binary image to measure", "None"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="area_occupied_mixed",
        source_cppipe=Path("source.pipeline"),
        modules=modules,
    )
    assert [spec.name for spec in contract.inputs] == ["DNA", "Nuclei"]
    assert [spec.name for spec in contract.outputs] == [
        "OccupiedNuclei",
        "MeasureImageAreaOccupied_2_measurements",
    ]
    assert "'operand_choices': ('binary_image', 'objects')" in generated.code
    assert "'input_names':" not in generated.code
    assert "AreaOccupiedInvocationOptions" not in generated.code


def test_align_compiles_two_image_contract():
    module = _module(
        1,
        "Align",
        {
            "Select the alignment method": "Mutual Information",
            "Crop mode": "Keep size",
            "Select the first input image": "Image1",
            "Name the first output image": "AlignedImage1",
            "Select the second input image": "Image2",
            "Name the second output image": "AlignedImage2",
        },
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="align", source_cppipe=Path("source.pipeline"), modules=[module]
    )
    assert [spec.name for spec in contract.inputs] == ["Image1", "Image2"]
    assert [spec.name for spec in contract.outputs] == [
        "AlignedImage1",
        "AlignedImage2",
        "Align_1_measurements",
    ]
    assert "align," in generated.code
    assert "'crop_mode': AlignModule.CropMode.KEEP_SIZE" in generated.code


def test_crop_contract_marks_mask_sidecar_with_typed_role():
    module = _module(
        1,
        "Crop",
        {
            "Select the input image": "OrigBlue",
            "Name the output image": "CropBlue",
            "Select the cropping shape": "Rectangle",
            "Crop mode": "Edges",
            "Left and right rectangle positions": "1,10",
            "Top and bottom rectangle positions": "2,20",
        },
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="crop", source_cppipe=Path("source.pipeline"), modules=[module]
    )
    crop_mask_spec = contract.outputs[1]
    assert crop_mask_spec.name == "CropBlue__crop_mask"
    assert crop_mask_spec.sidecar_role is ArtifactSidecarRole.CROP_MASK
    assert (
        generated.runtime_module_contracts_by_module_num[1].outputs[1].sidecar_role
        is ArtifactSidecarRole.CROP_MASK
    )


def test_align_compiles_additional_similar_image_contract():
    module = _module_with_records(
        1,
        "Align",
        [
            ("Select the alignment method", "Mutual Information"),
            ("Crop mode", "Keep size"),
            ("Select the first input image", "Image1"),
            ("Name the first output image", "AlignedImage1"),
            ("Select the second input image", "Image2"),
            ("Name the second output image", "AlignedImage2"),
            ("Select the additional image", "CombinedImage"),
            ("Name the output image", "AlignedCombined"),
            ("Select how the alignment is to be applied", "Similarly"),
        ],
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="align", source_cppipe=Path("source.pipeline"), modules=[module]
    )
    assert [spec.name for spec in contract.inputs] == [
        "Image1",
        "Image2",
        "CombinedImage",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "AlignedImage1",
        "AlignedImage2",
        "AlignedCombined",
        "Align_1_measurements",
    ]
    assert (
        "'additional_alignment_modes': (AlignModule.AdditionalMode.SIMILARLY,)"
        in generated.code
    )


def test_unmix_colors_compiles_escaped_multi_output_rows():
    module = _module_with_records(
        1,
        "UnmixColors",
        [
            ("Stain count", "3"),
            ("Color image\\x3A", "Color"),
            ("Image name\\x3A", "Hematoxylin"),
            ("Stain", "Hematoxylin"),
            ("Red absorbance\\x3A", "0.5"),
            ("Green absorbance\\x3A", "0.5"),
            ("Blue absorbance\\x3A", "0.5"),
            ("Image name\\x3A", "Eosin"),
            ("Stain", "Eosin"),
            ("Red absorbance\\x3A", "0.5"),
            ("Green absorbance\\x3A", "0.5"),
            ("Blue absorbance\\x3A", "0.5"),
            ("Image name\\x3A", "CustomStain"),
            ("Stain", "Custom"),
            ("Red absorbance\\x3A", "0.1"),
            ("Green absorbance\\x3A", "0.2"),
            ("Blue absorbance\\x3A", "0.3"),
        ],
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="unmix_colors",
        source_cppipe=Path("source.pipeline"),
        modules=[module],
    )
    assert [spec.name for spec in contract.inputs] == ["Color"]
    assert [spec.name for spec in contract.outputs] == [
        "Hematoxylin",
        "Eosin",
        "CustomStain",
    ]
    assert "'stain_names': ('Hematoxylin', 'Eosin', 'Custom')" in generated.code
    assert (
        "'custom_absorbances': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.1, 0.2, 0.3))"
        in generated.code
    )


def test_cppipe_parser_supports_unindented_legacy_pipeline_settings(tmp_path: Path):
    pipeline_path = tmp_path / "legacy.pipeline"
    pipeline_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "",
                "MeasureColocalization:[module_num:1|enabled:True]",
                "Hidden:2",
                "Select an image to measure:DNA",
                "Select an image to measure:Cytoplasm",
            )
        )
    )
    modules = CPPipeParser().parse(pipeline_path)
    assert modules[0].get_setting_values("Select an image to measure") == (
        "DNA",
        "Cytoplasm",
    )


def test_pipeline_generator_uses_image_variant_without_object_measurement_inputs():
    generator = PipelineGenerator()
    modules = [
        _module(
            1,
            "MeasureColocalization",
            {
                "Select images to measure": "OrigBlue, OrigGreen",
                "Select where to measure correlation": "Across entire image",
                "Select objects to measure": "",
            },
        )
    ]
    generated = generator.generate_from_registry(
        pipeline_name="image_colocalization",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    contract = generated.artifact_contracts[0]
    assert [spec.name for spec in contract.inputs] == ["OrigBlue", "OrigGreen"]
    assert "measure_colocalization," in generated.code


def test_pipeline_generator_preserves_default_materialization_for_tabular_outputs():
    generator = PipelineGenerator()
    modules = [
        _identify_primary(),
        _module(
            2,
            "MeasureImageIntensity",
            {"Select images to measure": "OrigBlue", "Select input object sets": ""},
        ),
    ]
    generated = generator.generate_from_registry(
        pipeline_name="cp_materialization_defaults",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    contracts_by_module = generated.runtime_module_contracts_by_module_num
    assert contracts_by_module[1].outputs[0].materialization is None
    assert contracts_by_module[2].outputs[0].materialization is None


def test_pipeline_generator_binds_correct_illumination_settings_as_literals():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_illumination_settings",
        source_cppipe=Path("source.cppipe"),
        modules=[
            _module(
                1,
                "CorrectIlluminationCalculate",
                {
                    "Select the input image": "CropGray",
                    "Name the output image": "Illumgray",
                    "Select how the illumination function is calculated": "Background",
                    "Block size": "40",
                    "Rescale the illumination function?": "No",
                    "Calculate function for each image individually, or based on all images?": "All: First cycle",
                    "Smoothing method": "Convex Hull",
                    "Method to calculate smoothing filter size": "Manually",
                    "Smoothing filter size": "10",
                    "Automatically calculate spline parameters?": "Yes",
                },
            ),
            _module(
                2,
                "CorrectIlluminationApply",
                {
                    "Select the input image": "CropGray",
                    "Name the output image": "CorrectedGray",
                    "Select the illumination function": "Illumgray",
                    "Select how the illumination function is applied": "Subtract",
                    "Set output image values less than 0 equal to 0?": "No",
                    "Set output image values greater than 1 equal to 1?": "Yes",
                },
            ),
        ],
    )
    assert "'intensity_choice': IntensityChoice.BACKGROUND" in generated.code
    assert "'block_size': 40" in generated.code
    assert "'rescale_option': RescaleOption.NO" in generated.code
    assert "'calculation_scope': CalculationScope.ALL_FIRST_CYCLE" in generated.code
    assert "'smoothing_method': SmoothingMethod.CONVEX_HULL" in generated.code
    assert "'filter_size_method': FilterSizeMethod.MANUALLY" in generated.code
    assert "'manual_filter_size': 10" in generated.code
    assert generated.pipeline_config is not None
    assert generated.pipeline_config.processing_config.variable_components == [
        VariableComponents.SITE
    ]
    assert generated.pipeline_config.processing_config.group_by is GroupBy.CHANNEL
    assert "'method': IlluminationCorrectionMethod.SUBTRACT" in generated.code
    assert "'truncate_low': False" in generated.code
    assert "'truncate_high': True" in generated.code


def test_pipeline_generator_coalesces_correct_illumination_apply_repeated_pairs():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"^(?P<Plate>.*)_s(?P<Site>[0-9])_ch(?P<ChannelNumber>[0-9])",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "2"),
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "OrigStain1"),
                ("Match metadata", "[]"),
                ("Image set matching method", "Order"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "1")'),
                ("Assign a name to", "Images matching rules"),
                ("Name to assign these images", "OrigStain2"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "2")'),
            ],
        ),
    ]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_illumination_apply_repeated",
        source_cppipe=Path("source.cppipe"),
        modules=[
            _module_with_records(
                3,
                "CorrectIlluminationCalculate",
                [
                    ("Select the input image", "OrigStain1"),
                    ("Name the output image", "IllumStain1"),
                    ("Select how the illumination function is calculated", "Regular"),
                    (
                        "Calculate function for each image individually, or based on all images?",
                        "Each",
                    ),
                    ("Smoothing method", "Fit Polynomial"),
                    ("Method to calculate smoothing filter size", "Automatic"),
                    ("Rescale the illumination function?", "Yes"),
                ],
            ),
            _module_with_records(
                4,
                "CorrectIlluminationCalculate",
                [
                    ("Select the input image", "OrigStain2"),
                    ("Name the output image", "IllumStain2"),
                    ("Select how the illumination function is calculated", "Regular"),
                    (
                        "Calculate function for each image individually, or based on all images?",
                        "Each",
                    ),
                    ("Smoothing method", "Fit Polynomial"),
                    ("Method to calculate smoothing filter size", "Automatic"),
                    ("Rescale the illumination function?", "Yes"),
                ],
            ),
            _module_with_records(
                5,
                "CorrectIlluminationApply",
                [
                    ("Select the input image", "OrigStain1"),
                    ("Name the output image", "CorrectedStain1"),
                    ("Select the illumination function", "IllumStain1"),
                    ("Select how the illumination function is applied", "Divide"),
                    ("Select the input image", "OrigStain2"),
                    ("Name the output image", "CorrectedStain2"),
                    ("Select the illumination function", "IllumStain2"),
                    ("Select how the illumination function is applied", "Divide"),
                    ("Set output image values less than 0 equal to 0?", "Yes"),
                    ("Set output image values greater than 1 equal to 1?", "Yes"),
                ],
            )
        ],
        skipped_modules=setup_modules,
    )

    assert generated.code.count('name="CorrectIlluminationApply"') == 1
    assert "func={" not in generated.code
    assert "'1': (correct_illumination_apply" not in generated.code
    assert "'2': (correct_illumination_apply" not in generated.code
    assert "func=(correct_illumination_apply" in generated.code
    assert "select_the_input_image" not in generated.code
    assert "select_the_illumination_function" not in generated.code
    assert "name_the_output_image" not in generated.code
    assert "'method': IlluminationCorrectionMethod.DIVIDE" in generated.code
    assert [
        [spec.name for spec in contract.outputs]
        for contract in generated.artifact_contracts
        if contract.module_name == "CorrectIlluminationApply"
    ] == [["CorrectedStain1"], ["CorrectedStain2"]]


def test_pipeline_generator_keeps_grouped_noncanonical_illumination_apply_outputs():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"^(?P<Plate>.*)_ch(?P<ChannelNumber>[0-9])",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "4"),
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "OrigProtein"),
                ("Match metadata", "[]"),
                ("Image set matching method", "Order"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "1")'),
                ("Assign a name to", "Images matching rules"),
                ("Name to assign these images", "IllumProtein"),
                ("Select the image type", "Illumination function"),
                (
                    "Select the rule criteria",
                    'and (file does contain "VitraChannel1ILLUM.npy")',
                ),
                ("Assign a name to", "Images matching rules"),
                ("Name to assign these images", "OrigDNA"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "2")'),
                ("Assign a name to", "Images matching rules"),
                ("Name to assign these images", "IllumDNA"),
                ("Select the image type", "Illumination function"),
                (
                    "Select the rule criteria",
                    'and (file does contain "VitraChannel2ILLUM.npy")',
                ),
            ],
        ),
    ]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_illumination_apply_noncanonical",
        source_cppipe=Path("source.cppipe"),
        modules=[
            _module_with_records(
                3,
                "CorrectIlluminationApply",
                [
                    ("Select the input image", "OrigProtein"),
                    ("Name the output image", "CorrProtein"),
                    ("Select the illumination function", "IllumProtein"),
                    ("Select how the illumination function is applied", "Divide"),
                    ("Select the input image", "OrigDNA"),
                    ("Name the output image", "CorrDNA"),
                    ("Select the illumination function", "IllumDNA"),
                    ("Select how the illumination function is applied", "Divide"),
                    ("Set output image values less than 0 equal to 0?", "Yes"),
                    ("Set output image values greater than 1 equal to 1?", "Yes"),
                ],
            )
        ],
        skipped_modules=setup_modules,
    )

    assert generated.code.count('name="CorrectIlluminationApply"') == 1
    assert "func={" in generated.code
    assert "'1': (correct_illumination_apply" in generated.code
    assert "'2': (correct_illumination_apply" in generated.code
    assert "'name_the_output_image': 'CorrProtein'" in generated.code
    assert "'name_the_output_image': 'CorrDNA'" in generated.code
    assert [
        [spec.name for spec in contract.outputs]
        for contract in generated.artifact_contracts
        if contract.module_name == "CorrectIlluminationApply"
    ] == [["CorrProtein"], ["CorrDNA"]]


def test_cellprofiler_symbol_table_compiles_singular_aliases_and_image_artifacts():
    modules = [
        _identify_primary(),
        _module(
            2,
            "CorrectIlluminationApply",
            {
                "Select the input image": "OrigBlue",
                "Select the illumination function": "IllumBlue",
                "Name the output image": "CorrBlue",
            },
        ),
        _module(
            3,
            "Opening",
            {
                "Select the input image": "CorrBlue",
                "Name the output image": "OpeningBlue",
            },
        ),
        _module(
            4,
            "ConvertObjectsToImage",
            {
                "Select the input objects": "Nuclei",
                "Name the output image": "NucleiImage",
            },
        ),
        _module(
            5,
            "GrayToColor",
            {
                "Select the image to be colored red": "Leave this black",
                "Select the image to be colored green": "OpeningBlue",
                "Select the image to be colored blue": "OrigBlue",
                "Name the output image": "ColorImage",
            },
        ),
        _module(
            6,
            "OverlayOutlines",
            {
                "Select image on which to display outlines": "ColorImage",
                "Select objects to display": "Nuclei",
                "Name the output image": "OverlayImage",
            },
        ),
        _module(
            7,
            "MeasureObjectIntensity",
            {
                "Select an image to measure": "OpeningBlue",
                "Select objects to measure": "Nuclei",
            },
        ),
        _module(
            8,
            "MeasureGranularity",
            {
                "Select an image to measure": "OpeningBlue",
                "Select objects to measure": "Nuclei",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    illumination_contract = table.contracts_by_module_num[2]
    assert tuple(
        (binding.alias for binding in illumination_contract.source_bindings.bindings)
    ) == ("OrigBlue", "IllumBlue")
    assert [spec.name for spec in illumination_contract.outputs] == ["CorrBlue"]
    gray_to_color_contract = table.contracts_by_module_num[5]
    assert [spec.name for spec in gray_to_color_contract.inputs] == [
        "OpeningBlue",
        "OrigBlue",
    ]
    assert [spec.name for spec in gray_to_color_contract.outputs] == ["ColorImage"]
    overlay_contract = table.contracts_by_module_num[6]
    assert [spec.name for spec in overlay_contract.runtime_artifact_inputs] == [
        "ColorImage",
        "Nuclei",
    ]
    measure_intensity_contract = table.contracts_by_module_num[7]
    assert measure_intensity_contract.source_bindings.is_empty
    assert [
        spec.name for spec in measure_intensity_contract.runtime_artifact_inputs
    ] == ["OpeningBlue", "Nuclei"]
    granularity_contract = table.contracts_by_module_num[8]
    assert [spec.name for spec in granularity_contract.runtime_artifact_inputs] == [
        "OpeningBlue",
        "Nuclei",
    ]
    assert granularity_contract.outputs[0].artifact_type is MeasurementsArtifactType


def test_correct_illumination_apply_contract_preserves_repeated_pairs() -> None:
    modules = [
        _module_with_records(
            1,
            "CorrectIlluminationApply",
            [
                ("Select the input image", "OrigRed"),
                ("Name the output image", "CorrRed"),
                ("Select the illumination function", "IllumRed"),
                ("Select how the illumination function is applied", "Subtract"),
                ("Select the input image", "OrigBlue"),
                ("Name the output image", "CorrBlue"),
                ("Select the illumination function", "IllumBlue"),
                ("Select how the illumination function is applied", "Subtract"),
                ("Select the input image", "OrigGreen"),
                ("Name the output image", "CorrGreen"),
                ("Select the illumination function", "IllumGreen"),
                ("Select how the illumination function is applied", "Subtract"),
            ],
        )
    ]
    contract = CellProfilerSymbolTable.compile(modules).contracts_by_module_num[1]
    assert [spec.name for spec in contract.inputs] == [
        "OrigRed",
        "IllumRed",
        "OrigBlue",
        "IllumBlue",
        "OrigGreen",
        "IllumGreen",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "CorrRed",
        "CorrBlue",
        "CorrGreen",
    ]


def test_correct_illumination_apply_contract_preserves_same_artifact_roles() -> None:
    modules = [
        _module_with_records(
            1,
            "CorrectIlluminationCalculate",
            [
                ("Select the input image", "OrigGreen"),
                ("Name the output image", "IllumGreen"),
            ],
        ),
        _module_with_records(
            2,
            "CorrectIlluminationApply",
            [
                ("Select the input image", "IllumGreen"),
                ("Name the output image", "CorrGreen"),
                ("Select the illumination function", "IllumGreen"),
                ("Select how the illumination function is applied", "Divide"),
            ],
        ),
    ]
    contract = CellProfilerSymbolTable.compile(modules).contracts_by_module_num[2]
    assert [spec.name for spec in contract.inputs] == ["IllumGreen", "IllumGreen"]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == ["IllumGreen"]
    assert [spec.name for spec in contract.outputs] == ["CorrGreen"]


def test_overlay_outlines_accepts_image_outline_rows() -> None:
    module = _module_with_records(
        1,
        "OverlayOutlines",
        [
            ("Display outlines on a blank image?", "No"),
            ("Select image on which to display outlines", "DNA"),
            ("Name the output image\\x3A", "Overlay"),
            ("Select outline display mode\\x3A", "Color"),
            ("Select method to determine brightness of outlines\\x3A", "Max of image"),
            ("Line width\\x3A", "1.5"),
            ("Select outlines to display\\x3A", "PrimaryOutlines"),
            ("Select outline color\\x3A", "Red"),
            ("Select outlines to display\\x3A", "SecondaryOutlines"),
            ("Select outline color\\x3A", "Green"),
        ],
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    assert [(spec.name, spec.artifact_type) for spec in contract.inputs] == [
        ("DNA", ImageArtifactType),
        ("PrimaryOutlines", ImageArtifactType),
        ("SecondaryOutlines", ImageArtifactType),
    ]
    assert contract.runtime_artifact_inputs == ()
    assert [spec.name for spec in contract.outputs] == ["Overlay"]


def test_overlay_outlines_accepts_mixed_image_and_object_rows() -> None:
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "OverlayOutlines",
            [
                ("Display outlines on a blank image?", "No"),
                ("Select image on which to display outlines", "DNA"),
                ("Name the output image", "Overlay"),
                ("Outline display mode", "Color"),
                ("Select method to determine brightness of outlines", "Max of image"),
                ("Width of outlines", "1.5"),
                ("Select outlines to display", "PrimaryOutlines"),
                ("Select outline color", "Red"),
                ("Load outlines from an image or objects?", "Image"),
                ("Select objects to display", "Nuclei"),
                ("Select outlines to display\\x3A", "SecondaryOutlines"),
                ("Select outline color\\x3A", "Green"),
                ("Load outlines from an image or objects?", "Objects"),
                ("Select objects to display", "Nuclei"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    assert [(spec.name, spec.artifact_type) for spec in contract.inputs] == [
        ("DNA", ImageArtifactType),
        ("PrimaryOutlines", ImageArtifactType),
        ("Nuclei", ObjectLabelsArtifactType),
    ]
    assert [
        (spec.name, spec.artifact_type) for spec in contract.runtime_artifact_inputs
    ] == [("Nuclei", ObjectLabelsArtifactType)]


def test_overlay_outlines_preserves_color_before_object_rows() -> None:
    module = _module_with_records(
        1,
        "OverlayOutlines",
        [
            ("Display outlines on a blank image?", "No"),
            ("Select image on which to display outlines", "DNA"),
            ("Name the output image", "OrigOverlay"),
            ("Outline display mode", "Color"),
            ("Select method to determine brightness of outlines", "Max of image"),
            ("How to outline", "Thick"),
            ("Select outline color", "#0080FF"),
            ("Select objects to display", "Cells"),
            ("Select outline color", "blue"),
            ("Select objects to display", "Nuclei"),
            ("Select outline color", "yellow"),
            ("Select objects to display", "PH3"),
        ],
    )
    assert OverlayOutlinesModule.settings_source(module)["outline_colors"] == (
        "#0080FF",
        "blue",
        "yellow",
    )


def test_color_to_gray_combine_contract_ignores_dormant_split_outputs() -> None:
    module = _module_with_records(
        1,
        "ColorToGray",
        [
            ("Select the input image", "OrigColor"),
            ("Conversion method", "Combine"),
            ("Image type", "RGB"),
            ("Name the output image", "OrigGray"),
            ("Relative weight of the red channel", "1.0"),
            ("Relative weight of the green channel", "1.0"),
            ("Relative weight of the blue channel", "1.0"),
            ("Convert red to gray?", "Yes"),
            ("Name the output image", "OrigRed"),
            ("Convert green to gray?", "Yes"),
            ("Name the output image", "OrigGreen"),
            ("Convert blue to gray?", "Yes"),
            ("Name the output image", "OrigBlue"),
        ],
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    assert [spec.name for spec in contract.inputs] == ["OrigColor"]
    assert [spec.name for spec in contract.outputs] == ["OrigGray"]


def test_color_to_gray_split_contract_uses_enabled_rgb_outputs() -> None:
    module = _module_with_records(
        1,
        "ColorToGray",
        [
            ("Select the input image", "OrigColor"),
            ("Conversion method", "Split"),
            ("Image type", "RGB"),
            ("Name the output image", "OrigGray"),
            ("Relative weight of the red channel", "1.0"),
            ("Relative weight of the green channel", "1.0"),
            ("Relative weight of the blue channel", "1.0"),
            ("Convert red to gray?", "Yes"),
            ("Name the output image", "OrigRed"),
            ("Convert green to gray?", "No"),
            ("Name the output image", "OrigGreen"),
            ("Convert blue to gray?", "Yes"),
            ("Name the output image", "OrigBlue"),
            ("Convert hue to gray?", "Yes"),
            ("Name the output image", "OrigHue"),
            ("Convert saturation to gray?", "Yes"),
            ("Name the output image", "OrigSaturation"),
            ("Convert value to gray?", "Yes"),
            ("Name the output image", "OrigValue"),
        ],
    )
    table = CellProfilerSymbolTable.compile([module])
    contract = table.contracts_by_module_num[1]
    assert [spec.name for spec in contract.outputs] == ["OrigRed", "OrigBlue"]


def test_pipeline_generator_keeps_color_to_gray_runtime_settings_public() -> None:
    module = _module_with_records(
        1,
        "ColorToGray",
        [
            ("Select the input image", "OrigColor"),
            ("Conversion method", "Split"),
            ("Image type", "RGB"),
            ("Name the output image", "OrigGray"),
            ("Relative weight of the red channel", "1.0"),
            ("Relative weight of the green channel", "1.0"),
            ("Relative weight of the blue channel", "1.0"),
            ("Convert red to gray?", "Yes"),
            ("Name the output image", "OrigRed"),
            ("Convert green to gray?", "No"),
            ("Name the output image", "OrigGreen"),
            ("Convert blue to gray?", "Yes"),
            ("Name the output image", "OrigBlue"),
        ],
    )
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="color_to_gray",
        source_cppipe=Path("source.cppipe"),
        modules=[module],
    )

    assert "'mode': ColorToGrayMode.SPLIT" in generated.code
    assert "'image_type': ImageChannelType.RGB" in generated.code
    assert "'channel_indices': (0, 2)" in generated.code
    assert "'contributions': (1.0, 1.0)" in generated.code
    assert "name_the_output_image" not in generated.code
    assert [spec.name for spec in generated.artifact_contracts[0].outputs] == [
        "OrigRed",
        "OrigBlue",
    ]


def test_cellprofiler_symbol_table_infers_common_image_transform_contract():
    modules = [
        _module(
            1,
            "CorrectIlluminationCalculate",
            {
                "Select the input image": "OrigBlue",
                "Name the output image": "IllumBlue",
            },
        )
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[1]
    assert [spec.name for spec in contract.inputs] == ["OrigBlue"]
    assert [spec.artifact_type for spec in contract.inputs] == [ImageArtifactType]
    assert tuple((binding.alias for binding in contract.source_bindings.bindings)) == (
        "OrigBlue",
    )
    assert [spec.name for spec in contract.outputs] == ["IllumBlue"]
    assert [spec.artifact_type for spec in contract.outputs] == [ImageArtifactType]


def test_cellprofiler_symbol_table_infers_common_object_transform_contract():
    modules = [
        _identify_primary(),
        _module(
            2,
            "DilateObjects",
            {
                "Select the input objects": "Nuclei",
                "Name the output objects": "DilatedNuclei",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == ["Nuclei"]
    assert [spec.artifact_type for spec in contract.runtime_artifact_inputs] == [
        ObjectLabelsArtifactType
    ]
    assert [spec.name for spec in contract.outputs] == [
        "DilateObjects_2_measurements",
        "DilatedNuclei",
    ]
    assert [spec.artifact_type for spec in contract.outputs] == [
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    ]


def test_cellprofiler_symbol_table_infers_special_output_only_contract():
    table = CellProfilerSymbolTable.compile(
        [_module(1, "CalculateMath", {"Operation": "Add"})]
    )
    contract = table.contracts_by_module_num[1]
    assert contract.inputs == ()
    assert [spec.name for spec in contract.outputs] == ["CalculateMath_1_measurements"]
    assert [spec.artifact_type for spec in contract.outputs] == [
        MeasurementsArtifactType
    ]


def test_generated_inputless_measurement_only_step_disables_default_grouping():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="calculate_math_measurement_only",
        source_cppipe=Path("source.pipeline"),
        modules=[
            _module(
                1,
                "CalculateMath",
                {
                    "Name the output measurement": "Ratio",
                    "Operation": "Divide",
                    "Select the numerator measurement": "AreaOccupied_Nuclei",
                    "Select the denominator measurement": "AreaOccupied_Cells",
                    "Select the numerator objects": "None",
                    "Select the denominator objects": "None",
                },
            )
        ],
    )
    assert 'name="CalculateMath"' in generated.code
    assert "'output_name':" not in generated.code
    assert "'operand1_object_name':" not in generated.code
    assert "'operand2_object_name':" not in generated.code
    assert "CalculateMathInvocationOptions(output_name='Ratio'" in generated.code
    assert "variable_components=[VariableComponents.SITE]" in generated.code
    assert "group_by=GroupBy.CHANNEL" in generated.code


def test_cellprofiler_symbol_table_infers_mask_objects_contract():
    modules = [
        _identify_primary(),
        _module(
            2,
            "MaskObjects",
            {
                "Select the input objects": "Nuclei",
                "Select the masking image": "OrigBlue",
                "Name the output objects": "MaskedNuclei",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in contract.inputs] == ["Nuclei", "OrigBlue"]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == ["Nuclei"]
    assert tuple((binding.alias for binding in contract.source_bindings.bindings)) == (
        "OrigBlue",
    )
    assert [spec.name for spec in contract.outputs] == [
        "MaskObjects_2_measurements",
        "Nuclei_MaskedNuclei_relationships",
        "MaskedNuclei",
    ]
    assert [spec.artifact_type for spec in contract.outputs] == [
        MeasurementsArtifactType,
        RelationshipsArtifactType,
        ObjectLabelsArtifactType,
    ]


def test_cellprofiler_symbol_table_rejects_unknown_generic_object_input():
    with pytest.raises(
        ValueError,
        match="Module FilterObjects\\(1\\) references unknown object_labels symbol 'Nuclei'",
    ):
        CellProfilerSymbolTable.compile(
            [
                _module(
                    1,
                    "FilterObjects",
                    {
                        "Select the input objects": "Nuclei",
                        "Name the output objects": "FilteredNuclei",
                        "Name the output image": "FilteredNucleiImage",
                    },
                )
            ]
        )


def test_cellprofiler_symbol_table_reads_gray_to_color_stack_inputs_from_records():
    modules = [
        _module_with_records(
            1,
            "GrayToColor",
            [
                ("Select a color scheme", "Stack"),
                ("Image name", "OrigBlue"),
                ("Color", "#0000ff"),
                ("Weight", "1.0"),
                ("Image name", "OrigGreen"),
                ("Color", "#00ff00"),
                ("Weight", "2.0"),
                ("Name the output image", "StackedColor"),
            ],
        )
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[1]
    assert [spec.name for spec in contract.inputs] == ["OrigBlue", "OrigGreen"]
    assert tuple((binding.alias for binding in contract.source_bindings.bindings)) == (
        "OrigBlue",
        "OrigGreen",
    )
    assert [spec.name for spec in contract.outputs] == ["StackedColor"]


def test_classifyobjects_alias_compiles_variant_contract_and_settings():
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "ClassifyObjects",
            [
                (
                    "Make each classification decision on how many measurements?",
                    "Single measurement",
                ),
                ("Select the object to be classified", "Nuclei"),
                ("Select the measurement to classify by", "Math_Ratio"),
                ("Select bin spacing", "Custom-defined bins"),
                (
                    "Enter the custom thresholds separating the values between bins",
                    "0.25,0.75",
                ),
                ("Give each bin a name?", "Yes"),
                ("Enter the bin names separated by commas", "Low,High"),
                ("Retain an image of the classified objects?", "No"),
                ("Name the output image", "IgnoredClassifiedImage"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="classify", source_cppipe=Path("source.cppipe"), modules=modules
    )
    assert PipelineGenerator().has_module("ClassifyObjects")
    assert contract.module_name == "ClassifyObjectsSingleMeasurement"
    assert [spec.name for spec in contract.inputs] == [
        "Nuclei",
        "IdentifyPrimaryObjects_1_measurements",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "ClassifyObjects_2_measurements"
    ]
    assert "classify_objects_single_measurement," in generated.code
    assert "'measurement_feature': 'Math_Ratio'" in generated.code
    assert "'bin_choice': ClassificationBinChoice.CUSTOM" in generated.code
    assert "'custom_thresholds': '0.25,0.75'" in generated.code


def test_feature_addressed_measurement_consumers_require_prior_measurement_outputs():
    from openhcs.processing.backends.cellprofiler.object_filtering import (
        FilterObjectsInputPolicy,
    )

    cases = (
        (
            _module(
                2,
                "CalculateMath",
                {
                    "Name the output measurement": "Ratio",
                    "Operation": "Add",
                    "Select the numerator measurement": "AreaShape_Area",
                    "Select the denominator measurement": "AreaShape_Perimeter",
                    "Select the numerator objects": "None",
                    "Select the denominator objects": "None",
                },
            ),
            ("IdentifyPrimaryObjects_1_measurements",),
        ),
        (
            _module(
                2,
                "FilterObjects",
                {
                    "Select the object to filter": "Nuclei",
                    "Name the output objects": "FilteredNuclei",
                    "Filter using classifier rules or measurements?": "Measurements",
                    "Select the filtering method": "Limits",
                    "Select the measurement to filter by": "AreaShape_Area",
                    "Filter using a minimum measurement value?": "Yes",
                    "Minimum value": "1",
                    "Filter using a maximum measurement value?": "No",
                },
            ),
            ("Nuclei", "IdentifyPrimaryObjects_1_measurements"),
        ),
        (
            _module(
                2,
                "DisplayDataOnImage",
                {
                    "Select the image on which to display the measurements": "OrigBlue",
                    "Select the input objects": "Nuclei",
                    "Name the output image that has the measurements displayed": "Annotated",
                    "Display object or image measurements?": "Object",
                    "Measurement to display": "AreaShape_Area",
                },
            ),
            ("Nuclei", "IdentifyPrimaryObjects_1_measurements"),
        ),
        (
            _module_with_records(
                2,
                "ClassifyObjects",
                [
                    (
                        "Make each classification decision on how many measurements?",
                        "Single measurement",
                    ),
                    ("Select the object to be classified", "Nuclei"),
                    ("Select the measurement to classify by", "AreaShape_Area"),
                    ("Retain an image of the classified objects?", "No"),
                ],
            ),
            ("Nuclei", "IdentifyPrimaryObjects_1_measurements"),
        ),
    )

    for module, expected_runtime_input_names in cases:
        table = CellProfilerSymbolTable.compile([_identify_primary(), module])
        contract = table.contracts_by_module_num[2]
        assert tuple(
            spec.name for spec in contract.runtime_artifact_inputs
        ) == expected_runtime_input_names
    assert (
        MeasurementsArtifactType
        in FilterObjectsInputPolicy.supported_non_object_input_kinds
    )


def test_classifyobjects_repeated_single_measurement_rows_are_bound_as_rules():
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "ClassifyObjects",
            [
                (
                    "Make each classification decision on how many measurements?",
                    "Single measurement",
                ),
                ("Select the object to be classified", "Nuclei"),
                ("Select the measurement to classify by", "AreaShape_Area"),
                ("Select bin spacing", "Custom-defined bins"),
                ("Number of bins", "1"),
                ("Lower threshold", "0"),
                ("Use a bin for objects below the threshold?", "No"),
                ("Upper threshold", "1"),
                ("Use a bin for objects above the threshold?", "No"),
                (
                    "Enter the custom thresholds separating the values between bins",
                    "0,5,75,1300",
                ),
                ("Give each bin a name?", "Yes"),
                ("Enter the bin names separated by commas", "Tiny,Small,Large"),
                ("Retain an image of the classified objects?", "No"),
                ("Name the output image", "Do not use"),
                ("Select the object to be classified", "Nuclei"),
                (
                    "Select the measurement to classify by",
                    "Intensity_MeanIntensity_DNA",
                ),
                ("Select bin spacing", "Custom-defined bins"),
                ("Number of bins", "3"),
                ("Lower threshold", "0"),
                ("Use a bin for objects below the threshold?", "Yes"),
                ("Upper threshold", "1"),
                ("Use a bin for objects above the threshold?", "Yes"),
                (
                    "Enter the custom thresholds separating the values between bins",
                    "0.05",
                ),
                ("Give each bin a name?", "Yes"),
                ("Enter the bin names separated by commas", "White,Red"),
                ("Retain an image of the classified objects?", "No"),
                ("Name the output image", "ClassifiedNuclei"),
            ],
        ),
    ]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="classify_repeated",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    assert "'classification_rules': (" in generated.code
    assert "'measurement_feature': 'AreaShape_Area'" in generated.code
    assert "'measurement_feature': 'Intensity_MeanIntensity_DNA'" in generated.code
    assert "'bin_names': 'Tiny,Small,Large'" in generated.code
    assert "'bin_names': 'White,Red'" in generated.code


def test_grid_variants_do_not_treat_shape_choices_as_object_symbols():
    modules = [
        _identify_primary(),
        _module_with_records(
            2,
            "DefineGrid",
            [
                ("Name the grid", "Grid"),
                ("Number of rows", "8"),
                ("Number of columns", "12"),
                ("Select the method to define the grid", "Automatic"),
                ("Select the previously identified objects", "Nuclei"),
                ("Retain an image of the grid?", "No"),
                ("Name the output image", "IgnoredGridImage"),
                ("Select the image on which to display the grid", "OrigBlue"),
            ],
        ),
        _module_with_records(
            3,
            "IdentifyObjectsInGrid",
            [
                ("Select the defined grid", "Grid"),
                ("Name the objects to be identified", "GridObjects"),
                ("Select object shapes and locations", "Natural Shape and Location"),
                ("Specify the circle diameter automatically?", "Automatic"),
                ("Circle diameter", "20"),
                ("Select the guiding objects", "Nuclei"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    define_grid = table.contracts_by_module_num[2]
    identify_grid = table.contracts_by_module_num[3]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="grid", source_cppipe=Path("source.cppipe"), modules=modules
    )
    assert [spec.name for spec in define_grid.inputs] == ["OrigBlue", "Nuclei"]
    assert [spec.name for spec in define_grid.outputs] == ["Grid"]
    assert [spec.artifact_type for spec in define_grid.outputs] == [
        SpatialGridArtifactType
    ]
    assert [spec.name for spec in identify_grid.inputs] == ["Grid", "Nuclei"]
    assert [spec.artifact_type for spec in identify_grid.inputs] == [
        SpatialGridArtifactType,
        ObjectLabelsArtifactType,
    ]
    assert [spec.name for spec in identify_grid.outputs] == [
        "IdentifyObjectsInGrid_3_measurements",
        "GridObjects",
    ]
    assert "define_grid_automatic," in generated.code
    assert "DefineGridInvocationOptions(" in generated.code
    assert "cycle_scope=DefineGridCycleScope.EACH_CYCLE" in generated.code
    assert "_cellprofiler_grid_cycle_scope" not in generated.code
    assert "identify_objects_in_grid_with_guides," in generated.code
    assert "Natural Shape and Location" not in [
        spec.name for spec in identify_grid.inputs
    ]


def test_define_grid_drops_blank_optional_artifact_symbols():
    modules = [
        _module_with_records(
            1,
            "DefineGrid",
            [
                ("Name the grid", "Grid"),
                ("Number of rows", "8"),
                ("Number of columns", "12"),
                ("Select the method to define the grid", "Manual"),
                ("Select the image on which to display the grid", "None"),
                ("Select the image to display when drawing", "None"),
                ("Select the previously identified objects", "None"),
                ("Retain an image of the grid?", "No"),
                ("Name the output image", "IgnoredGridImage"),
            ],
        )
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[1]
    assert contract.inputs == ()
    assert contract.source_bindings.is_empty
    assert [spec.name for spec in contract.outputs] == ["Grid"]
    assert [spec.artifact_type for spec in contract.outputs] == [
        SpatialGridArtifactType
    ]


def test_mask_and_worm_output_object_names_are_declared_generically():
    modules = [
        _identify_primary(),
        _module(
            2,
            "MaskObjects",
            {
                "Select objects to be masked": "Nuclei",
                "Select the masking object": "Nuclei",
                "Name the masked objects": "MaskedNuclei",
            },
        ),
        _module(
            3,
            "UntangleWorms",
            {
                "Select the input image": "OrigBlue",
                "Name the output overlapping worm objects": "OverlappingWorms",
                "Name the output non-overlapping worm objects": "NonOverlappingWorms",
            },
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    assert [spec.name for spec in table.contracts_by_module_num[2].outputs] == [
        "MaskObjects_2_measurements",
        "Nuclei_MaskedNuclei_relationships",
        "MaskedNuclei",
    ]
    assert [spec.name for spec in table.contracts_by_module_num[3].outputs] == [
        "UntangleWorms_3_measurements",
        "OverlappingWorms",
        "NonOverlappingWorms",
    ]


def test_straightenworms_compiles_repeated_image_outputs_and_settings():
    modules = [
        _module(
            1,
            "UntangleWorms",
            {
                "Select the input image": "WormsBinary",
                "Overlap style": "Both",
                "Name the output overlapping worm objects": "OverlappingWorms",
                "Name the output non-overlapping worm objects": "NonOverlappingWorms",
            },
        ),
        _module_with_records(
            2,
            "StraightenWorms",
            [
                ("Select the input untangled worm objects", "NonOverlappingWorms"),
                ("Name the output straightened worm objects", "StraightenedWorms"),
                ("Worm width", "20"),
                ("Measure intensity distribution?", "Yes"),
                ("Number of transverse segments", "5"),
                ("Number of longitudinal stripes", "1"),
                ("Align worms?", "Top brightest"),
                ("Select an input image to straighten", "mCherry"),
                ("Name the output straightened image", "Straightened_mCherry"),
                ("Select an input image to straighten", "GFP"),
                ("Name the output straightened image", "Straightened_GFP"),
            ],
        ),
    ]
    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_straighten_worms",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    assert [spec.name for spec in contract.inputs] == [
        "NonOverlappingWorms",
        "UntangleWorms_1_measurements",
        "mCherry",
        "GFP",
    ]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == [
        "NonOverlappingWorms",
        "UntangleWorms_1_measurements",
    ]
    assert [spec.name for spec in contract.outputs] == [
        "Straightened_mCherry",
        "Straightened_GFP",
        "StraightenedWorms",
        "StraightenWorms_2_measurements",
    ]
    assert "'worm_width': 20" in generated.code
    assert "'measure_intensity': True" in generated.code
    assert "'number_of_segments': 5" in generated.code
    assert "'number_of_stripes': 1" in generated.code
    assert "'flip_mode': FlipMode.TOP" in generated.code


def test_partition_cppipe_modules_skips_setup_and_export_modules():
    modules = (
        _module(0, "LoadImages", {}),
        _module(1, "Images", {}),
        _module(2, "Metadata", {}),
        _module(3, "NamesAndTypes", {}),
        _module(4, "Groups", {}),
        _identify_primary(5),
        _module(6, "SaveImages", {}),
        _module(7, "ExportToSpreadsheet", {}),
    )
    partition = partition_cppipe_modules(modules)
    assert [module.name for module in partition.infrastructure_modules] == [
        "LoadImages",
        "Images",
        "Metadata",
        "NamesAndTypes",
        "Groups",
        "SaveImages",
        "ExportToSpreadsheet",
    ]
    assert [module.name for module in partition.processing_modules] == [
        "IdentifyPrimaryObjects"
    ]


def test_partition_cppipe_modules_preserves_disabled_modules_outside_execution():
    modules = (
        _module(1, "Images", {}),
        ModuleBlock(
            name="IdentifyPrimaryObjects", module_num=2, enabled=False, settings={}
        ),
        _identify_primary(3),
    )
    partition = partition_cppipe_modules(modules)
    assert [module.module_num for module in partition.infrastructure_modules] == [1]
    assert [module.module_num for module in partition.processing_modules] == [3]
    assert [module.module_num for module in partition.disabled_modules] == [2]


def test_object_transform_measurement_outputs_inherit_declared_source_scope():
    modules = (
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "CropGreen",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
        _module(
            2,
            "IdentifySecondaryObjects",
            {
                "Select the input objects": "Nuclei",
                "Select the input image": "CropGreen",
                "Name the objects to be identified": "Cells",
            },
        ),
        _module(
            3,
            "IdentifyTertiaryObjects",
            {
                "Select the larger identified objects": "Cells",
                "Select the smaller identified objects": "Nuclei",
                "Name the tertiary objects to be identified": "Cytoplasm",
            },
        ),
    )
    table = CellProfilerSymbolTable.compile(modules)

    secondary_output = table.contract_for(modules[1]).outputs[0]
    tertiary_output = table.contract_for(modules[2]).outputs[2]

    assert secondary_output.relations == (
        GroupLineageSourceRelation(
            source=ArtifactSpecRef.input("CropGreen", ImageArtifactType)
        ),
    )
    assert tertiary_output.relations == (
        GroupLineageSourceRelation(
            source=ArtifactSpecRef.input("Cells", ObjectLabelsArtifactType)
        ),
    )
