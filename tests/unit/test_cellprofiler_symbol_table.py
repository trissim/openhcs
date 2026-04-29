from pathlib import Path

import pytest

from benchmark.converter.parser import CPPipeParser, ModuleBlock, ModuleSetting
from benchmark.converter.pipeline_generator import PipelineGenerator
from benchmark.converter.runtime_pipeline import partition_cppipe_modules
from benchmark.converter.symbol_table import (
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
)
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.module_artifact_contract import ModuleArtifactContract


def _module(
    module_num: int,
    name: str,
    settings: dict[str, str],
) -> ModuleBlock:
    return ModuleBlock(name=name, module_num=module_num, settings=settings)


def _module_with_records(
    module_num: int,
    name: str,
    setting_pairs: list[tuple[str, str]],
) -> ModuleBlock:
    records = [ModuleSetting(setting_name, value) for setting_name, value in setting_pairs]
    return ModuleBlock(
        name=name,
        module_num=module_num,
        settings={setting.name: setting.value for setting in records},
        setting_records=records,
    )


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
            {
                "Select images to measure": "OrigBlue",
                "Select input object sets": "",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)

    assert table.symbol_for("OrigBlue", CellProfilerSymbolKind.IMAGE).kind is (
        CellProfilerSymbolKind.IMAGE
    )
    assert (
        table.symbol_for("OrigBlue", CellProfilerSymbolKind.IMAGE).producer_module_num
        is None
    )
    assert table.symbol_for("Nuclei", CellProfilerSymbolKind.OBJECTS).kind is (
        CellProfilerSymbolKind.OBJECTS
    )
    assert (
        table.symbol_for("Nuclei", CellProfilerSymbolKind.OBJECTS).producer_module_num
        == 1
    )
    assert table.symbol_for("Cytoplasm", CellProfilerSymbolKind.OBJECTS).kind is (
        CellProfilerSymbolKind.OBJECTS
    )
    assert table.symbol_for(
        "MeasureObjectIntensity_4_measurements",
        CellProfilerSymbolKind.MEASUREMENTS,
    ).kind is (
        CellProfilerSymbolKind.MEASUREMENTS
    )

    primary_contract = table.contracts_by_module_num[1]
    assert [spec.kind for spec in primary_contract.inputs] == [ArtifactKind.IMAGE]
    assert tuple(
        binding.alias
        for binding in primary_contract.source_bindings.groups[0].bindings
    ) == ("OrigBlue",)
    assert primary_contract.runtime_artifact_inputs == ()
    assert primary_contract.outputs[0].kind is ArtifactKind.OBJECT_LABELS
    assert isinstance(primary_contract.module_contract, ModuleArtifactContract)

    secondary_contract = table.contracts_by_module_num[2]
    assert [spec.name for spec in secondary_contract.outputs] == ["Cells"]

    measure_contract = table.contracts_by_module_num[4]
    assert tuple(
        binding.alias
        for binding in measure_contract.source_bindings.groups[0].bindings
    ) == ("OrigBlue", "OrigGreen")
    assert [spec.name for spec in measure_contract.runtime_artifact_inputs] == [
        "Nuclei",
        "Cells",
        "Cytoplasm",
    ]
    assert measure_contract.outputs[0].kind is ArtifactKind.MEASUREMENTS


def test_cellprofiler_symbol_table_fails_for_unknown_object_input():
    modules = [
        _module(
            1,
            "MeasureObjectSizeShape",
            {"Select object sets to measure": "MissingObjects"},
        )
    ]

    with pytest.raises(ValueError, match="unknown objects symbol 'MissingObjects'"):
        CellProfilerSymbolTable.compile(modules)


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
        2,
        "MeasureObjectSizeShape",
        {"Select object sets to measure": "LoadedNuclei"},
    )

    table = CellProfilerSymbolTable.compile([setup_module, measurement_module])
    contract = table.contracts_by_module_num[2]

    assert table.symbol_for(
        "LoadedNuclei",
        CellProfilerSymbolKind.OBJECTS,
    ).source_bound is True
    assert contract.runtime_artifact_inputs == ()
    assert contract.source_bindings.groups[0].bindings[0].artifact_kind is (
        ArtifactKind.OBJECT_LABELS
    )
    assert [spec.name for spec in contract.inputs] == ["LoadedNuclei"]
    assert [spec.kind for spec in contract.inputs] == [ArtifactKind.OBJECT_LABELS]


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
    ]
    assert [spec.name for spec in contract.outputs] == [
        "FilterObjects_4_measurements",
        "MyFilteredObjects",
        "FilteredCells",
        "FilteredCytoplasm",
    ]
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.OBJECT_LABELS,
        ArtifactKind.OBJECT_LABELS,
        ArtifactKind.OBJECT_LABELS,
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
                ("Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?", "Yes"),
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
        "FilteredObjects",
        "OutlinesFilteredCells",
    ]
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.OBJECT_LABELS,
        ArtifactKind.OBJECT_LABELS,
        ArtifactKind.IMAGE,
        ArtifactKind.IMAGE,
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
    ]
    assert [spec.name for spec in contract.inputs] == ["Cells", "Tiles"]


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
            3,
            "MeasureObjectSizeShape",
            {"Select object sets to measure": "Nuclei"},
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)

    assert table.symbol_for("Nuclei", CellProfilerSymbolKind.OBJECTS).producer_module_num == 2
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

    image_symbol = table.symbol_for("PH3", CellProfilerSymbolKind.IMAGE)
    object_symbol = table.symbol_for("PH3", CellProfilerSymbolKind.OBJECTS)
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
            3,
            "RelateObjects",
            {
                "Parent objects": "Nuclei",
                "Child objects": "PH3",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[3]

    assert [symbol.name for symbol in contract.input_symbols] == ["Nuclei", "PH3"]
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.RELATIONSHIPS,
        ArtifactKind.MEASUREMENTS,
    ]


def test_pipeline_generator_emits_compiled_artifact_contracts():
    generator = PipelineGenerator()
    modules = [_identify_primary(), _identify_secondary()]

    generated = generator.generate_from_registry(
        pipeline_name="cp_graph",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )

    assert len(generated.artifact_contracts) == 2
    assert "CELLPROFILER_MODULE_CONTRACTS" in generated.code
    assert "ModuleArtifactContract(" in generated.code
    assert "source_bindings=StepSourceBindingsConfig(" in generated.code
    assert "runtime_artifact_inputs=(ArtifactSpec('Nuclei'" in generated.code
    assert "identify_primary_objects_1 = require_function" in generated.code
    assert "identify_secondary_objects_2 = require_function" in generated.code
    assert "CellProfilerModuleExecutor" in generated.code
    assert "cellprofiler_runtime_adapter_factory" in generated.code
    assert "@artifact_outputs(*CELLPROFILER_MODULE_CONTRACTS[1]" in generated.code
    assert "@artifact_inputs(*CELLPROFILER_MODULE_CONTRACTS[2]" in generated.code
    assert "@runtime_adapter(\"cellprofiler_runtime\"" in generated.code
    assert "identify_primary_objects_1_runtime.input_memory_type" in generated.code
    assert "func=identify_primary_objects_1_runtime" in generated.code
    assert "func=identify_secondary_objects_2_runtime" in generated.code


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

    assert (
        'measure_texture_objects_2 = require_function("MeasureTexture", '
        'function_name="measure_texture_objects")'
    ) in generated.code
    assert (
        'measure_colocalization_objects_3 = require_function('
        '"MeasureColocalization", function_name="measure_colocalization_objects")'
    ) in generated.code
    assert (
        'measure_granularity_objects_4 = require_function('
        '"MeasureGranularity", function_name="measure_granularity_objects")'
    ) in generated.code


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
    assert (
        'measure_colocalization_objects_2 = require_function('
        '"MeasureCorrelation", function_name="measure_colocalization_objects")'
    ) in generated.code
    assert "module_name='MeasureColocalization'" in generated.code


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
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.IMAGE,
        ArtifactKind.MEASUREMENTS,
    ]
    assert [spec.name for spec in contract.outputs] == [
        "Foreground",
        "MeasureImageAreaOccupied_1_measurements",
    ]
    assert (
        'measure_image_area_occupied_1 = require_function('
        '"MeasureImageAreaOccupied", '
        'function_name="measure_image_area_occupied")'
    ) in generated.code


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
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.IMAGE,
        ArtifactKind.MEASUREMENTS,
    ]
    assert [spec.name for spec in contract.outputs] == [
        "OccupiedNuclei",
        "MeasureImageAreaOccupied_2_measurements",
    ]
    assert (
        'measure_image_area_occupied_2 = require_function('
        '"MeasureImageAreaOccupied", '
        'function_name="measure_image_area_occupied")'
    ) in generated.code


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
    assert "'input_names': ('DNA', 'Nuclei')" in generated.code


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
        pipeline_name="align",
        source_cppipe=Path("source.pipeline"),
        modules=[module],
    )

    assert [spec.name for spec in contract.inputs] == ["Image1", "Image2"]
    assert [spec.name for spec in contract.outputs] == [
        "AlignedImage1",
        "AlignedImage2",
    ]
    assert (
        'align_1 = require_function("Align", function_name="align")'
        in generated.code
    )
    assert "'crop_mode': 'Keep size'" in generated.code


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
        "'custom_absorbances': ((0.5, 0.5, 0.5), "
        "(0.5, 0.5, 0.5), (0.1, 0.2, 0.3))"
    ) in generated.code


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
        ),
    ]

    generated = generator.generate_from_registry(
        pipeline_name="image_colocalization",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )
    contract = generated.artifact_contracts[0]

    assert [spec.name for spec in contract.inputs] == ["OrigBlue", "OrigGreen"]
    assert (
        'measure_colocalization_1 = require_function('
        '"MeasureColocalization", function_name="measure_colocalization")'
    ) in generated.code


def test_pipeline_generator_preserves_default_materialization_for_tabular_outputs():
    generator = PipelineGenerator()
    modules = [
        _identify_primary(),
        _module(
            2,
            "MeasureImageIntensity",
            {
                "Select images to measure": "OrigBlue",
                "Select input object sets": "",
            },
        ),
    ]

    generated = generator.generate_from_registry(
        pipeline_name="cp_materialization_defaults",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )

    assert (
        "ArtifactSpec('Nuclei', ArtifactKind.OBJECT_LABELS, "
        "materialization=NO_ARTIFACT_MATERIALIZATION)"
    ) in generated.code
    assert (
        "ArtifactSpec('MeasureImageIntensity_2_measurements', "
        "ArtifactKind.MEASUREMENTS)"
    ) in generated.code


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

    assert "'intensity_choice': 'background'" in generated.code
    assert "'block_size': 40" in generated.code
    assert "'rescale_option': 'no'" in generated.code
    assert "'smoothing_method': 'convex_hull'" in generated.code
    assert "'filter_size_method': 'manually'" in generated.code
    assert "'manual_filter_size': 10" in generated.code
    assert "'method': 'subtract'" in generated.code
    assert "'truncate_low': False" in generated.code
    assert "'truncate_high': True" in generated.code


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
        binding.alias
        for binding in illumination_contract.source_bindings.groups[0].bindings
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
    assert [spec.name for spec in measure_intensity_contract.runtime_artifact_inputs] == [
        "OpeningBlue",
        "Nuclei",
    ]

    granularity_contract = table.contracts_by_module_num[8]
    assert [spec.name for spec in granularity_contract.runtime_artifact_inputs] == [
        "OpeningBlue",
        "Nuclei",
    ]
    assert granularity_contract.outputs[0].kind is ArtifactKind.MEASUREMENTS


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

    assert [(spec.name, spec.kind) for spec in contract.inputs] == [
        ("DNA", ArtifactKind.IMAGE),
        ("PrimaryOutlines", ArtifactKind.IMAGE),
        ("SecondaryOutlines", ArtifactKind.IMAGE),
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

    assert [(spec.name, spec.kind) for spec in contract.inputs] == [
        ("DNA", ArtifactKind.IMAGE),
        ("PrimaryOutlines", ArtifactKind.IMAGE),
        ("Nuclei", ArtifactKind.OBJECT_LABELS),
    ]
    assert [(spec.name, spec.kind) for spec in contract.runtime_artifact_inputs] == [
        ("Nuclei", ArtifactKind.OBJECT_LABELS),
    ]


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
    assert [spec.kind for spec in contract.inputs] == [ArtifactKind.IMAGE]
    assert tuple(
        binding.alias
        for binding in contract.source_bindings.groups[0].bindings
    ) == ("OrigBlue",)
    assert [spec.name for spec in contract.outputs] == [
        "IllumBlue",
        "CorrectIlluminationCalculate_1_measurements",
    ]
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.IMAGE,
        ArtifactKind.MEASUREMENTS,
    ]


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
    assert [spec.kind for spec in contract.runtime_artifact_inputs] == [
        ArtifactKind.OBJECT_LABELS
    ]
    assert [spec.name for spec in contract.outputs] == [
        "DilateObjects_2_measurements",
        "DilatedNuclei",
    ]
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.OBJECT_LABELS,
    ]


def test_cellprofiler_symbol_table_infers_special_output_only_contract():
    table = CellProfilerSymbolTable.compile(
        [
            _module(
                1,
                "CalculateMath",
                {"Operation": "Add"},
            )
        ]
    )
    contract = table.contracts_by_module_num[1]

    assert contract.inputs == ()
    assert [spec.name for spec in contract.outputs] == [
        "CalculateMath_1_measurements"
    ]
    assert [spec.kind for spec in contract.outputs] == [ArtifactKind.MEASUREMENTS]


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
        )
    ]

    table = CellProfilerSymbolTable.compile(modules)
    contract = table.contracts_by_module_num[2]

    assert [spec.name for spec in contract.inputs] == ["Nuclei", "OrigBlue"]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == ["Nuclei"]
    assert tuple(
        binding.alias
        for binding in contract.source_bindings.groups[0].bindings
    ) == ("OrigBlue",)
    assert [spec.name for spec in contract.outputs] == [
        "MaskObjects_2_measurements",
        "MaskedNuclei",
    ]
    assert [spec.kind for spec in contract.outputs] == [
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.OBJECT_LABELS,
    ]


def test_cellprofiler_symbol_table_rejects_unknown_generic_object_input():
    with pytest.raises(
        ValueError,
        match=(
            r"Module FilterObjects\(1\) references unknown objects "
            r"symbol 'Nuclei'"
        ),
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
    assert tuple(
        binding.alias for binding in contract.source_bindings.groups[0].bindings
    ) == ("OrigBlue", "OrigGreen")
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
        pipeline_name="classify",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )

    assert PipelineGenerator().has_module("ClassifyObjects")
    assert contract.module_name == "ClassifyObjectsSingleMeasurement"
    assert [spec.name for spec in contract.inputs] == ["Nuclei"]
    assert [spec.name for spec in contract.outputs] == [
        "ClassifyObjects_2_measurements"
    ]
    assert (
        'classify_objects_single_measurement_2 = require_function('
        '"ClassifyObjects", function_name="classify_objects_single_measurement")'
    ) in generated.code
    assert "'measurement_feature': 'Math_Ratio'" in generated.code
    assert "'bin_choice': 'custom'" in generated.code
    assert "'custom_thresholds': '0.25,0.75'" in generated.code


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
        pipeline_name="grid",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )

    assert [spec.name for spec in define_grid.inputs] == ["OrigBlue", "Nuclei"]
    assert [spec.name for spec in define_grid.outputs] == [
        "DefineGrid_2_measurements"
    ]
    assert [spec.name for spec in identify_grid.inputs] == ["Nuclei"]
    assert [spec.name for spec in identify_grid.outputs] == [
        "IdentifyObjectsInGrid_3_measurements",
        "GridObjects",
    ]
    assert (
        'define_grid_automatic_2 = require_function('
        '"DefineGrid", function_name="define_grid_automatic")'
    ) in generated.code
    assert (
        'identify_objects_in_grid_with_guides_3 = require_function('
        '"IdentifyObjectsInGrid", '
        'function_name="identify_objects_in_grid_with_guides")'
    ) in generated.code
    assert "Natural Shape and Location" not in [
        spec.name for spec in identify_grid.inputs
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
                "Name the output non-overlapping worm objects": (
                    "NonOverlappingWorms"
                ),
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)

    assert [spec.name for spec in table.contracts_by_module_num[2].outputs] == [
        "MaskObjects_2_measurements",
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
        "mCherry",
        "GFP",
    ]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == [
        "NonOverlappingWorms",
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
    assert "'flip_mode': 'top_brightest'" in generated.code


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
        "IdentifyPrimaryObjects",
    ]
