from pathlib import Path

import pytest

from benchmark.cellprofiler_compat import CellProfilerModuleContract
from benchmark.converter.parser import CPPipeParser, ModuleBlock, ModuleSetting
from benchmark.converter.pipeline_generator import PipelineGenerator
from benchmark.converter.runtime_pipeline import partition_cppipe_modules
from benchmark.converter.symbol_table import (
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
)
from openhcs.core.artifacts import ArtifactKind


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

    assert table.symbols["OrigBlue"].kind is CellProfilerSymbolKind.IMAGE
    assert table.symbols["OrigBlue"].producer_module_num is None
    assert table.symbols["Nuclei"].kind is CellProfilerSymbolKind.OBJECTS
    assert table.symbols["Nuclei"].producer_module_num == 1
    assert table.symbols["Cytoplasm"].kind is CellProfilerSymbolKind.OBJECTS
    assert table.symbols["MeasureObjectIntensity_4_measurements"].kind is (
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
    assert isinstance(primary_contract.module_contract, CellProfilerModuleContract)

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

    assert table.symbols["LoadedNuclei"].source_bound is True
    assert contract.runtime_artifact_inputs == ()
    assert contract.source_bindings.groups[0].bindings[0].artifact_kind is (
        ArtifactKind.OBJECT_LABELS
    )
    assert [spec.name for spec in contract.inputs] == ["LoadedNuclei"]
    assert [spec.kind for spec in contract.inputs] == [ArtifactKind.OBJECT_LABELS]


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


def test_cellprofiler_symbol_table_fails_for_duplicate_producer():
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
    ]

    with pytest.raises(ValueError, match="already produced by module 1"):
        CellProfilerSymbolTable.compile(modules)


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
    assert "CellProfilerModuleContract(" in generated.code
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


def test_partition_cppipe_modules_skips_setup_and_export_modules():
    modules = (
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
