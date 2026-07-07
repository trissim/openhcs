import ast
import re
from pathlib import Path

import pytest

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.module_processing_components import (
    SourceBindingProcessingScope,
    SourceProcessingAxisPlan,
    SourceProcessingComponentSemantics,
)
from openhcs.interop.cellprofiler.pipeline_generator import (
    PipelineGenerator,
)
from openhcs.interop.cellprofiler.source_schema import compile_image_schema
from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbolTable
from openhcs.core.artifacts import ObjectLabelsArtifactType
from openhcs.core.pipeline_image_schema import (
    ImagePlaneSource,
    PipelineImageSchema,
    SourceImageStackPlan,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceFilterMatchType,
    SourceBindingOrigin,
    SourceFilterSubject,
    StepSourceBindingsConfig,
)
from openhcs.processing.backends.cellprofiler.alignment import AlignModule


def _module_with_records(
    module_num: int,
    name: str,
    setting_pairs: list[tuple[str, str]],
) -> ModuleBlock:
    records = [ModuleSetting(setting_name, value) for setting_name, value in setting_pairs]
    settings: dict[str, str] = {}
    for record in records:
        settings[record.name] = record.value
    return ModuleBlock(
        name=name,
        module_num=module_num,
        settings=settings,
        setting_records=records,
    )


def _schema_from_in_tree_cppipe(cppipe_name: str):
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / cppipe_name
    )
    modules = CPPipeParser().parse(cppipe_path)
    setup_module_names = {
        "LoadImages",
        "Images",
        "Metadata",
        "NamesAndTypes",
        "Groups",
    }
    setup_modules = [
        module
        for module in modules
        if module.name in setup_module_names
    ]
    return compile_image_schema(setup_modules)


def test_cppipe_parser_preserves_repeated_settings_in_order(tmp_path: Path):
    cppipe_path = tmp_path / "repeated.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:5",
                "",
                "NamesAndTypes:[module_num:3|enabled:True]",
                "    Assignments count:2",
                "    Select the rule criteria:and (metadata does channel \"1\")",
                "    Name to assign these images:DAPI",
                "    Select the rule criteria:and (metadata does channel \"2\")",
                "    Name to assign these images:Actin",
                "",
            ]
        )
    )

    modules = CPPipeParser().parse(cppipe_path)

    assert len(modules) == 1
    names_and_types = modules[0]
    assert names_and_types.settings["Name to assign these images"] == "Actin"
    assert names_and_types.get_setting_values("Name to assign these images") == (
        "DAPI",
        "Actin",
    )
    assert tuple(
        setting.value
        for setting in names_and_types.iter_settings("Select the rule criteria")
    ) == (
        'and (metadata does channel "1")',
        'and (metadata does channel "2")',
    )


def test_cppipe_parser_supports_indented_legacy_pipeline_modules(tmp_path: Path):
    pipeline_path = tmp_path / "legacy_indented.pipeline"
    pipeline_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "        Version:1",
                "",
                "        LoadImages:[module_num:1|enabled:True]",
                "            What type of files are you loading?:individual images",
                "            Type the text that these images have in common "
                "(case-sensitive):Channel2",
                "            What do you want to call this image in CellProfiler?:DNA",
            ]
        )
    )

    modules = CPPipeParser().parse(pipeline_path)

    assert len(modules) == 1
    assert modules[0].name == "LoadImages"
    assert (
        modules[0].get_setting(
            "What do you want to call this image in CellProfiler?"
        )
        == "DNA"
    )


def test_cppipe_parser_extracts_embedded_image_plane_sources(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "url_planes.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "HasImagePlaneDetails:True",
                "",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "",
                '"Version":"1","PlaneCount":"2"',
                '"URL","Series","Index","Channel"',
                '"https://example.invalid/A_D.TIF",,,',
                '"file:/tmp/A_F.TIF","0","1","2"',
            ]
        ),
        encoding="utf-8",
    )

    parser = CPPipeParser()
    modules = parser.parse(cppipe_path)

    assert parser.image_plane_sources == (
        {
            "uri": "https://example.invalid/A_D.TIF",
            "series": None,
            "index": None,
            "channel": None,
        },
        {
            "uri": "file:/tmp/A_F.TIF",
            "series": "0",
            "index": "1",
            "channel": "2",
        },
    )
    assert modules[0].metadata["image_plane_sources"] == parser.image_plane_sources


def test_compile_image_schema_carries_embedded_image_plane_sources() -> None:
    images_module = _module_with_records(
        1,
        "Images",
        [
            ("Filter images?", "Images only"),
            ("Select the rule criteria", "and (extension does isimage)"),
        ],
    )
    images_module.metadata["image_plane_sources"] = (
        {
            "uri": "https://example.invalid/A_D.TIF",
            "series": None,
            "index": None,
            "channel": None,
        },
    )

    schema = compile_image_schema([images_module])

    assert schema.image_plane_sources == (
        ImagePlaneSource(uri="https://example.invalid/A_D.TIF"),
    )


def test_compile_image_schema_lowers_images_module_to_source_universe_filters():
    images_module = _module_with_records(
        1,
        "Images",
        [
            ("Filter images?", "Images only"),
            ("Select the rule criteria", 'or (file does containregexp "A01")'),
        ],
    )

    schema = compile_image_schema([images_module])

    assert schema.images_rule is not None
    assert schema.images_rule.filters[0].match_type is SourceFilterMatchType.IS_IMAGE
    assert schema.images_rule.filters[1].subject is SourceFilterSubject.FILE
    assert (
        schema.images_rule.filters[1].match_type
        is SourceFilterMatchType.CONTAINS_REGEX
    )
    assert schema.images_rule.filters[1].value == "A01"


def test_compile_image_schema_lowers_extension_is_suffix_filters():
    names_and_types_module = _module_with_records(
        1,
        "NamesAndTypes",
        [
            ("Assignments count", "1"),
            ("Select the rule criteria", "and (extension does ispng)"),
            ("Name to assign these images", "DNA"),
            ("Select the image type", "Grayscale image"),
        ],
    )

    schema = compile_image_schema([names_and_types_module])
    assignment = schema.assignment_for_alias("DNA")

    assert assignment is not None
    assert assignment.selector.filters[0].subject is SourceFilterSubject.EXTENSION
    assert assignment.selector.filters[0].match_type is SourceFilterMatchType.EQUALS
    assert assignment.selector.filters[0].value == ".png"


def test_compile_image_schema_drops_empty_scalar_filter_clauses():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Regular expression to extract from file name", r"(?P<Well>A01)"),
            ("Select the filtering criteria", 'and (file doesnot contain "")'),
        ],
    )

    schema = compile_image_schema([metadata_module])

    assert schema.metadata_rules[0].filters == ()


def test_compile_image_schema_preserves_disabled_path_metadata_for_source_projection():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Extract metadata?", "No"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])",
            ),
            ("Select the filtering criteria", 'and (file does contain "")'),
        ],
    )

    schema = compile_image_schema([metadata_module])

    assert len(schema.metadata_rules) == 1
    assert schema.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert "Well" in schema.metadata_rules[0].pattern


def test_compile_image_schema_combines_imported_metadata_location_and_filename():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Import from file"),
            ("Metadata file location", "Default Input Folder|metadata"),
            ("Metadata file name", "plate.csv"),
            ("Match file and image metadata", "[]"),
        ],
    )

    schema = compile_image_schema([metadata_module])

    assert schema.imported_metadata_tables[0].location == "metadata/plate.csv"


def test_compile_image_schema_does_not_conjoin_images_module_disjunctions():
    images_module = _module_with_records(
        1,
        "Images",
        [
            ("Filter images?", "Images only"),
            (
                "Select the rule criteria",
                'or (extension does isimage) (file does endwith ".npy")',
            ),
        ],
    )

    schema = compile_image_schema([images_module])

    assert schema.images_rule is None


def test_compile_image_schema_does_not_conjoin_nested_images_module_disjunctions():
    images_module = _module_with_records(
        1,
        "Images",
        [
            ("Filter images?", "Images only"),
            (
                "Select the rule criteria",
                'and (extension does isimage) (or (file does contain "_s1_") '
                '(file does contain "_s2_"))',
            ),
        ],
    )

    schema = compile_image_schema([images_module])

    assert schema.images_rule is None


def test_compile_image_schema_lowers_names_and_types_to_typed_selectors():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            (
                "Regular expression to extract from file name",
                r".*(?P<well>[A-Z]\d+)_s(?P<site>\d+)_w(?P<channel>\d)",
            ),
            (
                "Regular expression to extract from folder name",
                r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$",
            ),
            ("Select the filtering criteria", 'and (file does contain "")'),
        ],
    )
    names_and_types_module = _module_with_records(
        2,
        "NamesAndTypes",
        [
            ("Assignments count", "2"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Name to assign these images", "DAPI"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (metadata does illum "DAPI")'),
            ("Name to assign these images", "DAPIillum"),
            ("Select the image type", "Illumination function"),
            ("Match metadata", "[{'DAPI': 'folder', 'DAPIillum': 'folder_illum'}]"),
            ("Image set matching method", "Metadata"),
        ],
    )
    groups_module = _module_with_records(
        3,
        "Groups",
        [
            ("Do you want to group your images?", "Yes"),
            ("Metadata category", "folder"),
            ("Metadata category", "well"),
        ],
    )

    schema = compile_image_schema(
        [metadata_module, names_and_types_module, groups_module]
    )

    dapi = schema.assignment_for_alias("DAPI")
    assert dapi is not None
    assert dapi.origin is SourceBindingOrigin.STEP_INPUT
    assert dapi.selector.metadata == (
        MetadataSelector("channel", "1"),
    )

    illumination = schema.source_artifact_for_alias("DAPIillum")
    assert illumination is not None
    assert illumination.origin is SourceBindingOrigin.PIPELINE_START
    assert illumination.selector.metadata == (
        MetadataSelector("illum", "DAPI"),
    )
    assert schema.match_plan is not None
    assert schema.match_plan.method is SourceBindingMatchMethod.METADATA
    assert schema.match_plan.dimensions[0].field_for_alias("DAPI") == "folder"
    assert (
        schema.match_plan.dimensions[0].field_for_alias("DAPIillum")
        == "folder_illum"
    )

    assert schema.grouping is not None
    assert schema.grouping.metadata_fields == ("folder", "well")
    assert schema.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert schema.metadata_rules[0].filters == ()


def test_compile_image_schema_lowers_object_loads_to_source_artifacts():
    names_and_types_module = _module_with_records(
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

    schema = compile_image_schema([names_and_types_module])
    source_artifact = schema.resolved_source_artifact_for_alias(
        "LoadedNuclei",
        ObjectLabelsArtifactType,
    )

    assert source_artifact is not None
    assert source_artifact.artifact_kind is ObjectLabelsArtifactType
    assert source_artifact.selector.metadata == (
        MetadataSelector("channel", "3"),
    )
    assert schema.assignment_for_alias("IgnoredImageAlias") is None


def test_compile_image_schema_ignores_disabled_metadata_module():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Extract metadata?", "No"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])",
            ),
            ("Select the filtering criteria", 'and (file does contain "")'),
        ],
    )

    schema = compile_image_schema([metadata_module])

    assert schema.metadata_rules == ()


def test_compile_image_schema_ignores_disabled_metadata_regex_for_ordered_image_sets():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Extract metadata?", "No"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])",
            ),
            ("Select the filtering criteria", 'and (file does contain "")'),
        ],
    )
    names_and_types_module = _module_with_records(
        2,
        "NamesAndTypes",
        [
            ("Assignments count", "2"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (file does contain "Ch1")'),
            ("Name to assign these images", "BF_image"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (file does contain "Ch6")'),
            ("Name to assign these images", "DF_image"),
            ("Select the image type", "Grayscale image"),
        ],
    )

    schema = compile_image_schema([metadata_module, names_and_types_module])

    assert schema.match_plan is not None
    assert schema.match_plan.method is SourceBindingMatchMethod.ORDER
    assert schema.metadata_rules == ()


def test_codegen_groups_metadata_free_ordered_image_sets_by_workspace_site():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Extract metadata?", "No"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Site>[0-9]+)_(?P<Channel>[A-Z])",
            ),
        ],
    )
    names_and_types_module = _module_with_records(
        2,
        "NamesAndTypes",
        [
            ("Assignments count", "3"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (file does contain "_D")'),
            ("Name to assign these images", "OrigBlue"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (file does contain "_F")'),
            ("Name to assign these images", "OrigGreen"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (file does contain "_R")'),
            ("Name to assign these images", "OrigRed"),
            ("Select the image type", "Grayscale image"),
        ],
    )
    crop_module = ModuleBlock(
        name="Crop",
        module_num=3,
        settings={
            "Select the input image": "OrigBlue",
            "Name the output image": "CropBlue",
            "Select the cropping shape": "Rectangle",
        },
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_ordered_sources",
        source_cppipe=Path("source.cppipe"),
        modules=[crop_module],
        skipped_modules=[metadata_module, names_and_types_module],
    )

    assert generated.pipeline_config is not None
    assert generated.pipeline_config.source_bindings_config.match_plan is not None
    assert (
        generated.pipeline_config.source_bindings_config.match_plan.method
        is SourceBindingMatchMethod.ORDER
    )
    assert "variable_components=[VariableComponents.SITE]" in generated.code
    assert "group_by=GroupBy.CHANNEL" in generated.code


def test_codegen_preserves_single_source_alias_image_set_axis():
    names_and_types_module = _module_with_records(
        1,
        "NamesAndTypes",
        [
            ("Assignments count", "1"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (file does contain "_D")'),
            ("Name to assign these images", "OrigBlue"),
            ("Select the image type", "Grayscale image"),
        ],
    )
    crop_module = ModuleBlock(
        name="Crop",
        module_num=2,
        settings={
            "Select the input image": "OrigBlue",
            "Name the output image": "CropBlue",
            "Select the cropping shape": "Rectangle",
        },
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_single_ordered_source",
        source_cppipe=Path("source.cppipe"),
        modules=[crop_module],
        skipped_modules=[names_and_types_module],
    )
    step_start = generated.code.index('# CellProfiler artifact outputs: image:CropBlue')
    step_source = generated.code[step_start:]
    processing_start = step_source.index("processing_config=LazyProcessingConfig(")
    processing_end = step_source.index("        ),", processing_start)
    processing_config = step_source[processing_start:processing_end]

    assert "source_bindings=LazyStepSourceBindingsConfig" in step_source
    assert "variable_components=[VariableComponents.SITE]" in processing_config
    assert "group_by=GroupBy.CHANNEL" in processing_config


def test_compile_image_schema_treats_binary_masks_as_stack_images():
    names_and_types_module = _module_with_records(
        1,
        "NamesAndTypes",
        [
            ("Assignments count", "1"),
            ("Select the rule criteria", 'and (metadata does channel "mask")'),
            ("Name to assign these images", "BinaryMask"),
            ("Select the image type", "Binary mask"),
        ],
    )

    schema = compile_image_schema([names_and_types_module])
    assignment = schema.assignment_for_alias("BinaryMask")

    assert assignment is not None
    assert assignment.origin is SourceBindingOrigin.STEP_INPUT
    assert assignment.selector.metadata == (
        MetadataSelector("channel", "mask"),
    )


def test_compile_image_schema_lowers_load_images_to_typed_source_schema():
    load_images_module = _module_with_records(
        1,
        "LoadImages",
        [
            ("What type of files are you loading?", "individual images"),
            ("How do you want to load these files?", "Text-Exact match"),
            ("Do you want to exclude certain files?", "Yes"),
            ("Type the text that the excluded images have in common", "ILLUM"),
            ("Do you want to group image sets by metadata?", "Yes"),
            ("What metadata fields do you want to group by?", "WellRow,WellCol"),
            (
                "Type the text that these images have in common (case-sensitive)",
                "Channel2",
            ),
            ("What do you want to call this image in CellProfiler?", "DNA"),
            ("What is the position of this image in each group?", "1"),
            (
                "Do you want to extract metadata from the file name, "
                "the subfolder path or both?",
                "File name",
            ),
            (
                "Type the regular expression that finds metadata in the file name\\x3A",
                r"^.*-(?P<WellRow>.+)-(?P<WellCol>\x5B0-9\x5D{2})",
            ),
            (
                "Type the regular expression that finds metadata in the "
                "subfolder path\\x3A",
                "None",
            ),
        ],
    )

    schema = compile_image_schema([load_images_module])
    dna = schema.assignment_for_alias("DNA")

    assert dna is not None
    assert dna.origin is SourceBindingOrigin.PIPELINE_START
    assert dna.selector.filters[0].subject is SourceFilterSubject.FILE
    assert dna.selector.filters[0].match_type is SourceFilterMatchType.CONTAINS
    assert dna.selector.filters[0].value == "Channel2"
    assert dna.selector.filters[1].match_type is SourceFilterMatchType.DOES_NOT_CONTAIN
    assert dna.selector.filters[1].value == "ILLUM"
    assert len(schema.metadata_rules) == 1
    assert schema.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert schema.metadata_rules[0].pattern == (
        r"^.*-(?P<WellRow>.+)-(?P<WellCol>[0-9]{2})"
    )
    assert schema.metadata_rules[0].filters == dna.selector.filters
    assert schema.grouping is not None
    assert schema.grouping.metadata_fields == ("WellRow", "WellCol")


def test_compile_image_schema_supports_v5_regex_labels_and_file_filters():
    metadata_module = _module_with_records(
        1,
        "Metadata",
        [
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Regular expression", r".*-(?P<ImageNumber>\d*)-(?P<WellRow>.*)"),
            ("Regular expression", r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$"),
            ("Select the filtering criteria", 'and (file does contain "Channel1-")'),
            ("Metadata extraction method", "Import from file"),
            ("Metadata source", "File name"),
            ("Metadata file location", "Default Input Folder|metadata.csv"),
            (
                "Match file and image metadata",
                "[{'Image Metadata': 'WellRow', 'CSV Metadata': 'Row'}]",
            ),
        ],
    )
    names_and_types_module = _module_with_records(
        2,
        "NamesAndTypes",
        [
            ("Assignments count", "2"),
            (
                "Select the rule criteria",
                'and (file does contain "Channel1-") (extension does istif)',
            ),
            ("Name to assign these images", "rawGFP"),
            ("Select the image type", "Grayscale image"),
            (
                "Select the rule criteria",
                'and (file does contain "Channel1") (file does endwith ".mat")',
            ),
            ("Name to assign these images", "IllumDNA"),
            ("Select the image type", "Illumination function"),
        ],
    )

    schema = compile_image_schema([metadata_module, names_and_types_module])

    assert len(schema.metadata_rules) == 1
    assert (
        schema.metadata_rules[0].pattern
        == r".*-(?P<ImageNumber>\d*)-(?P<WellRow>.*)"
    )
    assert (
        schema.metadata_rules[0].filters[0].match_type
        is SourceFilterMatchType.CONTAINS
    )
    assert len(schema.imported_metadata_tables) == 1
    assert schema.imported_metadata_tables[0].location == "metadata.csv"
    assert (
        schema.imported_metadata_tables[0].joins[0].image_metadata_field
        == "WellRow"
    )
    assert (
        schema.imported_metadata_tables[0].joins[0].imported_metadata_field
        == "Row"
    )

    raw_gfp = schema.assignment_for_alias("rawGFP")
    illum_dna = schema.source_artifact_for_alias("IllumDNA")
    assert raw_gfp is not None
    assert raw_gfp.selector.filters[0].match_type is SourceFilterMatchType.CONTAINS
    assert raw_gfp.selector.filters[1].match_type is SourceFilterMatchType.IS_TIF
    assert illum_dna is not None
    assert illum_dna.selector.filters[0].match_type is SourceFilterMatchType.CONTAINS
    assert illum_dna.selector.filters[1].match_type is SourceFilterMatchType.ENDS_WITH


def test_compile_image_schema_lowers_cellprofiler_file_equality_filter():
    names_and_types_module = _module_with_records(
        1,
        "NamesAndTypes",
        [
            ("Assignments count", "1"),
            (
                "Select the rule criteria",
                'and (file does eq "VitraChannel1ILLUM.npy")',
            ),
            ("Name to assign these images", "IllumChannel1"),
            ("Select the image type", "Illumination function"),
        ],
    )

    schema = compile_image_schema([names_and_types_module])
    illum = schema.source_artifact_for_alias("IllumChannel1")

    assert illum is not None
    assert len(illum.selector.filters) == 1
    assert illum.selector.filters[0].subject is SourceFilterSubject.FILE
    assert illum.selector.filters[0].match_type is SourceFilterMatchType.EQUALS
    assert illum.selector.filters[0].value == "VitraChannel1ILLUM.npy"
    assert illum.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )


def test_symbol_table_and_codegen_use_compiled_setup_schema():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r".*(?P<well>[A-Z]\d+)_s(?P<site>\d+)_w(?P<channel>\d)",
                ),
                (
                    "Regular expression to extract from folder name",
                    r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$",
                ),
                ("Select the filtering criteria", 'and (file does contain "")'),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "2"),
                ("Select the rule criteria", 'and (metadata does channel "1")'),
                ("Name to assign these images", "DAPI"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does illum "DAPI")'),
                ("Name to assign these images", "DAPIillum"),
                ("Select the image type", "Illumination function"),
                ("Match metadata", "[{'DAPI': 'folder', 'DAPIillum': 'folder_illum'}]"),
                ("Image set matching method", "Metadata"),
            ],
        ),
        _module_with_records(
            3,
            "Groups",
            [
                ("Do you want to group your images?", "Yes"),
                ("Metadata category", "folder"),
            ],
        ),
    ]
    processing_module = ModuleBlock(
        name="CorrectIlluminationApply",
        module_num=4,
        settings={
            "Select the input image": "DAPI",
            "Select the illumination function": "DAPIillum",
            "Name the output image": "CorrDAPI",
        },
    )

    table = CellProfilerSymbolTable.compile([*setup_modules, processing_module])
    contract = table.contracts_by_module_num[4]
    bindings = contract.source_bindings.bindings

    assert bindings[0].alias == "DAPI"
    assert bindings[0].selector.metadata == (
        MetadataSelector("channel", "1"),
    )
    assert bindings[1].alias == "DAPIillum"
    assert bindings[1].origin is SourceBindingOrigin.PIPELINE_START
    assert bindings[1].selector.metadata == (
        MetadataSelector("illum", "DAPI"),
    )
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_setup_schema",
        source_cppipe=Path("source.cppipe"),
        modules=[processing_module],
        skipped_modules=setup_modules,
    )

    assert "MetadataSelector('channel', '1')" in generated.code
    assert "MetadataSelector('illum', 'DAPI')" in generated.code
    assert generated.pipeline_config is not None
    assert generated.pipeline_config.source_bindings_config.match_plan is not None
    assert "SourceBindingOrigin.PIPELINE_START" in generated.code
    assert "input_source=InputSource.PIPELINE_START" in generated.code
    assert "group_by=GroupBy.CHANNEL" in generated.code


def test_measure_image_quality_all_loaded_images_uses_module_declared_sources():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r".*(?P<well>[A-Z]\d+)_s(?P<site>\d+)_w(?P<channel>\d)",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "2"),
                ("Select the rule criteria", 'and (metadata does channel "1")'),
                ("Name to assign these images", "DAPI"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does channel "2")'),
                ("Name to assign these images", "GFP"),
                ("Select the image type", "Grayscale image"),
                ("Match metadata", "[{'DAPI': 'well', 'GFP': 'well'}]"),
                ("Image set matching method", "Metadata"),
            ],
        ),
        _module_with_records(
            3,
            "Groups",
            [
                ("Do you want to group your images?", "Yes"),
                ("Metadata category", "well"),
            ],
        ),
    ]
    processing_module = _module_with_records(
        4,
        "MeasureImageQuality",
        [
            ("Calculate metrics for which images?", "All loaded images"),
            ("Image count", "1"),
            ("Calculate blur metrics?", "Yes"),
            ("Calculate saturation metrics?", "Yes"),
            ("Calculate intensity metrics?", "Yes"),
            ("Calculate thresholds?", "No"),
        ],
    )

    table = CellProfilerSymbolTable.compile([*setup_modules, processing_module])
    contract = table.contracts_by_module_num[4]

    assert tuple(
        binding.alias for binding in contract.source_bindings.bindings
    ) == ("DAPI", "GFP")
    assert contract.source_bindings.requires_step_input_channel_stack
    assert contract.runtime_artifact_inputs == ()

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_measure_image_quality_all_loaded",
        source_cppipe=Path("source.cppipe"),
        modules=[processing_module],
        skipped_modules=setup_modules,
    )

    assert "source_bindings=LazyStepSourceBindingsConfig(" in generated.code
    assert "enabled=True" in generated.code
    assert "source_bindings=StepSourceBindingsConfig(" not in generated.code
    assert "# CellProfiler artifact inputs: image:DAPI, image:GFP" in generated.code
    assert "VariableComponents.CHANNEL" in generated.code
    assert "group_by=GroupBy.SITE" in generated.code
    assert "input_source=InputSource.PIPELINE_START" in generated.code


def test_codegen_upgrades_pure_2d_runtime_callable_when_step_input_binding_selects_stack():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r".*(?P<well>[A-Z]\d+)_s(?P<site>\d+)_w(?P<channel>\d)",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "DNA"),
                ("Match metadata", "[{'DNA': 'well'}, {'DNA': 'site'}]"),
                ("Image set matching method", "Metadata"),
                ("Select the rule criteria", 'and (metadata does channel "1")'),
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "Actin"),
                ("Match metadata", "[{'Actin': 'well'}, {'Actin': 'site'}]"),
                ("Image set matching method", "Metadata"),
                ("Select the rule criteria", 'and (metadata does channel "2")'),
            ],
        ),
    ]
    processing_module = ModuleBlock(
        name="GrayToColor",
        module_num=3,
        settings={
            "Select the image to be colored red": "Actin",
            "Select the image to be colored blue": "DNA",
            "Name the output image": "Composite",
        },
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_gray_to_color",
        source_cppipe=Path("source.cppipe"),
        modules=[processing_module],
        skipped_modules=setup_modules,
    )

    assert "gray_to_color," in generated.code
    assert "CellProfilerModuleRuntimeBinding" not in generated.code


def test_imagemath_pipeline_start_operands_consume_source_alias_axis():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"^(?P<Plate>.*)_xy(?P<Site>[0-9])_ch(?P<ChannelNumber>[0-9])",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "3"),
                ("Process as 3D?", "Yes"),
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "origDNA"),
                ("Match metadata", "[]"),
                ("Image set matching method", "Order"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "2")'),
                ("Name to assign these images", "origMito"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "1")'),
                ("Name to assign these images", "origMemb"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "0")'),
            ],
        ),
    ]
    processing_module = _module_with_records(
        3,
        "ImageMath",
        [
            ("Operation", "Add"),
            ("Name the output image", "Monolayer"),
            ("Image or measurement?", "Image"),
            ("Select the first image", "origDNA"),
            ("Multiply the first image by", "1.0"),
            ("Image or measurement?", "Image"),
            ("Select the second image", "origMemb"),
            ("Multiply the second image by", "1.0"),
            ("Image or measurement?", "Image"),
            ("Select the third image", "origMito"),
            ("Multiply the third image by", "1.0"),
        ],
    )

    table = CellProfilerSymbolTable.compile([*setup_modules, processing_module])
    bindings = table.contracts_by_module_num[3].source_bindings.bindings

    assert tuple(binding.alias for binding in bindings) == (
        "origDNA",
        "origMemb",
        "origMito",
    )
    assert tuple(binding.participates_in_image_stack for binding in bindings) == (
        True,
        False,
        False,
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_imagemath_sources",
        source_cppipe=Path("source.cppipe"),
        modules=[processing_module],
        skipped_modules=setup_modules,
    )
    step_start = generated.code.index("# CellProfiler artifact outputs: image:Monolayer")
    step_source = generated.code[step_start:]

    assert step_source.count("participates_in_image_stack=False") == 2
    assert "variable_components=[VariableComponents.Z_INDEX]" in step_source
    assert "VariableComponents.SITE" not in step_source
    assert "VariableComponents.CHANNEL" not in step_source


def test_align_source_images_infer_axis_from_selected_source_aliases():
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
                ("Name to assign these images", "OrigStain2"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "2")'),
            ],
        ),
    ]
    align_module = _module_with_records(
        3,
        "Align",
        [
            ("Select the alignment method", "Mutual Information"),
            ("Crop mode", "Keep size"),
            ("Select the first input image", "OrigStain1"),
            ("Name the first output image", "Stain1"),
            ("Select the second input image", "OrigStain2"),
            ("Name the second output image", "Stain2"),
        ],
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_align_sources",
        source_cppipe=Path("source.cppipe"),
        modules=[align_module],
        skipped_modules=setup_modules,
    )
    step_start = generated.code.index('# CellProfiler artifact outputs: image:Stain1')
    step_source = generated.code[step_start:]

    assert "source_bindings=LazyStepSourceBindingsConfig" in step_source
    assert "variable_components=[VariableComponents.CHANNEL]" in step_source
    assert "group_by=GroupBy.SITE" in step_source
    assert "VariableComponents.SITE" not in step_source.split("group_by=")[0]


def test_runtime_image_artifacts_preserve_source_alias_axis_from_output_lineage():
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
                ("Name to assign these images", "OrigStain2"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "2")'),
            ],
        ),
    ]
    align_module = _module_with_records(
        3,
        "Align",
        [
            ("Select the alignment method", "Mutual Information"),
            ("Crop mode", "Keep size"),
            ("Select the first input image", "OrigStain1"),
            ("Name the first output image", "Stain1"),
            ("Select the second input image", "OrigStain2"),
            ("Name the second output image", "Stain2"),
        ],
    )
    colocalization_module = _module_with_records(
        4,
        "MeasureColocalization",
        [
            ("Select images to measure", "Stain1, Stain2"),
            ("Select where to measure correlation", "Both"),
        ],
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_colocalization_sources",
        source_cppipe=Path("source.cppipe"),
        modules=[align_module, colocalization_module],
        skipped_modules=setup_modules,
    )
    step_start = generated.code.index(
        "# CellProfiler artifact outputs: measurements:MeasureColocalization"
    )
    step_source = generated.code[step_start:]
    processing_start = step_source.index("processing_config=LazyProcessingConfig(")
    processing_end = step_source.index("        ),", processing_start)
    processing_config = step_source[processing_start:processing_end]

    assert "variable_components=[VariableComponents.CHANNEL]" in processing_config
    assert "group_by=GroupBy.SITE" in processing_config
    assert "select_images_to_measure" not in step_source


def test_align_source_images_adapt_to_site_axis_from_selected_source_aliases():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"^(?P<Plate>.*)_s(?P<Site>[0-9])",
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
                ("Name to assign these images", "SiteOne"),
                ("Match metadata", "[]"),
                ("Image set matching method", "Order"),
                ("Select the rule criteria", 'and (metadata does Site "1")'),
                ("Assign a name to", "Images matching rules"),
                ("Name to assign these images", "SiteTwo"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does Site "2")'),
            ],
        ),
    ]
    align_module = _module_with_records(
        3,
        "Align",
        [
            ("Select the alignment method", "Mutual Information"),
            ("Crop mode", "Keep size"),
            ("Select the first input image", "SiteOne"),
            ("Name the first output image", "Aligned1"),
            ("Select the second input image", "SiteTwo"),
            ("Name the second output image", "Aligned2"),
        ],
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_align_sites",
        source_cppipe=Path("source.cppipe"),
        modules=[align_module],
        skipped_modules=setup_modules,
    )
    step_start = generated.code.index('# CellProfiler artifact outputs: image:Aligned1')
    step_source = generated.code[step_start:]
    processing_start = step_source.index("processing_config=LazyProcessingConfig(")
    processing_end = step_source.index("        ),", processing_start)
    processing_config = step_source[processing_start:processing_end]

    assert "source_bindings=LazyStepSourceBindingsConfig" in step_source
    assert "variable_components=[VariableComponents.SITE]" in processing_config
    assert "VariableComponents.CHANNEL" not in processing_config.split("group_by=")[0]
    assert "group_by=GroupBy.CHANNEL" in processing_config


def test_source_binding_axis_inference_rejects_incompatible_module_requirements(monkeypatch):
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
                ("Name to assign these images", "OrigStain2"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does ChannelNumber "2")'),
            ],
        ),
    ]
    align_module = _module_with_records(
        3,
        "Align",
        [
            ("Select the alignment method", "Mutual Information"),
            ("Crop mode", "Keep size"),
            ("Select the first input image", "OrigStain1"),
            ("Name the first output image", "Stain1"),
            ("Select the second input image", "OrigStain2"),
            ("Name the second output image", "Stain2"),
        ],
    )
    monkeypatch.setattr(
        AlignModule,
        "required_variable_components",
        (VariableComponents.TIMEPOINT,),
    )

    with pytest.raises(ValueError, match="Align requires variable_components"):
        PipelineGenerator().generate_from_registry(
            pipeline_name="cp_align_incompatible_source_axis",
            source_cppipe=Path("source.cppipe"),
            modules=[align_module],
            skipped_modules=setup_modules,
        )


def test_source_binding_scope_ignores_stack_axes_without_image_stack_anchor():
    source_schema = PipelineImageSchema(
        source_image_stack=SourceImageStackPlan((AllComponents.Z_INDEX,)),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="HelperImage",
                participates_in_image_stack=False,
            ),
        ),
    )
    axis_plan = SourceProcessingAxisPlan.from_schema(
        source_schema,
        source_bindings,
    )
    source_stack_components = (
        source_schema.source_stack_components
        if source_bindings.image_stack_bindings
        else ()
    )

    components = SourceBindingProcessingScope(
        source_bindings,
        source_schema,
        axis_plan,
        source_stack_components,
    ).components()

    assert components.variable_components == ()


def test_codegen_uses_pipeline_start_for_load_images_filter_bindings():
    setup_modules = [
        _module_with_records(
            1,
            "LoadImages",
            [
                ("What type of files are you loading?", "individual images"),
                ("How do you want to load these files?", "Text-Exact match"),
                (
                    "Type the text that these images have in common (case-sensitive)",
                    "Channel2",
                ),
                ("What do you want to call this image in CellProfiler?", "DNA"),
                ("What is the position of this image in each group?", "1"),
                (
                    "Do you want to extract metadata from the file name, "
                    "the subfolder path or both?",
                    "None",
                ),
            ],
        )
    ]
    processing_module = ModuleBlock(
        name="IdentifyPrimaryObjects",
        module_num=2,
        settings={
            "Select the input image": "DNA",
            "Name the primary objects to be identified": "Nuclei",
        },
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_load_images",
        source_cppipe=Path("source.pipeline"),
        modules=[processing_module],
        skipped_modules=setup_modules,
    )

    assert "SourceFilterClause(" in generated.code
    assert "SourceFilterMatchType.CONTAINS" in generated.code
    assert "input_source=InputSource.PIPELINE_START," in generated.code
    assert "group_by=GroupBy.CHANNEL" in generated.code


def test_codegen_preserves_source_timepoint_lineage_for_runtime_artifact_steps():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"^(?P<Specimen>.*)_(?P<Stain>.*)_(?P<FrameNumber>[0-9]*)",
                ),
                ("Select the filtering criteria", 'and (file does contain "GFPHistone")'),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "1"),
                (
                    "Select the rule criteria",
                    'and (file does contain "GFPHistone")',
                ),
                ("Name to assign these images", "OrigGray"),
                ("Select the image type", "Grayscale image"),
            ],
        ),
    ]
    processing_modules = [
        ModuleBlock(
            name="IdentifyPrimaryObjects",
            module_num=3,
            settings={
                "Select the input image": "OrigGray",
                "Name the primary objects to be identified": "Embryos",
            },
        ),
        ModuleBlock(
            name="MeasureObjectSizeShape",
            module_num=4,
            settings={"Select object sets to measure": "Embryos"},
        ),
    ]

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_timepoint_lineage",
        source_cppipe=Path("source.pipeline"),
        modules=processing_modules,
        skipped_modules=setup_modules,
    )

    primary_match = re.search(
        r'name="IdentifyPrimaryObjects".*?'
        r"processing_config=LazyProcessingConfig\(\n(?P<body>.*?)\n        \),",
        generated.code,
        re.S,
    )
    assert primary_match is not None
    primary_config = primary_match.group("body")
    assert "variable_components=[VariableComponents.TIMEPOINT]," in primary_config
    assert "group_by=GroupBy.CHANNEL" in primary_config

    measurement_match = re.search(
        r'name="MeasureObjectSizeShape".*?'
        r"processing_config=LazyProcessingConfig\(\n(?P<body>.*?)\n        \),",
        generated.code,
        re.S,
    )
    assert measurement_match is not None
    measurement_config = measurement_match.group("body")
    assert "VariableComponents.SITE" not in measurement_config
    assert "variable_components=[VariableComponents.TIMEPOINT]," in measurement_config
    assert "group_by=GroupBy.CHANNEL" in measurement_config
    tree = ast.parse(generated.code)
    measurement_steps = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FunctionStep"
        and any(
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "MeasureObjectSizeShape"
            for keyword in node.keywords
        )
    )
    assert len(measurement_steps) == 1
    func_keyword = next(
        keyword.value
        for keyword in measurement_steps[0].keywords
        if keyword.arg == "func"
    )
    public_kwarg_names: set[str] = set()
    if isinstance(func_keyword, ast.Tuple):
        assert isinstance(func_keyword.elts[1], ast.Dict)
        public_kwarg_names = {
            key.value
            for key in func_keyword.elts[1].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    assert "slice_by_slice" not in public_kwarg_names


def test_codegen_keeps_source_binding_channel_out_of_runtime_artifact_scope():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"(?P<well>[A-Z]\d+)_s(?P<site>\d+)_w(?P<channel>\d)",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "2"),
                ("Select the rule criteria", 'and (metadata does channel "1")'),
                ("Name to assign these images", "OrigBlue"),
                ("Select the image type", "Grayscale image"),
                ("Select the rule criteria", 'and (metadata does channel "2")'),
                ("Name to assign these images", "OrigGreen"),
                ("Select the image type", "Grayscale image"),
            ],
        ),
    ]
    processing_modules = [
        ModuleBlock(
            name="IdentifyPrimaryObjects",
            module_num=3,
            settings={
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
        ModuleBlock(
            name="IdentifySecondaryObjects",
            module_num=4,
            settings={
                "Select the input objects": "Nuclei",
                "Select the input image": "OrigGreen",
                "Name the objects to be identified": "Cells",
            },
        ),
    ]

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_mixed_runtime_source_binding",
        source_cppipe=Path("source.pipeline"),
        modules=processing_modules,
        skipped_modules=setup_modules,
    )

    primary_match = re.search(
        r'name="IdentifyPrimaryObjects".*?'
        r"processing_config=LazyProcessingConfig\(\n(?P<body>.*?)\n        \),",
        generated.code,
        re.S,
    )
    assert primary_match is not None
    primary_config = primary_match.group("body")
    assert "VariableComponents.CHANNEL" not in primary_config
    assert "variable_components=[VariableComponents.SITE]," in primary_config
    assert "group_by=GroupBy.CHANNEL" in primary_config

    secondary_match = re.search(
        r'name="IdentifySecondaryObjects".*?'
        r"processing_config=LazyProcessingConfig\(\n(?P<body>.*?)\n        \),",
        generated.code,
        re.S,
    )
    assert secondary_match is not None
    secondary_config = secondary_match.group("body")
    assert "VariableComponents.CHANNEL" not in secondary_config
    assert "variable_components=[VariableComponents.SITE]," in secondary_config
    assert "group_by=GroupBy.CHANNEL" in secondary_config


def test_straightenworms_does_not_declare_step_source_identity_axis():
    setup_modules = [
        _module_with_records(
            1,
            "Metadata",
            [
                ("Metadata extraction method", "Extract from file/folder names"),
                ("Metadata source", "File name"),
                (
                    "Regular expression to extract from file name",
                    r"^(?P<well>[A-Z]\d+)_s(?P<site>\d+)_w(?P<channel>\d)",
                ),
            ],
        ),
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "WormsBinary"),
                ("Match metadata", "[{'WormsBinary': 'well'}, {'WormsBinary': 'site'}]"),
                ("Image set matching method", "Metadata"),
                ("Select the rule criteria", 'and (metadata does channel "1")'),
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "mCherry"),
                ("Match metadata", "[{'mCherry': 'well'}, {'mCherry': 'site'}]"),
                ("Image set matching method", "Metadata"),
                ("Select the rule criteria", 'and (metadata does channel "2")'),
                ("Assign a name to", "Images matching rules"),
                ("Select the image type", "Grayscale image"),
                ("Name to assign these images", "GFP"),
                ("Match metadata", "[{'GFP': 'well'}, {'GFP': 'site'}]"),
                ("Image set matching method", "Metadata"),
                ("Select the rule criteria", 'and (metadata does channel "3")'),
            ],
        ),
    ]
    processing_modules = [
        ModuleBlock(
            name="UntangleWorms",
            module_num=3,
            settings={
                "Select the input image": "WormsBinary",
                "Name the output overlapping worm objects": "OverlappingWorms",
                "Name the output non-overlapping worm objects": "NonOverlappingWorms",
            },
        ),
        _module_with_records(
            4,
            "StraightenWorms",
            [
                ("Select the input untangled worm objects", "NonOverlappingWorms"),
                ("Name the output straightened worm objects", "StraightenedWorms"),
                ("Select an input image to straighten", "mCherry"),
                ("Name the output straightened image", "Straightened_mCherry"),
                ("Select an input image to straighten", "GFP"),
                ("Name the output straightened image", "Straightened_GFP"),
            ],
        ),
    ]

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_straighten_worms_variable_components",
        source_cppipe=Path("source.cppipe"),
        modules=processing_modules,
        skipped_modules=setup_modules,
    )

    step_match = re.search(
        r'name="StraightenWorms",\n(?P<body>.*?)'
        r"processing_config=LazyProcessingConfig",
        generated.code,
        re.S,
    )

    assert step_match is not None


def test_correct_illumination_all_scope_allows_single_channel_schema():
    setup_modules = [
        _module_with_records(
            2,
            "NamesAndTypes",
            [
                ("Assignments count", "1"),
                ("Select the rule criteria", 'and (file does contain "")'),
                ("Name to assign these images", "OrigGreen"),
                ("Select the image type", "Grayscale image"),
                ("Image set matching method", "Order"),
            ],
        ),
    ]
    processing_modules = [
        _module_with_records(
            5,
            "CorrectIlluminationCalculate",
            [
                ("Select the input image", "OrigGreen"),
                ("Name the output image", "IllumGreen"),
                (
                    "Calculate function for each image individually, or based on all images?",
                    "All: First cycle",
                ),
            ],
        ),
    ]

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="single_channel_illumination_all_scope",
        source_cppipe=Path("source.cppipe"),
        modules=processing_modules,
        skipped_modules=setup_modules,
    )

    step_match = re.search(
        r'name="CorrectIlluminationCalculate",\n(?P<body>.*?)\n    \),',
        generated.code,
        re.S,
    )

    assert step_match is not None
    assert "variable_components=[VariableComponents.SITE]" in step_match.group("body")
    assert "group_by=GroupBy.CHANNEL" in step_match.group("body")
    tree = ast.parse(generated.code)
    illumination_steps = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FunctionStep"
        and any(
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "CorrectIlluminationCalculate"
            for keyword in node.keywords
        )
    )
    assert len(illumination_steps) == 1
    func_keyword = next(
        keyword.value
        for keyword in illumination_steps[0].keywords
        if keyword.arg == "func"
    )
    public_kwarg_names: set[str] = set()
    if isinstance(func_keyword, ast.Tuple):
        assert isinstance(func_keyword.elts[1], ast.Dict)
        public_kwarg_names = {
            key.value
            for key in func_keyword.elts[1].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    assert "slice_by_slice" not in public_kwarg_names


def test_compile_image_schema_decodes_legacy_escaped_match_metadata():
    names_and_types_module = _module_with_records(
        2,
        "NamesAndTypes",
        [
            ("Assignments count", "2"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Name to assign these images", "DAPI"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (metadata does illum "DAPI")'),
            ("Name to assign these images", "DAPIillum"),
            ("Select the image type", "Illumination function"),
            (
                "Match metadata",
                "\\x5B{u\\'DAPI\\'\\x3A u\\'folder\\', "
                "u\\'DAPIillum\\'\\x3A u\\'folder_illum\\'}\\x5D",
            ),
            ("Image set matching method", "Metadata"),
        ],
    )

    schema = compile_image_schema([names_and_types_module])

    assert schema.match_plan is not None
    assert schema.match_plan.dimensions[0].field_for_alias("DAPI") == "folder"
    assert (
        schema.match_plan.dimensions[0].field_for_alias("DAPIillum")
        == "folder_illum"
    )


def test_compile_image_schema_preserves_real_names_and_types_block_order():
    names_and_types_module = _module_with_records(
        3,
        "NamesAndTypes",
        [
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "DNA"),
            ("Match metadata", "[{'DNA': 'well'}, {'DNA': 'site'}]"),
            ("Image set matching method", "Metadata"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "Actin"),
            ("Match metadata", "[{'Actin': 'well'}, {'Actin': 'site'}]"),
            ("Image set matching method", "Metadata"),
            ("Select the rule criteria", 'and (metadata does channel "2")'),
        ],
    )

    schema = compile_image_schema([names_and_types_module])

    dna = schema.assignment_for_alias("DNA")
    actin = schema.assignment_for_alias("Actin")
    assert dna is not None
    assert actin is not None
    assert dna.selector.metadata == (
        MetadataSelector("channel", "1"),
    )
    assert actin.selector.metadata == (
        MetadataSelector("channel", "2"),
    )
    assert schema.match_plan is not None
    assert schema.match_plan.dimensions[0].field_for_alias("DNA") == "well"
    assert schema.match_plan.dimensions[0].field_for_alias("Actin") == "well"
    assert schema.match_plan.dimensions[1].field_for_alias("DNA") == "site"
    assert schema.match_plan.dimensions[1].field_for_alias("Actin") == "site"


def test_compile_image_schema_uses_rule_row_alias_over_stale_preamble_alias():
    names_and_types_module = _module_with_records(
        3,
        "NamesAndTypes",
        [
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "DNA"),
            ("Image set matching method", "Order"),
            ("Assignments count", "1"),
            ("Single images count", "0"),
            ("Select the rule criteria", 'and (file does contain "AS_09047_")'),
            ("Name to assign these images", "OrigGreen"),
            ("Name to assign these objects", "Cell"),
            ("Select the image type", "Color image"),
        ],
    )

    schema = compile_image_schema([names_and_types_module])

    assert schema.assignment_for_alias("DNA") is None
    orig_green = schema.assignment_for_alias("OrigGreen")
    assert orig_green is not None
    assert orig_green.image_type == "Color image"
    assert orig_green.origin is SourceBindingOrigin.PIPELINE_START
    assert len(orig_green.selector.filters) == 1
    assert orig_green.selector.filters[0].subject is SourceFilterSubject.FILE
    assert (
        orig_green.selector.filters[0].match_type
        is SourceFilterMatchType.CONTAINS
    )
    assert orig_green.selector.filters[0].value == "AS_09047_"


def test_compile_image_schema_supports_order_based_matching():
    names_and_types_module = _module_with_records(
        3,
        "NamesAndTypes",
        [
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "DNA"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "Actin"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (metadata does channel "2")'),
        ],
    )

    schema = compile_image_schema([names_and_types_module])

    assert schema.match_plan is not None
    assert schema.match_plan.method is SourceBindingMatchMethod.ORDER
    assert schema.match_plan.dimensions == ()
    dna = schema.assignment_for_alias("DNA")
    actin = schema.assignment_for_alias("Actin")
    assert dna is not None
    assert actin is not None
    assert dna.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )
    assert actin.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "2"),
    )


def test_ordered_image_set_axis_separates_sample_and_alias_axes():
    names_and_types_module = _module_with_records(
        3,
        "NamesAndTypes",
        [
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "DNA"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Assign a name to", "Images matching rules"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "Actin"),
            ("Image set matching method", "Order"),
            ("Select the rule criteria", 'and (metadata does channel "2")'),
        ],
    )

    schema = compile_image_schema([names_and_types_module])
    semantics = SourceProcessingComponentSemantics(schema)

    assert semantics.sample_group_component() is AllComponents.SITE
    assert semantics.image_set_components() == (AllComponents.CHANNEL,)
    assert semantics.source_alias_components() == (AllComponents.CHANNEL,)


def test_cellprofiler_image_schema_resolves_legacy_orig_color_aliases():
    schema = compile_image_schema([])

    blue = schema.resolved_assignment_for_alias("OrigBlue")
    green = schema.resolved_assignment_for_alias("OrigGreen")

    assert blue is not None
    assert blue.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )
    assert green is not None
    assert green.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "2"),
    )


def test_generated_pipeline_exposes_pipeline_level_source_schema():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_legacy_aliases",
        source_cppipe=Path("source.cppipe"),
        modules=[
            ModuleBlock(
                name="IdentifyPrimaryObjects",
                module_num=1,
                settings={
                    "Select the input image": "OrigBlue",
                    "Name the primary objects to be identified": "Nuclei",
                },
            )
        ],
    )

    assert isinstance(generated.source_schema, PipelineImageSchema)
    assignment = generated.source_schema.resolved_assignment_for_alias("OrigBlue")

    assert assignment is not None
    assert assignment.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )


def test_generated_runtime_callables_with_non_image_artifacts_are_flexible():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_runtime_artifact_only",
        source_cppipe=Path("source.cppipe"),
        modules=[
            ModuleBlock(
                name="IdentifyPrimaryObjects",
                module_num=1,
                settings={
                    "Select the input image": "OrigBlue",
                    "Name the primary objects to be identified": "Nuclei",
                },
            ),
            ModuleBlock(
                name="IdentifySecondaryObjects",
                module_num=2,
                settings={
                    "Select the input objects": "Nuclei",
                    "Select the input image": "OrigGreen",
                    "Name the objects to be identified": "Cells",
                },
            ),
            ModuleBlock(
                name="IdentifyTertiaryObjects",
                module_num=3,
                settings={
                    "Select the larger identified objects": "Cells",
                    "Select the smaller identified objects": "Nuclei",
                    "Name the tertiary objects to be identified": "Cytoplasm",
                },
            ),
        ],
    )

    assert "identify_tertiary_objects," in generated.code
    assert "CellProfilerModuleRuntimeBinding" not in generated.code
    assert "name=\"IdentifyTertiaryObjects\"," in generated.code
    assert "variable_components=[VariableComponents.SITE]," in generated.code
    assert "group_by=GroupBy.CHANNEL" in generated.code


def test_shape_changing_runtime_images_do_not_recompose_source_alias_stack():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_crop_colocalization",
        source_cppipe=Path("source.cppipe"),
        modules=[
            _module_with_records(
                1,
                "Crop",
                [
                    ("Select the input image", "OrigBlue"),
                    ("Name the output image", "CropBlue"),
                    ("Select the cropping shape", "Rectangle"),
                    ("Select the cropping method", "Coordinates"),
                    ("Remove empty rows and columns?", "Edges"),
                    ("Left and right rectangle positions", "1,10"),
                    ("Top and bottom rectangle positions", "1,10"),
                ],
            ),
            _module_with_records(
                2,
                "Crop",
                [
                    ("Select the input image", "OrigGreen"),
                    ("Name the output image", "CropGreen"),
                    ("Select the cropping shape", "Rectangle"),
                    ("Select the cropping method", "Coordinates"),
                    ("Remove empty rows and columns?", "Edges"),
                    ("Left and right rectangle positions", "1,20"),
                    ("Top and bottom rectangle positions", "1,20"),
                ],
            ),
            ModuleBlock(
                name="IdentifyPrimaryObjects",
                module_num=3,
                settings={
                    "Select the input image": "CropBlue",
                    "Name the primary objects to be identified": "Nuclei",
                },
            ),
            _module_with_records(
                4,
                "MeasureColocalization",
                [
                    ("Select where to measure correlation", "Both"),
                    ("Select objects to measure", "Nuclei"),
                    ("Select images to measure", "CropBlue"),
                    ("Select images to measure", "CropGreen"),
                ],
            ),
        ],
    )

    assert (
        "# Runtime artifact inputs: image:CropBlue, image:CropGreen, object_labels:Nuclei"
        in generated.code
    )
    step_start = generated.code.index(
        "# CellProfiler artifact outputs: measurements:MeasureColocalization"
    )
    step_source = generated.code[step_start:]
    processing_start = step_source.index("processing_config=LazyProcessingConfig(")
    processing_end = step_source.index("        ),", processing_start)
    processing_config = step_source[processing_start:processing_end]

    assert "name=\"MeasureColocalization\"," in generated.code
    assert "variable_components=[VariableComponents.SITE]" in processing_config
    assert "group_by=GroupBy.CHANNEL" in processing_config


def test_compile_image_schema_for_bbbc021_analysis_preserves_real_matching_plan():
    schema = _schema_from_in_tree_cppipe("BBBC021_analysis.cppipe")

    assert set(schema.assignments_by_alias) == {"DAPI", "Actin", "Tubulin"}
    assert set(schema.source_artifacts_by_alias) == {
        "ActinIllum",
        "DAPIillum",
        "TubIllum",
    }
    dapi = schema.assignment_for_alias("DAPI")
    tubulin = schema.assignment_for_alias("Tubulin")
    actin_illum = schema.source_artifact_for_alias("ActinIllum")
    assert dapi is not None
    assert tubulin is not None
    assert actin_illum is not None
    assert dapi.origin is SourceBindingOrigin.STEP_INPUT
    assert dapi.selector.metadata == (
        MetadataSelector("channel", "1"),
    )
    assert tubulin.selector.metadata == (
        MetadataSelector("channel", "4"),
    )
    assert actin_illum.origin is SourceBindingOrigin.PIPELINE_START
    assert actin_illum.selector.metadata == (
        MetadataSelector("illum", "Actin"),
    )
    assert actin_illum.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )

    assert schema.grouping is not None
    assert schema.grouping.metadata_fields == ("folder", "well")
    assert schema.match_plan is not None
    assert schema.match_plan.method is SourceBindingMatchMethod.METADATA
    assert schema.match_plan.dimensions[0].field_for_alias("DAPI") == "folder"
    assert schema.match_plan.dimensions[0].field_for_alias("ActinIllum") == (
        "folder_illum"
    )
    assert schema.match_plan.dimensions[1].field_for_alias("Actin") == "well"
    assert schema.match_plan.dimensions[2].field_for_alias("Tubulin") == "site"
    assert len(schema.metadata_rules) == 3
    assert any(
        rule.source is MetadataSource.FOLDER_NAME for rule in schema.metadata_rules
    )
    assert any(
        rule.source is MetadataSource.FILE_NAME and "(?P<channel>" in rule.pattern
        for rule in schema.metadata_rules
    )
    illum_rule = next(
        rule for rule in schema.metadata_rules if "(?P<illum>" in rule.pattern
    )
    illum_match = re.search(illum_rule.pattern, "fields_IllumDAPI.tif")
    assert illum_match is not None
    assert illum_match.groupdict() == {
        "folder_illum": "fields",
        "illum": "DAPI",
    }
    folder_rule = next(
        rule for rule in schema.metadata_rules if rule.source is MetadataSource.FOLDER_NAME
    )
    folder_match = re.search(folder_rule.pattern, "/tmp/Week1_22123/fields")
    assert folder_match is not None
    assert folder_match.group("folder") == "fields"


def test_compile_image_schema_for_bbbc021_illumination_pipeline():
    schema = _schema_from_in_tree_cppipe("BBBC021_illum.cppipe")

    assert set(schema.assignments_by_alias) == {"DAPI", "Actin", "Tubulin"}
    dapi = schema.assignment_for_alias("DAPI")
    tubulin = schema.assignment_for_alias("Tubulin")
    assert dapi is not None
    assert tubulin is not None
    assert dapi.origin is SourceBindingOrigin.STEP_INPUT
    assert dapi.selector.metadata == (
        MetadataSelector("channel", "1"),
    )
    assert tubulin.selector.metadata == (
        MetadataSelector("channel", "4"),
    )

    assert schema.grouping is not None
    assert schema.grouping.metadata_fields == ("folder",)
    assert schema.match_plan is not None
    assert schema.match_plan.method is SourceBindingMatchMethod.METADATA
    assert schema.match_plan.dimensions[0].field_for_alias("DAPI") == "folder"
    assert schema.match_plan.dimensions[1].field_for_alias("Actin") == "well"
    assert schema.match_plan.dimensions[2].field_for_alias("Tubulin") == "site"
    assert schema.match_plan.dimensions[0].field_for_alias("ActinIllum") is None
    assert len(schema.metadata_rules) == 2
    assert {rule.source for rule in schema.metadata_rules} == {
        MetadataSource.FILE_NAME,
        MetadataSource.FOLDER_NAME,
    }
