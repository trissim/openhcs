import re
from pathlib import Path

from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_generator import PipelineGenerator
from openhcs.interop.cellprofiler.source_schema import compile_image_schema
from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbolTable
from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import ImagePlaneSource, PipelineImageSchema
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataSource,
    MetadataSelector,
    SourceBindingMatchMethod,
    SourceFilterMatchType,
    SourceBindingOrigin,
    SourceFilterSubject,
)


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
        ArtifactKind.OBJECT_LABELS,
    )

    assert source_artifact is not None
    assert source_artifact.artifact_kind is ArtifactKind.OBJECT_LABELS
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
    bindings = contract.source_bindings.groups[0].bindings

    assert bindings[0].alias == "DAPI"
    assert bindings[0].selector.metadata == (
        MetadataSelector("channel", "1"),
    )
    assert bindings[1].alias == "DAPIillum"
    assert bindings[1].origin is SourceBindingOrigin.PIPELINE_START
    assert bindings[1].selector.metadata == (
        MetadataSelector("illum", "DAPI"),
    )
    assert contract.source_bindings.match_plan is not None

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cp_setup_schema",
        source_cppipe=Path("source.cppipe"),
        modules=[processing_module],
        skipped_modules=setup_modules,
    )

    assert "MetadataSelector('channel', '1')" in generated.code
    assert "MetadataSelector('illum', 'DAPI')" in generated.code
    assert "MetadataExtractionRule(" in generated.code
    assert "SourceBindingMatchPlan(" in generated.code
    assert "SourceBindingOrigin.PIPELINE_START" in generated.code
    assert "input_source=InputSource.PIPELINE_START" in generated.code
    assert "variable_components=[VariableComponents.CHANNEL]" in generated.code
    assert "group_by=GroupBy.SITE" in generated.code


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
    assert "variable_components=[VariableComponents.SITE]," in generated.code
    assert "group_by=GroupBy.NONE," in generated.code


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
    assert (
        "variable_components=[VariableComponents.SITE, VariableComponents.TIMEPOINT]"
        in primary_config
    )
    assert "group_by=GroupBy.NONE," in primary_config

    measurement_match = re.search(
        r'name="MeasureObjectSizeShape".*?'
        r"processing_config=LazyProcessingConfig\(\n(?P<body>.*?)\n        \),",
        generated.code,
        re.S,
    )
    assert measurement_match is not None
    measurement_config = measurement_match.group("body")
    assert "variable_components=[VariableComponents.TIMEPOINT]," in measurement_config
    assert "VariableComponents.SITE" not in measurement_config
    assert "group_by=GroupBy.SITE," in measurement_config


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
    assert "group_by=GroupBy.NONE," in primary_config

    secondary_match = re.search(
        r'name="IdentifySecondaryObjects".*?'
        r"processing_config=LazyProcessingConfig\(\n(?P<body>.*?)\n        \),",
        generated.code,
        re.S,
    )
    assert secondary_match is not None
    secondary_config = secondary_match.group("body")
    assert "VariableComponents.CHANNEL" not in secondary_config
    assert "variable_components=[]," in secondary_config
    assert "group_by=GroupBy.SITE," in secondary_config


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
    assert schema.assignment_for_alias("DNA") is not None
    assert schema.assignment_for_alias("Actin") is not None


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
    assert "variable_components=[]," in generated.code
    assert "group_by=GroupBy.SITE," in generated.code


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
