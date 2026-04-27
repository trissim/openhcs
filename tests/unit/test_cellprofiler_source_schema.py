from pathlib import Path

from benchmark.converter.parser import CPPipeParser, ModuleBlock, ModuleSetting
from benchmark.converter.pipeline_generator import PipelineGenerator
from benchmark.converter.source_schema import compile_image_schema
from benchmark.converter.symbol_table import CellProfilerSymbolTable
from openhcs.constants.constants import AllComponents
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataSource,
    MetadataSelector,
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
    assert dapi.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )

    illumination = schema.assignment_for_alias("DAPIillum")
    assert illumination is not None
    assert illumination.origin is SourceBindingOrigin.PIPELINE_START
    assert illumination.selector.metadata == (
        MetadataSelector("illum", "DAPI"),
    )

    assert schema.grouping is not None
    assert schema.grouping.metadata_fields == ("folder", "well")
    assert schema.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert schema.metadata_rules[0].filters[0].subject is SourceFilterSubject.FILE
    assert (
        schema.metadata_rules[0].filters[0].match_type
        is SourceFilterMatchType.CONTAINS
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
    assert bindings[0].selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
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

    assert "ComponentSelector(AllComponents.CHANNEL, '1')" in generated.code
    assert "MetadataSelector('illum', 'DAPI')" in generated.code
    assert "MetadataExtractionRule(" in generated.code
    assert "SourceBindingOrigin.PIPELINE_START" in generated.code
