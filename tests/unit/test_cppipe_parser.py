from dataclasses import fields
from pathlib import Path

import pytest

from openhcs.constants import Backend
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    decode_cellprofiler_setting_literal,
)
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting


class _FakeFileManager:
    def __init__(self, content: str) -> None:
        self.content = content
        self.loaded: list[tuple[str, str]] = []

    def load(self, path: str, backend: str) -> str:
        self.loaded.append((path, backend))
        return self.content


def test_module_setting_canonicalizes_v3_metadata_backreferences() -> None:
    setting = ModuleSetting("Enter single file name", r"\\\\g<folder>_output")

    assert setting.value == r"\g<folder>_output"
    assert decode_cellprofiler_setting_literal(setting.value) == setting.value


def test_module_block_derives_read_only_settings_from_ordered_records() -> None:
    module = ModuleBlock(
        name="Example",
        module_num=1,
        setting_records=[
            ModuleSetting("Repeated", "first"),
            ModuleSetting("Other", "value"),
            ModuleSetting("Repeated", "last"),
        ],
    )

    assert "settings" not in {field.name for field in fields(ModuleBlock)}
    assert module.get_setting_values("Repeated") == ("first", "last")
    assert module.settings == {"Repeated": "last", "Other": "value"}
    with pytest.raises(TypeError):
        module.settings["Repeated"] = "mutated"  # type: ignore[index]

    module.setting_records.append(ModuleSetting("Repeated", "new last"))

    assert module.settings["Repeated"] == "new last"


def test_cppipe_parser_ignores_legacy_empty_setting_labels(tmp_path: Path) -> None:
    cppipe = tmp_path / "legacy_empty_settings.pipeline"
    cppipe.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Images:[module_num:1|enabled:True]",
                "    :",
                "    Filter based on rules:No",
                "Metadata:[module_num:2|enabled:True]",
                '    :or (file does contain "")',
                "    Extract metadata?:Yes",
            )
        )
    )

    modules = CPPipeParser(cppipe).parse()

    assert [module.name for module in modules] == ["Images", "Metadata"]
    assert modules[0].settings == {"Filter based on rules": "No"}
    assert modules[1].settings == {"Extract metadata?": "Yes"}


def test_cppipe_parser_preserves_indented_hash_prefixed_setting_names(
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "hash_prefixed_setting.cppipe"
    cppipe.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "# This comment should still be ignored",
                "IdentifyPrimaryObjects:[module_num:7|enabled:True]",
                "    Thresholding method:RobustBackground",
                "    # of deviations:0.75",
            )
        )
    )

    modules = CPPipeParser(cppipe).parse()

    assert modules[0].settings["# of deviations"] == "0.75"
    assert modules[0].get_setting_values("# of deviations") == ("0.75",)


def test_cppipe_parser_can_read_through_explicit_filemanager() -> None:
    content = "\n".join(
        (
            "CellProfiler Pipeline: http://www.cellprofiler.org",
            "Images:[module_num:1|enabled:True]",
            "    Filter based on rules:No",
        )
    )
    filemanager = _FakeFileManager(content)

    modules = CPPipeParser().parse(
        Path("/virtual/pipeline.cppipe"),
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert [module.name for module in modules] == ["Images"]
    assert filemanager.loaded == [("/virtual/pipeline.cppipe", "memory")]


def test_cppipe_parser_keeps_example_fly_image_plane_rows_out_of_last_module() -> None:
    cppipe = (
        Path(__file__).resolve().parents[2]
        / "benchmark/native_refs/official30_scoped_rows"
        / "ExampleFly_ExampleFlyURL_wells_include_first1"
        / "native_cellprofiler_headless/ExampleFlyURL.cppipe"
    )
    parser = CPPipeParser(cppipe)

    modules = parser.parse()

    spreadsheet = modules[-1]
    assert spreadsheet.name == "ExportToSpreadsheet"
    assert len(spreadsheet.setting_records) == 33
    assert spreadsheet.setting_records[-1].name == (
        "Use the object name for the file name?"
    )
    assert spreadsheet.get_setting_values("Data to export") == (
        "Image",
        "Nuclei",
        "Cells",
        "Cytoplasm",
    )
    assert len(parser.image_plane_sources) == 9
    assert parser.image_plane_sources[0].uri.endswith("01_POS002_D.TIF")
    assert "image_plane_sources" not in spreadsheet.metadata
