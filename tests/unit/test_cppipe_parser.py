from pathlib import Path

from benchmark.converter.parser import CPPipeParser
from openhcs.constants import Backend


class _FakeFileManager:
    def __init__(self, content: str) -> None:
        self.content = content
        self.loaded: list[tuple[str, str]] = []

    def load(self, path: str, backend: str) -> str:
        self.loaded.append((path, backend))
        return self.content


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
                "    :or (file does contain \"\")",
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
