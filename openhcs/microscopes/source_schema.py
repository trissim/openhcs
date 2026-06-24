"""Filename parser for OpenHCS source-schema virtual workspaces."""

from __future__ import annotations

import re
from pathlib import Path

from openhcs.core.components.parser_metaprogramming import (
    format_filename_component,
    require_filename_component,
)
from openhcs.microscopes.microscope_interfaces import (
    FilenameParseResult,
    FilenameParser,
)


class SourceSchemaFilenameParser(FilenameParser):
    """Parser for normalized filenames emitted by source-schema materialization."""

    _pattern = re.compile(
        r"^(?P<well>[^_]+)"
        r"_s(?P<site>[^_]+)"
        r"_w(?P<channel>[^_]+)"
        r"_z(?P<z_index>[^_]+)"
        r"_t(?P<timepoint>[^_.]+)"
        r"(?:_[^.]*)?"
        r"(?P<extension>(?:\.\w+)+)$"
    )

    def __init__(self, filemanager=None, pattern_format=None):
        super().__init__()
        self.filemanager = filemanager
        self.pattern_format = pattern_format

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        return cls._pattern.match(Path(str(filename)).name) is not None

    def parse_filename(self, filename: str) -> FilenameParseResult | None:
        match = self._pattern.match(Path(str(filename)).name)
        if match is None:
            return None
        values = match.groupdict()
        return FilenameParseResult({
            "well": values["well"],
            "site": _parse_component(values["site"]),
            "channel": _parse_component(values["channel"]),
            "z_index": _parse_component(values["z_index"]),
            "timepoint": _parse_component(values["timepoint"]),
            "extension": values["extension"],
        })

    def extract_component_coordinates(self, component_value: str) -> tuple[str, str]:
        match = re.match(r"^([A-Za-z]+)([0-9]+)$", component_value)
        if match is not None:
            return match.group(1), match.group(2)
        return component_value, ""

    def construct_filename(
        self,
        extension: str = ".tif",
        site_padding: int = 3,
        z_padding: int = 3,
        timepoint_padding: int = 3,
        **component_values,
    ) -> str:
        well = require_filename_component(component_values, "well")
        site = format_filename_component(require_filename_component(component_values, "site"), site_padding)
        channel = format_filename_component(require_filename_component(component_values, "channel"))
        z_index = format_filename_component(require_filename_component(component_values, "z_index"), z_padding)
        timepoint = format_filename_component(require_filename_component(component_values, "timepoint"), timepoint_padding)
        return f"{well}_s{site}_w{channel}_z{z_index}_t{timepoint}{extension}"


def _parse_component(value: str) -> int | str | None:
    if "{" in value:
        return None
    return int(value) if value.isdecimal() else value
