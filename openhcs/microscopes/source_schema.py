"""Filename parser for OpenHCS source-schema virtual workspaces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openhcs.microscopes.microscope_interfaces import FilenameParser


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
    def can_parse(cls, filename: str | Any) -> bool:
        return cls._pattern.match(Path(str(filename)).name) is not None

    def parse_filename(self, filename: str | Any) -> dict[str, Any] | None:
        match = self._pattern.match(Path(str(filename)).name)
        if match is None:
            return None
        values = match.groupdict()
        return {
            "well": values["well"],
            "site": _parse_component(values["site"]),
            "channel": _parse_component(values["channel"]),
            "z_index": _parse_component(values["z_index"]),
            "timepoint": _parse_component(values["timepoint"]),
            "extension": values["extension"],
        }

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
        well = _required_component(component_values, "well")
        site = _component_token(component_values.get("site", 1), site_padding)
        channel = _component_token(component_values.get("channel", 1), 0)
        z_index = _component_token(component_values.get("z_index", 1), z_padding)
        timepoint = _component_token(
            component_values.get("timepoint", 1),
            timepoint_padding,
        )
        return f"{well}_s{site}_w{channel}_z{z_index}_t{timepoint}{extension}"


def _parse_component(value: str) -> int | str | None:
    if "{" in value:
        return None
    return int(value) if value.isdecimal() else value


def _required_component(component_values: dict[str, Any], name: str) -> str:
    value = component_values.get(name)
    if value is None or value == "":
        raise ValueError(f"{name!r} component cannot be empty or None.")
    return str(value)


def _component_token(value: Any, padding: int) -> str:
    if isinstance(value, str):
        return value
    return f"{int(value):0{padding}d}" if padding else str(int(value))
