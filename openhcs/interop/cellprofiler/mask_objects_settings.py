"""Typed lowering for CellProfiler MaskObjects settings."""

from __future__ import annotations

from enum import Enum

from openhcs.interop.cellprofiler.settings_binder import (
    SettingParser,
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)


class MaskObjectsOverlapHandling(Enum):
    """CellProfiler MaskObjects overlap handling mode."""

    MASK = "keep_overlapping_region"
    KEEP = "keep"
    REMOVE = "remove"
    REMOVE_PERCENTAGE = "remove_depending_on_overlap"


class MaskObjectsNumberingChoice(Enum):
    """CellProfiler MaskObjects output label numbering mode."""

    RENUMBER = "renumber"
    RETAIN = "retain"


def _enum_literal(enum_type: type[Enum]) -> SettingParser:
    def parse(value: str) -> str:
        member = coerce_cellprofiler_enum(enum_type, value)
        if not isinstance(member.value, str):
            raise TypeError(f"{enum_type.__name__}.{member.name} must have a string value.")
        return member.value

    return parse


MASK_OBJECTS_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Handling of objects that are partially masked",
        "overlap_handling",
        _enum_literal(MaskObjectsOverlapHandling),
    ),
    SettingToKeywordBinding(
        "Fraction of object that must overlap",
        "overlap_fraction",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding(
        "Numbering of resulting objects",
        "numbering",
        _enum_literal(MaskObjectsNumberingChoice),
    ),
    SettingToKeywordBinding(
        "Invert the mask?",
        "invert_mask",
        parse_cellprofiler_bool,
    ),
)
