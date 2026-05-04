"""Typed lowering for CellProfiler MaskObjects settings."""

from __future__ import annotations

from enum import Enum

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.maskobjects import (
    NumberingChoice,
    OverlapHandling,
)

from .settings_binder import (
    SettingParser,
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)


def _enum_literal(enum_type: type[Enum]) -> SettingParser:
    def parse(value: str) -> str:
        return _coerce_function_enum(enum_type, value).value

    return parse


MASK_OBJECTS_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Handling of objects that are partially masked",
        "overlap_handling",
        _enum_literal(OverlapHandling),
    ),
    SettingToKeywordBinding(
        "Fraction of object that must overlap",
        "overlap_fraction",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding(
        "Numbering of resulting objects",
        "numbering",
        _enum_literal(NumberingChoice),
    ),
    SettingToKeywordBinding(
        "Invert the mask?",
        "invert_mask",
        parse_cellprofiler_bool,
    ),
)
