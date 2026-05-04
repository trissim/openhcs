"""Compatibility aliases for CellProfiler setting binding."""

from openhcs.interop.cellprofiler.settings_binder import (
    BoundParameter,
    SettingParser,
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)

__all__ = (
    "BoundParameter",
    "SettingParser",
    "SettingToKeywordBinding",
    "SettingsBinder",
    "normalize_cellprofiler_setting_name",
    "parse_cellprofiler_bool",
    "parse_cellprofiler_float",
    "parse_cellprofiler_int",
)
