"""Compatibility aliases for CellProfiler setting lookup helpers."""

from openhcs.interop.cellprofiler.setting_names import (
    IMAGE_MEASUREMENT_SETTING,
    OBJECT_MEASUREMENT_SETTING,
    SettingNameFamily,
    block_setting_value,
    block_setting_value_by_prefix,
    decode_cellprofiler_setting_literal,
    is_blank_symbol_name,
    normalized_symbol_name,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_name_matches,
    setting_name_startswith,
    setting_names,
    setting_values,
    split_symbol_names,
)

__all__ = (
    "IMAGE_MEASUREMENT_SETTING",
    "OBJECT_MEASUREMENT_SETTING",
    "SettingNameFamily",
    "block_setting_value",
    "block_setting_value_by_prefix",
    "decode_cellprofiler_setting_literal",
    "is_blank_symbol_name",
    "normalized_symbol_name",
    "optional_setting_value",
    "repeating_setting_blocks",
    "required_setting_value",
    "setting_name_matches",
    "setting_name_startswith",
    "setting_names",
    "setting_values",
    "split_symbol_names",
)
