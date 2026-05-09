"""Declarative RelateObjects CellProfiler setting contracts."""

from __future__ import annotations

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.relateobjects import DistanceMethod
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily


RELATE_OBJECTS_DISTANCE_SETTING = SettingNameFamily("Calculate child-parent distances?")
RELATE_OBJECTS_PER_PARENT_MEANS_SETTING = SettingNameFamily(
    "Calculate per-parent means for all child measurements?"
)
RELATE_OBJECTS_SAVE_CHILDREN_SETTING = SettingNameFamily(
    "Do you want to save the children with parents as a new object set?"
)


def parse_relate_objects_distance_method(value: str) -> str:
    """Return the declared RelateObjects distance calculation mode."""
    return _coerce_function_enum(DistanceMethod, value).value
