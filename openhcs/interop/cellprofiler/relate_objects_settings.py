"""Declarative RelateObjects CellProfiler setting contracts."""

from __future__ import annotations

from enum import Enum

from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


class RelateObjectsDistanceMethod(Enum):
    """CellProfiler RelateObjects child-parent distance calculation mode."""

    NONE = ("none", False, False)
    CENTROID = ("centroid", True, False)
    MINIMUM = ("minimum", False, True)
    BOTH = ("both", True, True)

    def __init__(
        self,
        label: str,
        calculates_centroid_distance: bool,
        calculates_minimum_distance: bool,
    ) -> None:
        self._value_ = label
        self._calculates_centroid_distance = calculates_centroid_distance
        self._calculates_minimum_distance = calculates_minimum_distance

    @property
    def calculates_centroid_distance(self) -> bool:
        return self._calculates_centroid_distance

    @property
    def calculates_minimum_distance(self) -> bool:
        return self._calculates_minimum_distance


RELATE_OBJECTS_DISTANCE_SETTING = SettingNameFamily("Calculate child-parent distances?")
RELATE_OBJECTS_PARENT_OBJECTS_SETTING = SettingNameFamily(
    "Select the parent objects",
    aliases=("Parent objects",),
)
RELATE_OBJECTS_CHILD_OBJECTS_SETTING = SettingNameFamily(
    "Select the child objects",
    aliases=("Child objects",),
)
RELATE_OBJECTS_PER_PARENT_MEANS_SETTING = SettingNameFamily(
    "Calculate per-parent means for all child measurements?"
)
RELATE_OBJECTS_SAVE_CHILDREN_SETTING = SettingNameFamily(
    "Do you want to save the children with parents as a new object set?"
)


def parse_relate_objects_distance_method(value: str) -> str:
    """Return the declared RelateObjects distance calculation mode."""
    return coerce_cellprofiler_enum(RelateObjectsDistanceMethod, value).value
