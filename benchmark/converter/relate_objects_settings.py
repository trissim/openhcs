"""Compatibility aliases for RelateObjects setting contracts."""

from openhcs.interop.cellprofiler.relate_objects_settings import (
    RELATE_OBJECTS_DISTANCE_SETTING,
    RELATE_OBJECTS_PER_PARENT_MEANS_SETTING,
    RELATE_OBJECTS_SAVE_CHILDREN_SETTING,
    RelateObjectsDistanceMethod,
    parse_relate_objects_distance_method,
)

__all__ = (
    "RELATE_OBJECTS_DISTANCE_SETTING",
    "RELATE_OBJECTS_PER_PARENT_MEANS_SETTING",
    "RELATE_OBJECTS_SAVE_CHILDREN_SETTING",
    "RelateObjectsDistanceMethod",
    "parse_relate_objects_distance_method",
)
