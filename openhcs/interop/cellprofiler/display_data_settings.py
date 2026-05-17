"""Typed lowering for CellProfiler DisplayDataOnImage settings."""

from __future__ import annotations

from typing import Any

from openhcs.core.public_api import declared_public_names

from .parser import ModuleBlock
from .setting_names import SettingNameFamily, optional_setting_value


DISPLAY_OBJECT_OR_IMAGE_SETTING = SettingNameFamily(
    "Display object or image measurements?",
)
MEASUREMENT_TO_DISPLAY_SETTING = SettingNameFamily(
    "Measurement to display",
)


def display_data_on_image_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return kwargs needed to bind DisplayDataOnImage runtime measurements."""

    measurement_feature = optional_setting_value(module, MEASUREMENT_TO_DISPLAY_SETTING)
    if measurement_feature is None:
        raise ValueError("DisplayDataOnImage requires a measurement feature.")
    return {
        "objects_or_image": optional_setting_value(
            module,
            DISPLAY_OBJECT_OR_IMAGE_SETTING,
        )
        or "Object",
        "measurement_feature": measurement_feature,
    }


__all__ = declared_public_names(
    globals(),
    constant_prefixes=("DISPLAY_", "MEASUREMENT_"),
)
