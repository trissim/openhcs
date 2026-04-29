"""CellProfiler Crop setting and artifact lowering."""

from __future__ import annotations

from typing import Any

from benchmark.cellprofiler_semantics.crop import (
    CropShape,
    CroppingMethod,
    RemovalMethod,
)
from openhcs.core.artifacts import CROP_MASK_ARTIFACT_SIDECAR

from .parser import ModuleBlock
from .setting_names import optional_setting_value, required_setting_value
from .settings_binder import SettingsBinder

CROP_SHAPE_SETTING = "Select the cropping shape"
CROP_METHOD_SETTING = "Select the cropping method"
CROP_REMOVAL_SETTING = "Remove empty rows and columns?"
CROP_INPUT_IMAGE_SETTING = "Select the input image"
CROP_OUTPUT_IMAGE_SETTING = "Name the output image"
CROP_MASK_IMAGE_SETTING = "Select the masking image"
CROP_PREVIOUS_IMAGE_SETTING = "Select the image with a cropping mask"
CROP_OBJECTS_SETTING = "Select the objects"
CROP_LEFT_RIGHT_SETTING = "Left and right rectangle positions"
CROP_TOP_BOTTOM_SETTING = "Top and bottom rectangle positions"
CROP_ELLIPSE_CENTER_SETTING = "Coordinates of ellipse center"
CROP_ELLIPSE_X_RADIUS_SETTING = "Ellipse radius, X direction"
CROP_ELLIPSE_Y_RADIUS_SETTING = "Ellipse radius, Y direction"

_NO_SYMBOL_LITERALS = frozenset({"", "none", "do not use", "leave this black"})


def crop_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed Crop kwargs from typed .cppipe settings."""
    return _without_none_values(
        {
            "crop_shape": crop_shape(module).value,
            "cropping_method": crop_method(module).value,
            "removal_method": crop_removal_method(module).value,
            "left_right_rectangle_positions": _typed_setting(
                module,
                binder,
                CROP_LEFT_RIGHT_SETTING,
            ),
            "top_bottom_rectangle_positions": _typed_setting(
                module,
                binder,
                CROP_TOP_BOTTOM_SETTING,
            ),
            "ellipse_center": _typed_setting(
                module,
                binder,
                CROP_ELLIPSE_CENTER_SETTING,
            ),
            "ellipse_x_radius": _typed_setting(
                module,
                binder,
                CROP_ELLIPSE_X_RADIUS_SETTING,
            ),
            "ellipse_y_radius": _typed_setting(
                module,
                binder,
                CROP_ELLIPSE_Y_RADIUS_SETTING,
            ),
        }
    )


def crop_shape(module: ModuleBlock) -> CropShape:
    """Return the declared Crop shape mode."""
    return CropShape(
        optional_setting_value(module, CROP_SHAPE_SETTING)
        or CropShape.RECTANGLE.value
    )


def crop_method(module: ModuleBlock) -> CroppingMethod:
    """Return the declared Crop coordinate/input method."""
    return CroppingMethod(
        optional_setting_value(module, CROP_METHOD_SETTING)
        or CroppingMethod.COORDINATES.value
    )


def crop_removal_method(module: ModuleBlock) -> RemovalMethod:
    """Return the declared Crop row/column removal mode."""
    return RemovalMethod(
        optional_setting_value(module, CROP_REMOVAL_SETTING)
        or RemovalMethod.NO.value
    )


def crop_input_image_name(module: ModuleBlock) -> str:
    """Return the current image consumed by Crop."""
    return required_setting_value(module, CROP_INPUT_IMAGE_SETTING)


def crop_output_image_name(module: ModuleBlock) -> str:
    """Return the cropped image produced by Crop."""
    return required_setting_value(module, CROP_OUTPUT_IMAGE_SETTING)


def crop_previous_mask_artifact_name(module: ModuleBlock) -> str | None:
    """Return the prior Crop crop-mask artifact consumed by this module."""
    previous_image_name = _optional_symbol(module, CROP_PREVIOUS_IMAGE_SETTING)
    if previous_image_name is None:
        return None
    return CROP_MASK_ARTIFACT_SIDECAR.name_for(previous_image_name)


def crop_mask_image_name(module: ModuleBlock) -> str | None:
    """Return the binary mask image consumed by image-mask Crop mode."""
    return _optional_symbol(module, CROP_MASK_IMAGE_SETTING)


def crop_objects_name(module: ModuleBlock) -> str | None:
    """Return the object-label set consumed by object-mask Crop mode."""
    return _optional_symbol(module, CROP_OBJECTS_SETTING)


def _typed_setting(
    module: ModuleBlock,
    binder: SettingsBinder,
    setting_name: str,
) -> Any:
    value = optional_setting_value(module, setting_name)
    if value is None:
        return None
    return binder.parse_value(setting_name, value)


def _optional_symbol(
    module: ModuleBlock,
    setting_name: str,
) -> str | None:
    value = optional_setting_value(module, setting_name)
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in _NO_SYMBOL_LITERALS:
        return None
    return normalized


def _without_none_values(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }
