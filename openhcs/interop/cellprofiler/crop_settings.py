"""CellProfiler Crop setting, enum, and artifact lowering semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.artifacts import CROP_MASK_ARTIFACT_SIDECAR
from openhcs.core.public_api import declared_public_names

from .parser import ModuleBlock
from .setting_names import (
    OptionalSettingSymbol,
    optional_setting_value,
    required_setting_value,
)
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


class CropShape(str, Enum):
    """Closed CellProfiler Crop shape modes."""

    RECTANGLE = "Rectangle"
    ELLIPSE = "Ellipse"
    IMAGE = "Image"
    OBJECTS = "Objects"
    CROPPING = "Previous cropping"


class CroppingMethod(str, Enum):
    """Closed CellProfiler interactive/coordinate crop modes."""

    COORDINATES = "Coordinates"
    MOUSE = "Mouse"

    @property
    def is_coordinate_based(self) -> bool:
        """Whether the crop geometry is fully represented by stored settings."""
        return self is type(self).COORDINATES


class RemovalMethod(str, Enum):
    """Closed CellProfiler row/column removal modes."""

    NO = "No"
    EDGES = "Edges"
    ALL = "All"

    @property
    def removes_empty_rows_or_columns(self) -> bool:
        """Whether the image shape is reduced to the retained crop extent."""
        return self is not type(self).NO

    @property
    def removes_internal_empty_rows_or_columns(self) -> bool:
        """Whether all empty retained rows/columns are removed, not just edges."""
        return self is type(self).ALL


@dataclass(frozen=True, slots=True)
class CropTypedSettingValue:
    """A typed Crop setting value parsed through the declared binder."""

    module: ModuleBlock
    binder: SettingsBinder
    setting_name: str

    @property
    def value(self) -> Any:
        raw_value = optional_setting_value(self.module, self.setting_name)
        if raw_value is None:
            return None
        return self.binder.parse_value(self.setting_name, raw_value)


@dataclass(frozen=True, slots=True)
class CropPreviousMaskArtifact:
    """Prior Crop mask sidecar selected by a Crop module."""

    module: ModuleBlock

    @property
    def name(self) -> str | None:
        previous_image_name = OptionalSettingSymbol(
            self.module,
            CROP_PREVIOUS_IMAGE_SETTING,
        ).value
        if previous_image_name is None:
            return None
        return CROP_MASK_ARTIFACT_SIDECAR.name_for(previous_image_name)


@dataclass(frozen=True, slots=True)
class CropBoundKwargs:
    """Authoritative Crop settings projection into function kwargs."""

    module: ModuleBlock
    binder: SettingsBinder

    def setting_value(self, setting_name: str) -> Any:
        """Return a typed optional setting value."""
        return CropTypedSettingValue(self.module, self.binder, setting_name).value

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Return the non-empty function kwargs declared by Crop settings."""
        return tuple(
            (key, value)
            for key, value in (
                ("crop_shape", crop_shape(self.module).value),
                ("cropping_method", crop_method(self.module).value),
                ("removal_method", crop_removal_method(self.module).value),
                (
                    "left_right_rectangle_positions",
                    self.setting_value(CROP_LEFT_RIGHT_SETTING),
                ),
                (
                    "top_bottom_rectangle_positions",
                    self.setting_value(CROP_TOP_BOTTOM_SETTING),
                ),
                ("ellipse_center", self.setting_value(CROP_ELLIPSE_CENTER_SETTING)),
                (
                    "ellipse_x_radius",
                    self.setting_value(CROP_ELLIPSE_X_RADIUS_SETTING),
                ),
                (
                    "ellipse_y_radius",
                    self.setting_value(CROP_ELLIPSE_Y_RADIUS_SETTING),
                ),
            )
            if value is not None
        )

    @property
    def values(self) -> dict[str, Any]:
        """Return kwargs accepted by the CellProfiler-compatible Crop function."""
        return dict(self.items())


def crop_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed Crop kwargs from typed .cppipe settings."""
    return CropBoundKwargs(module, binder).values


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
    return CropPreviousMaskArtifact(module).name


def crop_mask_image_name(module: ModuleBlock) -> str | None:
    """Return the binary mask image consumed by image-mask Crop mode."""
    return OptionalSettingSymbol(module, CROP_MASK_IMAGE_SETTING).value


def crop_objects_name(module: ModuleBlock) -> str | None:
    """Return the object-label set consumed by object-mask Crop mode."""
    return OptionalSettingSymbol(module, CROP_OBJECTS_SETTING).value


__all__ = declared_public_names(
    globals(),
    constant_prefixes=("CROP_",),
    excluded_names=("CROP_MASK_ARTIFACT_SIDECAR",),
)
