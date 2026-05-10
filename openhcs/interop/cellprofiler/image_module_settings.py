"""CellProfiler image-module setting semantics."""

from __future__ import annotations

from enum import Enum


class ImageQualityThresholdMethod(Enum):
    """Threshold algorithms exposed by MeasureImageQuality settings."""

    OTSU = "otsu"
    LI = "li"
    TRIANGLE = "triangle"
    ISODATA = "isodata"
    MINIMUM = "minimum"
    MEAN = "mean"
    YEN = "yen"


class MaskImageSource(Enum):
    """Mask source domains exposed by MaskImage settings."""

    OBJECTS = "objects"
    IMAGE = "image"


class RescaleIntensityAutomaticHigh(Enum):
    """Automatic upper-bound policies exposed by RescaleIntensity settings."""

    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class RescaleIntensityAutomaticLow(Enum):
    """Automatic lower-bound policies exposed by RescaleIntensity settings."""

    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class RescaleIntensityMethod(Enum):
    """RescaleIntensity method literals exposed by CellProfiler settings."""

    STRETCH = "stretch"
    MANUAL_INPUT_RANGE = "manual_input_range"
    MANUAL_IO_RANGE = "manual_io_range"
    DIVIDE_BY_IMAGE_MINIMUM = "divide_by_image_minimum"
    DIVIDE_BY_IMAGE_MAXIMUM = "divide_by_image_maximum"
    DIVIDE_BY_VALUE = "divide_by_value"
