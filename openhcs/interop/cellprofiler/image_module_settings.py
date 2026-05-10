"""CellProfiler image-module setting semantics."""

from __future__ import annotations

from enum import Enum


class CombineObjectsMethod(Enum):
    """Overlap policies exposed by CombineObjects settings."""

    MERGE = "merge"
    PRESERVE = "preserve"
    DISCARD = "discard"
    SEGMENT = "segment"


class ConvertObjectsToImageMode(Enum):
    """Object-label rendering modes exposed by ConvertObjectsToImage settings."""

    BINARY = "binary"
    GRAYSCALE = "grayscale"
    COLOR = "color"
    UINT16 = "uint16"


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


class WatershedDeclumpMethod(Enum):
    """Declump priority families exposed by Watershed settings."""

    SHAPE = "shape"
    INTENSITY = "intensity"
    NONE = "none"


class WatershedMethod(Enum):
    """Watershed surface sources exposed by CellProfiler settings."""

    DISTANCE = "distance"
    INTENSITY = "intensity"
    MARKERS = "markers"


class WatershedInputKeyword(Enum):
    """Runtime kwargs used for Watershed special image inputs."""

    MASK = "mask"
    MARKERS = "markers"
