"""Shared substrate for processor method strategy families."""

from __future__ import annotations

from enum import Enum


class SpatialBinMethod(Enum):
    """Reduction applied within one spatial bin."""

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"


class StackProjectionMethod(Enum):
    """Max/mean reductions shared by processor backends."""

    MAX = "max_projection"
    MEAN = "mean_projection"


class EdgeMagnitudeMethod(Enum):
    """Spatial domain used for Sobel edge magnitude."""

    SLICE_2D = "2d"
    VOLUME_3D = "3d"


class ScipyBoundaryMode(Enum):
    """Boundary extension modes supported by SciPy-compatible filters."""

    CONSTANT = "constant"
    REFLECT = "reflect"
    NEAREST = "nearest"
    WRAP = "wrap"
    MIRROR = "mirror"


class SobelKernelBoundaryMode(Enum):
    """Boundary extension modes implemented by the custom 2-D Sobel kernel."""

    CONSTANT = ("constant", 0)
    REFLECT = ("reflect", 1)
    NEAREST = ("nearest", 2)
    WRAP = ("wrap", 3)

    def __new__(cls, label: str, kernel_code: int):
        member = object.__new__(cls)
        member._value_ = label
        member.kernel_code = kernel_code
        return member


class OrthogonalProjectionPlane(Enum):
    """Plane retained by an orthogonal stack projection."""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"
