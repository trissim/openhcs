"""Typed CellProfiler Crop semantics shared by conversion and execution."""

from __future__ import annotations

from enum import Enum


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
