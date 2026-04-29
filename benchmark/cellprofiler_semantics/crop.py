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


class RemovalMethod(str, Enum):
    """Closed CellProfiler row/column removal modes."""

    NO = "No"
    EDGES = "Edges"
    ALL = "All"
