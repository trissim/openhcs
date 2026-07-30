"""Nominal image-tile blending contract shared by assembly backends."""

from enum import Enum


class TileBlendMethod(Enum):
    """Weighting strategy used where assembled image tiles overlap."""

    NONE = "none"
    FIXED = "fixed"
    DYNAMIC = "dynamic"
