"""Benchmark-library facade for CellProfiler GrayToColor."""

from openhcs.interop.cellprofiler.gray_to_color_settings import GrayToColorScheme
from openhcs.processing.backends.cellprofiler.color import (
    CMYKGrayToColorRunner,
    CompositeGrayToColorRunner,
    GrayToColorRequest,
    GrayToColorSchemeRunner,
    RGBGrayToColorRunner,
    StackGrayToColorRunner,
    gray_to_color,
)

__all__ = [
    "CMYKGrayToColorRunner",
    "CompositeGrayToColorRunner",
    "GrayToColorRequest",
    "GrayToColorScheme",
    "GrayToColorSchemeRunner",
    "RGBGrayToColorRunner",
    "StackGrayToColorRunner",
    "gray_to_color",
]
