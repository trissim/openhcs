"""Benchmark-library facade for CellProfiler GrayToColor."""

from openhcs.processing.backends.cellprofiler.color import (
    CMYKGrayToColorRunner,
    CompositeGrayToColorRunner,
    GrayToColorModule,
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
        "GrayToColorSchemeRunner",
    "RGBGrayToColorRunner",
    "StackGrayToColorRunner",
    "gray_to_color",
]
