"""Shared CellProfiler color literal semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.interop.cellprofiler.gray_to_color_settings import (
    GrayToColorScheme,
    coerce_gray_to_color_scheme,
)

class CellProfilerColorFormat(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser family for CellProfiler RGB color literals."""

    __registry_key__ = "format_key"
    __skip_if_no_key__ = True
    format_key: ClassVar[str | None] = None

    @classmethod
    def for_value(cls, value: str | Sequence[float]) -> "CellProfilerColorFormat":
        for format_type in cls.__registry__.values():
            parser = format_type()
            if parser.matches(value):
                return parser
        raise ValueError(f"Unsupported CellProfiler color literal: {value!r}")

    @abstractmethod
    def matches(self, value: str | Sequence[float]) -> bool:
        """Return whether this parser owns the color literal."""

    @abstractmethod
    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        """Return RGB channel values, possibly in 0-255 space."""


class NamedCellProfilerColorFormat(CellProfilerColorFormat):
    """Named CellProfiler colors."""

    format_key = "named"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and value.strip().lower() in _COLOR_BY_NAME

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return _COLOR_BY_NAME[str(value).strip().lower()]


class HexCellProfilerColorFormat(CellProfilerColorFormat):
    """Hex CellProfiler colors such as #0800F7."""

    format_key = "hex"

    def matches(self, value: str | Sequence[float]) -> bool:
        if not isinstance(value, str):
            return False
        literal = value.strip()
        return literal.startswith("#") and len(literal) in {4, 7}

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        literal = str(value).strip().lstrip("#")
        if len(literal) == 3:
            literal = "".join(channel * 2 for channel in literal)
        return (
            int(literal[0:2], 16),
            int(literal[2:4], 16),
            int(literal[4:6], 16),
        )


class DelimitedCellProfilerColorFormat(CellProfilerColorFormat):
    """Comma-delimited RGB triples."""

    format_key = "delimited"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and "," in value

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(part.strip()) for part in str(value).split(","))  # type: ignore[return-value]


class SequenceCellProfilerColorFormat(CellProfilerColorFormat):
    """Already-structured RGB channel sequences."""

    format_key = "sequence"

    def matches(self, value: str | Sequence[float]) -> bool:
        return not isinstance(value, str) and isinstance(value, Sequence)

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(part) for part in value)  # type: ignore[arg-type, return-value]


def coerce_rgb_color(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse a CellProfiler color literal into an RGB tuple in 0-1 space."""
    parts = CellProfilerColorFormat.for_value(value).color_parts(value)
    if len(parts) != 3:
        raise ValueError(f"CellProfiler color must have three channels, got {parts!r}.")
    scale = 255.0 if max(parts) > 1.0 else 1.0
    return parts[0] / scale, parts[1] / scale, parts[2] / scale


_COLOR_BY_NAME: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class GrayToColorRequest:
    """Typed request record for one GrayToColor dispatch."""

    image: np.ndarray
    rescale_intensity: bool = True
    red_channel: int = -1
    green_channel: int = -1
    blue_channel: int = -1
    cyan_channel: int = -1
    magenta_channel: int = -1
    yellow_channel: int = -1
    gray_channel: int = -1
    red_weight: float = 1.0
    green_weight: float = 1.0
    blue_weight: float = 1.0
    cyan_weight: float = 1.0
    magenta_weight: float = 1.0
    yellow_weight: float = 1.0
    gray_weight: float = 1.0
    channel_colors: Sequence[str] = ()
    channel_weights: Sequence[float] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_colors", tuple(self.channel_colors))
        object.__setattr__(self, "channel_weights", tuple(self.channel_weights))


class GrayToColorSchemeRunner(ABC, metaclass=AutoRegisterMeta):
    """Nominal closed family for GrayToColor scheme dispatch."""

    __registry_key__ = "scheme_literal"
    __skip_if_no_key__ = True
    scheme_literal: ClassVar[str | None] = None

    @classmethod
    def for_scheme(
        cls,
        scheme: GrayToColorScheme,
    ) -> "GrayToColorSchemeRunner":
        runner_type = cls.__registry__.get(scheme.value)
        if runner_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return runner_type()

    @abstractmethod
    def run(self, request: GrayToColorRequest) -> np.ndarray:
        """Execute one GrayToColor request for the scheme owned by this runner."""

    def channel_or_black(
        self,
        image: np.ndarray,
        channel_index: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        if channel_index < 0:
            return np.zeros((height, width), dtype=np.float64)
        return image[channel_index].astype(np.float64)

    def rescale_positive_channel(self, channel: np.ndarray) -> np.ndarray:
        maximum = np.max(channel)
        if maximum > 0:
            return channel / maximum
        return channel

    def final_rgb(
        self,
        rgb_image: np.ndarray,
        request: GrayToColorRequest,
    ) -> np.ndarray:
        if request.rescale_intensity:
            rgb_image = np.clip(rgb_image, 0, 1)
        return rgb_image.astype(np.float32)


class RGBGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorScheme.RGB.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        height, width = image.shape[1], image.shape[2]

        red_img = self.channel_or_black(image, request.red_channel, height, width)
        green_img = self.channel_or_black(image, request.green_channel, height, width)
        blue_img = self.channel_or_black(image, request.blue_channel, height, width)

        if request.rescale_intensity:
            red_img = self.rescale_positive_channel(red_img)
            green_img = self.rescale_positive_channel(green_img)
            blue_img = self.rescale_positive_channel(blue_img)

        rgb_image = np.dstack(
            [
                red_img * request.red_weight,
                green_img * request.green_weight,
                blue_img * request.blue_weight,
            ]
        )
        return self.final_rgb(rgb_image, request)


class CMYKGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorScheme.CMYK.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        height, width = image.shape[1], image.shape[2]

        cyan_img = self.channel_or_black(image, request.cyan_channel, height, width)
        magenta_img = self.channel_or_black(
            image,
            request.magenta_channel,
            height,
            width,
        )
        yellow_img = self.channel_or_black(image, request.yellow_channel, height, width)
        gray_img = self.channel_or_black(image, request.gray_channel, height, width)

        if request.rescale_intensity:
            cyan_img = self.rescale_positive_channel(cyan_img)
            magenta_img = self.rescale_positive_channel(magenta_img)
            yellow_img = self.rescale_positive_channel(yellow_img)
            gray_img = self.rescale_positive_channel(gray_img)

        rgb_image = np.zeros((height, width, 3), dtype=np.float64)
        rgb_image[:, :, 1] += cyan_img * request.cyan_weight * 0.5
        rgb_image[:, :, 2] += cyan_img * request.cyan_weight * 0.5
        rgb_image[:, :, 0] += magenta_img * request.magenta_weight * 0.5
        rgb_image[:, :, 2] += magenta_img * request.magenta_weight * 0.5
        rgb_image[:, :, 0] += yellow_img * request.yellow_weight * 0.5
        rgb_image[:, :, 1] += yellow_img * request.yellow_weight * 0.5
        rgb_image[:, :, 0] += gray_img * request.gray_weight * (1.0 / 3.0)
        rgb_image[:, :, 1] += gray_img * request.gray_weight * (1.0 / 3.0)
        rgb_image[:, :, 2] += gray_img * request.gray_weight * (1.0 / 3.0)
        return self.final_rgb(rgb_image, request)


class StackGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorScheme.STACK.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        return np.transpose(request.image, (1, 2, 0)).astype(np.float32)


class CompositeGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorScheme.COMPOSITE.value
    default_colors: ClassVar[tuple[str, ...]] = (
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#808000",
        "#800080",
        "#008080",
    )

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        colors = list(request.channel_colors) or [
            self.default_colors[index % len(self.default_colors)]
            for index in range(image.shape[0])
        ]
        weights = list(request.channel_weights) or [1.0] * image.shape[0]
        height, width = image.shape[1], image.shape[2]
        rgb_image = np.zeros((height, width, 3), dtype=np.float64)

        for index in range(image.shape[0]):
            channel_img = image[index].astype(np.float64)
            if request.rescale_intensity:
                channel_img = self.rescale_positive_channel(channel_img)
            red, green, blue = coerce_rgb_color(colors[index])
            weight = weights[index]
            rgb_image[:, :, 0] += channel_img * red * weight
            rgb_image[:, :, 1] += channel_img * green * weight
            rgb_image[:, :, 2] += channel_img * blue * weight

        return self.final_rgb(rgb_image, request)


@numpy
def gray_to_color(
    image: np.ndarray,
    color_scheme: GrayToColorScheme | str = GrayToColorScheme.RGB.value,
    rescale_intensity: bool = True,
    red_channel: int = -1,
    green_channel: int = -1,
    blue_channel: int = -1,
    cyan_channel: int = -1,
    magenta_channel: int = -1,
    yellow_channel: int = -1,
    gray_channel: int = -1,
    red_weight: float = 1.0,
    green_weight: float = 1.0,
    blue_weight: float = 1.0,
    cyan_weight: float = 1.0,
    magenta_weight: float = 1.0,
    yellow_weight: float = 1.0,
    gray_weight: float = 1.0,
    channel_colors: Sequence[str] = (),
    channel_weights: Sequence[float] = (),
) -> np.ndarray:
    """Dispatch GrayToColor across its RGB, CMYK, Stack, and Composite variants."""
    scheme = coerce_gray_to_color_scheme(color_scheme)
    request = GrayToColorRequest(
        image=image,
        rescale_intensity=rescale_intensity,
        red_channel=red_channel,
        green_channel=green_channel,
        blue_channel=blue_channel,
        cyan_channel=cyan_channel,
        magenta_channel=magenta_channel,
        yellow_channel=yellow_channel,
        gray_channel=gray_channel,
        red_weight=red_weight,
        green_weight=green_weight,
        blue_weight=blue_weight,
        cyan_weight=cyan_weight,
        magenta_weight=magenta_weight,
        yellow_weight=yellow_weight,
        gray_weight=gray_weight,
        channel_colors=channel_colors,
        channel_weights=channel_weights,
    )
    return GrayToColorSchemeRunner.for_scheme(scheme).run(request)
