"""Shared CellProfiler color literal semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.interop.cellprofiler.gray_to_color_settings import (
    GrayToColorScheme,
    coerce_gray_to_color_scheme,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
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


class StainType(Enum):
    """Closed family of CellProfiler UnmixColors stain choices."""

    HEMATOXYLIN = ("Hematoxylin", (0.644, 0.717, 0.267))
    EOSIN = ("Eosin", (0.093, 0.954, 0.283))
    DAB = ("DAB", (0.268, 0.570, 0.776))
    FAST_RED = ("Fast red", (0.214, 0.851, 0.478))
    FAST_BLUE = ("Fast blue", (0.749, 0.606, 0.267))
    METHYL_BLUE = ("Methyl blue", (0.799, 0.591, 0.105))
    METHYL_GREEN = ("Methyl green", (0.980, 0.144, 0.133))
    AEC = ("AEC", (0.274, 0.679, 0.680))
    ANILINE_BLUE = ("Aniline blue", (0.853, 0.509, 0.113))
    AZOCARMINE = ("Azocarmine", (0.071, 0.977, 0.198))
    ALCIAN_BLUE = ("Alcian blue", (0.875, 0.458, 0.158))
    PAS = ("PAS", (0.175, 0.972, 0.155))
    HEMATOXYLIN_AND_PAS = ("Hematoxylin and PAS", (0.553, 0.754, 0.354))
    FEULGEN = ("Feulgen", (0.464, 0.830, 0.308))
    METHYLENE_BLUE = ("Methylene blue", (0.553, 0.754, 0.354))
    ORANGE_G = ("Orange-G", (0.107, 0.368, 0.923))
    PONCEAU_FUCHSIN = ("Ponceau-fuchsin", (0.100, 0.737, 0.668))
    CUSTOM = ("Custom", None)

    @property
    def display_name(self) -> str:
        return self.value[0]

    @property
    def calibrated_absorbance(self) -> tuple[float, float, float]:
        absorbance = self.value[1]
        if absorbance is None:
            raise ValueError("Custom stains require explicit absorbance values.")
        return absorbance


@dataclass(frozen=True, slots=True)
class StainDefinition:
    """One stain row participating in CellProfiler color deconvolution."""

    stain: StainType
    custom_absorbance: tuple[float, float, float] | None = None

    @property
    def absorbance(self) -> np.ndarray:
        if self.stain is StainType.CUSTOM:
            if self.custom_absorbance is None:
                raise ValueError("Custom UnmixColors rows require absorbance values.")
            absorbance = self.custom_absorbance
        else:
            absorbance = self.stain.calibrated_absorbance
        return _normalized_absorbance(absorbance)


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


@numpy(contract=ProcessingContract.FLEXIBLE)
def unmix_colors(
    image: np.ndarray,
    stain_names: Sequence[StainType | str] = (),
    custom_absorbances: Sequence[Sequence[float] | None] = (),
    stain1: StainType | str = StainType.HEMATOXYLIN,
    stain2: StainType | str = StainType.EOSIN,
    stain3: StainType | str | None = None,
    output_stain_index: int = 0,
    custom_red_absorbance_1: float = 0.5,
    custom_green_absorbance_1: float = 0.5,
    custom_blue_absorbance_1: float = 0.5,
    custom_red_absorbance_2: float = 0.5,
    custom_green_absorbance_2: float = 0.5,
    custom_blue_absorbance_2: float = 0.5,
    custom_red_absorbance_3: float = 0.5,
    custom_green_absorbance_3: float = 0.5,
    custom_blue_absorbance_3: float = 0.5,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Unmix one RGB image into one image per configured CellProfiler stain row."""
    rgb_image = _as_rgb_image(image)
    if stain_names:
        return _unmix_stain_outputs(
            rgb_image,
            _stain_definitions(stain_names, custom_absorbances),
        )

    definitions = _legacy_stain_definitions(
        stain1=stain1,
        stain2=stain2,
        stain3=stain3,
        custom_absorbances=(
            (
                custom_red_absorbance_1,
                custom_green_absorbance_1,
                custom_blue_absorbance_1,
            ),
            (
                custom_red_absorbance_2,
                custom_green_absorbance_2,
                custom_blue_absorbance_2,
            ),
            (
                custom_red_absorbance_3,
                custom_green_absorbance_3,
                custom_blue_absorbance_3,
            ),
        ),
    )
    outputs = _unmix_stain_outputs(rgb_image, definitions)
    if output_stain_index < 0 or output_stain_index >= len(outputs):
        raise ValueError(
            f"output_stain_index must be in [0, {len(outputs) - 1}], "
            f"got {output_stain_index}."
        )
    return outputs[output_stain_index]


def _stain_definitions(
    stain_names: Sequence[StainType | str],
    custom_absorbances: Sequence[Sequence[float] | None],
) -> tuple[StainDefinition, ...]:
    if len(stain_names) != len(custom_absorbances):
        raise ValueError(
            "UnmixColors stain_names and custom_absorbances must have the "
            "same length."
        )
    return tuple(
        StainDefinition(
            stain=coerce_cellprofiler_enum(StainType, stain_name),
            custom_absorbance=_coerce_custom_absorbance(custom_absorbance),
        )
        for stain_name, custom_absorbance in zip(
            stain_names,
            custom_absorbances,
            strict=True,
        )
    )


def _legacy_stain_definitions(
    *,
    stain1: StainType | str,
    stain2: StainType | str,
    stain3: StainType | str | None,
    custom_absorbances: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[StainDefinition, ...]:
    stains = (stain1, stain2, stain3)
    return tuple(
        StainDefinition(
            stain=coerce_cellprofiler_enum(StainType, stain),
            custom_absorbance=custom_absorbances[index],
        )
        for index, stain in enumerate(stains)
        if stain is not None
    )


def _unmix_stain_outputs(
    image: np.ndarray,
    definitions: tuple[StainDefinition, ...],
) -> tuple[np.ndarray, ...]:
    if not definitions:
        raise ValueError("UnmixColors requires at least one stain definition.")
    inverse_matrix = np.linalg.pinv(
        np.asarray([definition.absorbance for definition in definitions])
    )
    return tuple(
        _run_unmix_output(image, inverse_matrix[:, index])
        for index in range(len(definitions))
    )


def _run_unmix_output(
    image: np.ndarray,
    inverse_absorbances: np.ndarray,
) -> np.ndarray:
    eps = 1.0 / 256.0 / 2.0
    log_image = np.log(image + eps)
    broadcast_shape = (1,) * (log_image.ndim - 1) + (3,)
    scaled_image = log_image * inverse_absorbances.reshape(broadcast_shape)
    result = np.exp(np.sum(scaled_image, axis=-1)) - eps
    return (1.0 - np.clip(result, 0.0, 1.0)).astype(np.float32)


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return np.stack((array, array, array), axis=-1)
    if array.ndim >= 3 and array.shape[-1] == 3:
        return array
    if array.ndim == 3 and array.shape[0] == 3:
        return np.moveaxis(array, 0, -1)
    if array.ndim >= 4 and array.shape[1] == 3:
        return np.moveaxis(array, 1, -1)
    raise ValueError(
        "UnmixColors expects an RGB image with three color channels on the "
        f"first or last channel axis, got shape {array.shape}."
    )


def _normalized_absorbance(absorbance: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple(float(channel) for channel in absorbance))
    if vector.shape != (3,):
        raise ValueError(
            f"UnmixColors absorbance vectors must have three channels, got {vector}."
        )
    norm = np.sqrt(np.sum(vector**2))
    if norm <= 0:
        raise ValueError("UnmixColors absorbance vectors cannot be zero.")
    return vector / norm


def _coerce_custom_absorbance(
    absorbance: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    if absorbance is None:
        return None
    red, green, blue = absorbance
    return float(red), float(green), float(blue)
