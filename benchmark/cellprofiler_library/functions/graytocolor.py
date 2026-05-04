"""
Converted from CellProfiler: GrayToColor
Original: GrayToColor module

Takes grayscale images and produces a color image from them.
Supports RGB, CMYK, Stack, and Composite color schemes.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence

from metaclass_registry import AutoRegisterMeta
import numpy as np
from openhcs.core.memory import numpy

from benchmark.cellprofiler_library.color import coerce_rgb_color


class GrayToColorScheme(str, Enum):
    """Closed family of supported GrayToColor scheme literals."""

    RGB = "RGB"
    CMYK = "CMYK"
    STACK = "Stack"
    COMPOSITE = "Composite"


def _coerce_gray_to_color_scheme(
    value: GrayToColorScheme | str,
) -> GrayToColorScheme:
    if isinstance(value, GrayToColorScheme):
        return value
    normalized = value.strip()
    for scheme in GrayToColorScheme:
        if scheme.value == normalized:
            return scheme
    raise ValueError(f"Unsupported GrayToColor scheme: {value!r}")


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


def _gray_to_color_rgb(request: GrayToColorRequest) -> np.ndarray:
    """
    Combine grayscale images into an RGB color image.

    Args:
        image: Shape (N, H, W) - N grayscale images stacked along dim 0
        red_channel: Index of channel to use for red (default 0, use -1 for black)
        green_channel: Index of channel to use for green (default 1, use -1 for black)
        blue_channel: Index of channel to use for blue (default 2, use -1 for black)
        red_weight: Relative weight for the red image.
        green_weight: Relative weight for the green image.
        blue_weight: Relative weight for the blue image.
        rescale_intensity: Whether to rescale each channel to 0-1 range.

    Returns:
        Shape (H, W, 3) RGB color image.

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the image to be colored red' -> red_channel
        'Select the image to be colored green' -> green_channel
        'Select the image to be colored blue' -> blue_channel
        'Relative weight for the red image' -> red_weight
        'Relative weight for the green image' -> green_weight
        'Relative weight for the blue image' -> blue_weight
    """
    image = request.image
    h, w = image.shape[1], image.shape[2]

    # Get channels (use zeros if -1)
    red_img = image[request.red_channel].astype(np.float64) if request.red_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    green_img = image[request.green_channel].astype(np.float64) if request.green_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    blue_img = image[request.blue_channel].astype(np.float64) if request.blue_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    
    if request.rescale_intensity:
        if np.max(red_img) > 0:
            red_img = red_img / np.max(red_img)
        if np.max(green_img) > 0:
            green_img = green_img / np.max(green_img)
        if np.max(blue_img) > 0:
            blue_img = blue_img / np.max(blue_img)
    
    # Apply weights
    red_img = red_img * request.red_weight
    green_img = green_img * request.green_weight
    blue_img = blue_img * request.blue_weight
    
    # Stack into RGB image (H, W, 3)
    rgb_image = np.dstack([red_img, green_img, blue_img])
    
    # Clip values that went out of range after multiplication
    if request.rescale_intensity:
        rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image.astype(np.float32)



def _gray_to_color_cmyk(request: GrayToColorRequest) -> np.ndarray:
    """
    Combine grayscale images into a color image using CMYK scheme.

    Args:
        request: Typed CMYK request.

    Returns:
        Shape (H, W, 3) RGB color image.
    """
    image = request.image
    h, w = image.shape[1], image.shape[2]

    # Get channels (use zeros if -1)
    cyan_img = image[request.cyan_channel].astype(np.float64) if request.cyan_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    magenta_img = image[request.magenta_channel].astype(np.float64) if request.magenta_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    yellow_img = image[request.yellow_channel].astype(np.float64) if request.yellow_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    gray_img = image[request.gray_channel].astype(np.float64) if request.gray_channel >= 0 else np.zeros((h, w), dtype=np.float64)
    
    if request.rescale_intensity:
        if np.max(cyan_img) > 0:
            cyan_img = cyan_img / np.max(cyan_img)
        if np.max(magenta_img) > 0:
            magenta_img = magenta_img / np.max(magenta_img)
        if np.max(yellow_img) > 0:
            yellow_img = yellow_img / np.max(yellow_img)
        if np.max(gray_img) > 0:
            gray_img = gray_img / np.max(gray_img)
    
    # CMYK to RGB conversion with weights
    # Cyan adds to green and blue (0, 0.5, 0.5)
    # Magenta adds to red and blue (0.5, 0, 0.5)
    # Yellow adds to red and green (0.5, 0.5, 0)
    # Gray adds equally to all (1/3, 1/3, 1/3)
    
    rgb_image = np.zeros((h, w, 3), dtype=np.float64)
    
    # Cyan contribution
    rgb_image[:, :, 1] += cyan_img * request.cyan_weight * 0.5  # green
    rgb_image[:, :, 2] += cyan_img * request.cyan_weight * 0.5  # blue
    
    # Magenta contribution
    rgb_image[:, :, 0] += magenta_img * request.magenta_weight * 0.5  # red
    rgb_image[:, :, 2] += magenta_img * request.magenta_weight * 0.5  # blue
    
    # Yellow contribution
    rgb_image[:, :, 0] += yellow_img * request.yellow_weight * 0.5  # red
    rgb_image[:, :, 1] += yellow_img * request.yellow_weight * 0.5  # green
    
    # Gray contribution
    rgb_image[:, :, 0] += gray_img * request.gray_weight * (1.0 / 3.0)  # red
    rgb_image[:, :, 1] += gray_img * request.gray_weight * (1.0 / 3.0)  # green
    rgb_image[:, :, 2] += gray_img * request.gray_weight * (1.0 / 3.0)  # blue
    
    # Clip values
    if request.rescale_intensity:
        rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image.astype(np.float32)


def _gray_to_color_stack(request: GrayToColorRequest) -> np.ndarray:
    """
    Stack grayscale images into a multi-channel image.
    
    Args:
        image: Shape (N, H, W) - N grayscale images stacked.
    
    Returns:
        Shape (H, W, N) multi-channel image.
    """
    # Transpose from (N, H, W) to (H, W, N)
    return np.transpose(request.image, (1, 2, 0)).astype(np.float32)


def _gray_to_color_composite(request: GrayToColorRequest) -> np.ndarray:
    """
    Combine grayscale images into a composite color image.
    
    Each grayscale image is assigned a color and weighted, then
    all colored images are added together.
    
    Args:
        image: Shape (N, H, W) - N grayscale images stacked.
        colors: List of N hex color strings (e.g., ['#ff0000', '#00ff00']).
                Defaults to cycling through red, green, blue, yellow, magenta, cyan.
        weights: List of N weights for each image. Defaults to 1.0 for all.
        rescale_intensity: Whether to rescale each channel to 0-1 range.
    
    Returns:
        Shape (H, W, 3) RGB color image.
    """
    image = request.image
    colors = list(request.channel_colors) or None
    weights = list(request.channel_weights) or None
    n_channels = image.shape[0]
    h, w = image.shape[1], image.shape[2]
    
    # Default colors
    default_colors = ["#ff0000", "#00ff00", "#0000ff", "#808000", "#800080", "#008080"]
    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(n_channels)]
    
    # Default weights
    if weights is None:
        weights = [1.0] * n_channels
    
    rgb_image = np.zeros((h, w, 3), dtype=np.float64)
    
    for i in range(n_channels):
        channel_img = image[i].astype(np.float64)
        
        if request.rescale_intensity and np.max(channel_img) > 0:
            channel_img = channel_img / np.max(channel_img)
        
        # Get RGB color
        r, g, b = coerce_rgb_color(colors[i])
        weight = weights[i]
        
        # Add weighted colored image
        rgb_image[:, :, 0] += channel_img * r * weight
        rgb_image[:, :, 1] += channel_img * g * weight
        rgb_image[:, :, 2] += channel_img * b * weight
    
    # Clip values
    if request.rescale_intensity:
        rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image.astype(np.float32)


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


class HelperBackedGrayToColorSchemeRunner(GrayToColorSchemeRunner):
    """Shared declarative substrate for helper-backed scheme dispatch."""

    helper: ClassVar[Callable[..., np.ndarray] | None] = None

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        helper = type(self).helper
        if helper is None:
            raise TypeError(f"{type(self).__name__} must define helper.")
        return helper(**self._helper_kwargs(request))

    @abstractmethod
    def _helper_kwargs(self, request: GrayToColorRequest) -> Mapping[str, object]:
        """Return helper kwargs for this scheme."""


class _RequestBackedGrayToColorSchemeRunner(HelperBackedGrayToColorSchemeRunner):
    """Scheme runner whose helper directly consumes the authoritative request."""

    def _helper_kwargs(self, request: GrayToColorRequest) -> Mapping[str, object]:
        return {"request": request}


_REQUEST_BACKED_GRAY_TO_COLOR_RUNNER_SPECS: tuple[
    tuple[str, Callable[..., np.ndarray]],
    ...,
] = (
    (GrayToColorScheme.RGB.value, _gray_to_color_rgb),
    (GrayToColorScheme.CMYK.value, _gray_to_color_cmyk),
    (GrayToColorScheme.STACK.value, _gray_to_color_stack),
    (GrayToColorScheme.COMPOSITE.value, _gray_to_color_composite),
)


def _declare_request_backed_gray_to_color_runner(
    scheme_literal: str,
    helper: Callable[..., np.ndarray],
) -> None:
    class_name = f"{scheme_literal.replace(' ', '')}GrayToColorRunner"
    globals()[class_name] = type(
        class_name,
        (_RequestBackedGrayToColorSchemeRunner,),
        {
            "__module__": __name__,
            "scheme_literal": scheme_literal,
            "helper": helper,
        },
    )


for _scheme_literal, _helper in _REQUEST_BACKED_GRAY_TO_COLOR_RUNNER_SPECS:
    _declare_request_backed_gray_to_color_runner(_scheme_literal, _helper)


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
    """
    Dispatch GrayToColor across its RGB, CMYK, Stack, and Composite variants.

    CellProfiler Parameter Mapping:
        'Select a color scheme' -> color_scheme
        'Rescale intensity' -> rescale_intensity
        'Select the image to be colored red' -> red_channel
        'Select the image to be colored green' -> green_channel
        'Select the image to be colored blue' -> blue_channel
        'Relative weight for the red image' -> red_weight
        'Relative weight for the green image' -> green_weight
        'Relative weight for the blue image' -> blue_weight
        'Select the image to be colored cyan' -> cyan_channel
        'Select the image to be colored magenta' -> magenta_channel
        'Select the image to be colored yellow' -> yellow_channel
        'Select the image that determines brightness' -> gray_channel
        'Relative weight for the cyan image' -> cyan_weight
        'Relative weight for the magenta image' -> magenta_weight
        'Relative weight for the yellow image' -> yellow_weight
        'Relative weight for the brightness image' -> gray_weight
        'Image name' -> (pipeline-handled)
        'Color' -> channel_colors
        'Weight' -> channel_weights
        'Name the output image' -> (pipeline-handled)
    """
    scheme = _coerce_gray_to_color_scheme(color_scheme)
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
