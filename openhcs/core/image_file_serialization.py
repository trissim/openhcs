"""Nominal image-file formats for source metadata and disk serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.runtime_image_values import (
    image_intensity_scale_for_dtype,
    image_payload_data,
    image_payload_metadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceImagePixelSemantics:
    """Channel layout declared by an image-file format."""

    channel_axis: int | None = None
    channel_count: int | None = None

    def __post_init__(self) -> None:
        if (self.channel_axis is None) != (self.channel_count is None):
            raise ValueError(
                "Source image pixel channel axis and count must be declared together."
            )
        if self.channel_count is not None and self.channel_count <= 1:
            raise ValueError(
                "Source image pixel channel count must exceed one when declared."
            )

    def validated_channel_axis(self, payload: Any) -> int | None:
        """Validate loaded pixels against this format-owned declaration."""
        axis = self.channel_axis
        if axis is None:
            return None
        shape = tuple(int(value) for value in np.shape(image_payload_data(payload)))
        normalized = axis if axis >= 0 else len(shape) + axis
        if normalized < 0 or normalized >= len(shape):
            raise ValueError(
                f"Declared source pixel channel axis {axis} is invalid for loaded "
                f"payload shape {shape!r}."
            )
        if shape[normalized] != self.channel_count:
            raise ValueError(
                "Loaded payload conflicts with declared source pixel semantics: "
                f"axis {axis} carries {shape[normalized]} channel values, expected "
                f"{self.channel_count}."
            )
        return axis


@dataclass(frozen=True, slots=True)
class ImageFileSourceMetadata:
    """Source metadata read through one registered image-file format."""

    source_dtype: Any | None = None
    intensity_scale: float | None = None
    pixel_semantics: SourceImagePixelSemantics = SourceImagePixelSemantics()


class ImageFileFormat(ABC, metaclass=AutoRegisterMeta):
    """Nominal owner of image-file source and serialization semantics."""

    __registry_key__ = "format_key"
    __skip_if_no_key__ = True
    format_key: ClassVar[str | None] = None
    suffixes: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def matches_path(cls, path: str | Path) -> bool:
        """Return whether this exact nominal format owns ``path``."""
        return Path(path).suffix.lower() in cls.suffixes

    @classmethod
    def is_image_path(cls, path: str | Path) -> bool:
        """Return whether one registered image format exactly owns ``path``."""
        return any(
            format_type.matches_path(path) for format_type in cls.__registry__.values()
        )

    @classmethod
    def require_path(cls, path: str | Path) -> "ImageFileFormat":
        matches = tuple(
            format_type
            for format_type in cls.__registry__.values()
            if format_type.matches_path(path)
        )
        if len(matches) == 1:
            return matches[0]()
        suffix = Path(path).suffix.lower()
        if len(matches) > 1:
            raise ValueError(
                "Multiple image serialization formats are registered for suffix "
                f"{suffix!r}: {tuple(item.__name__ for item in matches)!r}."
            )
        raise ValueError(
            f"No image serialization format is registered for suffix {suffix!r}."
        )

    @abstractmethod
    def prepare(self, payload: Any) -> Any:
        """Return a payload suitable for this file format."""

    def read(self, path: str | Path) -> np.ndarray:
        """Read pixels through this exact registered image-file format."""
        import imageio.v3 as iio

        return np.asarray(iio.imread(path))

    def write(self, path: str | Path, payload: Any) -> None:
        """Write pixels through this exact registered image-file format."""
        import imageio.v3 as iio

        iio.imwrite(path, self.prepare(payload))

    def source_metadata(self, path: Path) -> ImageFileSourceMetadata:
        """Read format-owned source metadata without loading pixel data."""
        try:
            import imageio.v3 as iio

            dtype = iio.improps(path).dtype
        except Exception:
            logger.debug("Could not read image metadata for %s.", path, exc_info=True)
            return ImageFileSourceMetadata()
        return ImageFileSourceMetadata(
            source_dtype=dtype,
            intensity_scale=(
                self.declared_intensity_scale(path)
                or image_intensity_scale_for_dtype(dtype)
            ),
            pixel_semantics=self.pixel_semantics(path),
        )

    def requires_plane_store_decoder(self, path: Path) -> bool:
        """Return whether embedded metadata requires a richer plane decoder."""
        del path
        return False

    def declared_intensity_scale(self, path: Path) -> float | None:
        """Return a container-declared intensity scale when the format has one."""
        del path
        return None

    def pixel_semantics(self, path: Path) -> SourceImagePixelSemantics:
        """Return explicit channel-band semantics exposed by the file container."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                band_count = len(image.getbands())
        except Exception:
            logger.debug(
                "Could not read source pixel-band metadata for %s.",
                path,
                exc_info=True,
            )
            return SourceImagePixelSemantics()
        if band_count <= 1:
            return SourceImagePixelSemantics()
        return SourceImagePixelSemantics(channel_axis=-1, channel_count=band_count)


class NumpyImageFileFormat(ImageFileFormat):
    """NumPy array files preserve the payload dtype directly."""

    format_key = "numpy"
    suffixes = (".npy",)

    def prepare(self, payload: Any) -> Any:
        return image_payload_data(payload)

    def read(self, path: str | Path) -> np.ndarray:
        return np.asarray(np.load(path, allow_pickle=False))

    def write(self, path: str | Path, payload: Any) -> None:
        np.save(path, self.prepare(payload), allow_pickle=False)


class TiffImageFileFormat(ImageFileFormat):
    """TIFF preserves dtype and may declare a physical maximum sample value."""

    format_key = "tiff"
    suffixes = (".tif", ".tiff")

    def prepare(self, payload: Any) -> Any:
        return image_payload_data(payload)

    def requires_plane_store_decoder(self, path: Path) -> bool:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            return bool(tif.is_ome)

    def source_metadata(self, path: Path) -> ImageFileSourceMetadata:
        """Read TIFF dtype and sample scale through one container context."""
        try:
            import tifffile

            tif = tifffile.TiffFile(path)
        except Exception:
            logger.debug("Could not read image metadata for %s.", path, exc_info=True)
            return ImageFileSourceMetadata()
        with tif:
            try:
                series = tif.series[0]
                dtype = series.dtype
            except Exception:
                logger.debug(
                    "Could not read image metadata for %s.", path, exc_info=True
                )
                return ImageFileSourceMetadata()
            declared_scale = self._declared_intensity_scale_from_page(tif.pages[0])
            sample_axis = series.axes.find("S")
            sample_count = (
                int(series.shape[sample_axis]) if sample_axis >= 0 else None
            )
            pixel_semantics = SourceImagePixelSemantics()
            if sample_count is not None and sample_count > 1:
                pixel_semantics = SourceImagePixelSemantics(
                    channel_axis=(
                        -1 if sample_axis == len(series.shape) - 1 else sample_axis
                    ),
                    channel_count=sample_count,
                )
        return ImageFileSourceMetadata(
            source_dtype=dtype,
            intensity_scale=(
                declared_scale or image_intensity_scale_for_dtype(dtype)
            ),
            pixel_semantics=pixel_semantics,
        )

    def declared_intensity_scale(self, path: Path) -> float | None:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            return self._declared_intensity_scale_from_page(tif.pages[0])

    @staticmethod
    def _declared_intensity_scale_from_page(page: Any) -> float | None:
        tag = page.tags.get("SMaxSampleValue") or page.tags.get("MaxSampleValue")
        if tag is None:
            return None
        value = tag.value
        scale_value = value[0] if isinstance(value, (tuple, list)) else value
        if not isinstance(scale_value, (int, float, np.integer, np.floating)):
            return None
        scale = float(scale_value)
        return scale if scale > 0 else None


class EightBitRasterImageFileFormat(ImageFileFormat):
    """Raster formats that require 8-bit file-compatible image arrays."""

    format_key = "eight_bit_raster"
    suffixes = (".bmp", ".gif", ".jpeg", ".jpg")

    def prepare(self, payload: Any) -> Any:
        return image_payload_as_uint8(require_single_image_payload(payload))


class PngImageFileFormat(ImageFileFormat):
    """PNG preserves uint8/uint16 images but cannot encode float image modes."""

    format_key = "png"
    suffixes = (".png",)

    def prepare(self, payload: Any) -> Any:
        array = require_single_image_payload(payload)
        if array.dtype == np.uint8 or array.dtype == np.uint16:
            return array
        return image_payload_as_uint8(array)


class ImagePayloadUint8Strategy(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal family for dtype-specific uint8 image conversion."""

    @classmethod
    def for_dtype(cls, dtype: Any) -> "ImagePayloadUint8Strategy":
        normalized = np.dtype(dtype)
        strategy_types = cls.strategy_types_for_nominal_type(normalized.type)
        if strategy_types:
            return strategy_types[0]()
        raise TypeError(f"No uint8 conversion is registered for dtype {normalized!r}.")

    @abstractmethod
    def prepare(self, array: np.ndarray) -> np.ndarray:
        """Return a uint8-compatible image array."""


class NativeUint8ImagePayloadStrategy(ImagePayloadUint8Strategy):
    """Uint8 arrays are already compatible with 8-bit raster formats."""

    value_type = np.uint8

    def prepare(self, array: np.ndarray) -> np.ndarray:
        return array


class BoolImagePayloadUint8Strategy(ImagePayloadUint8Strategy):
    """Boolean masks serialize as black/white 8-bit images."""

    value_type = np.bool_

    def prepare(self, array: np.ndarray) -> np.ndarray:
        return array.astype(np.uint8) * np.uint8(255)


class NumericImagePayloadUint8Strategy(ImagePayloadUint8Strategy):
    """Numeric images serialize through explicit clipping/scaling semantics."""

    value_type = np.number

    def prepare(self, array: np.ndarray) -> np.ndarray:
        values = _uint8_conversion_values(array)
        if _is_unit_interval(values):
            values = values * _scale_value(values, 255.0)
        sanitized = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
        return np.rint(np.clip(sanitized, 0.0, 255.0)).astype(np.uint8)


@lru_cache(maxsize=8192)
def image_file_source_metadata(path: Path | None) -> ImageFileSourceMetadata:
    """Return source metadata through the exact registered file format."""
    if path is None or not path.exists() or not ImageFileFormat.is_image_path(path):
        return ImageFileSourceMetadata()
    return ImageFileFormat.require_path(path).source_metadata(path)


def prepare_disk_image_payloads(
    payloads: Sequence[Any],
    paths: Sequence[str | Path],
) -> list[Any]:
    """Prepare image payloads for disk paths without changing runtime values."""
    if len(payloads) != len(paths):
        raise ValueError(
            "Image payload/path length mismatch: "
            f"{len(payloads)} payloads for {len(paths)} paths."
        )
    return [
        ImageFileFormat.require_path(path).prepare(payload)
        for payload, path in zip(payloads, paths)
    ]


def image_payload_as_uint8(payload: Any) -> np.ndarray:
    """Convert numeric image payloads to uint8 using explicit file semantics."""
    array = np.asarray(image_payload_data(payload))
    return ImagePayloadUint8Strategy.for_dtype(array.dtype).prepare(array)


def require_single_image_payload(payload: Any) -> np.ndarray:
    """Return pixels only when no runtime plane axis remains to project."""
    plane_axis = image_payload_metadata(payload).plane_axis
    if plane_axis is not None:
        raise ValueError(
            "Single-image raster serialization requires a payload projected off "
            f"its declared {plane_axis.value!r} plane axis."
        )
    return np.asarray(image_payload_data(payload))


def _is_unit_interval(values: np.ndarray) -> bool:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return True
    return float(finite_values.min()) >= 0.0 and float(finite_values.max()) <= 1.0


def _uint8_conversion_values(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(array.dtype, copy=False)
    return array.astype(np.float64, copy=False)


def _scale_value(values: np.ndarray, value: float) -> Any:
    if np.issubdtype(values.dtype, np.floating):
        return values.dtype.type(value)
    return value
