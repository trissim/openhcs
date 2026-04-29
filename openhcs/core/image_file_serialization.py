"""File-format aware image payload preparation for disk serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
from metaclass_registry import AutoRegisterMeta


class ImageFileSerializationFormat(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for preparing image arrays for concrete file formats."""

    __registry_key__ = "format_key"
    __skip_if_no_key__ = True
    format_key: ClassVar[str | None] = None
    suffixes: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def for_path(cls, path: str | Path) -> "ImageFileSerializationFormat":
        suffix = Path(path).suffix.lower()
        for format_type in cls.__registry__.values():
            if suffix in format_type.suffixes:
                return format_type()
        return NativeImageFileSerializationFormat()

    @abstractmethod
    def prepare(self, payload: Any) -> Any:
        """Return a payload suitable for this file format."""


class NativeImageFileSerializationFormat(ImageFileSerializationFormat):
    """Formats whose writers can preserve the payload dtype directly."""

    format_key = "native"

    def prepare(self, payload: Any) -> Any:
        return payload


class EightBitRasterImageFileSerializationFormat(ImageFileSerializationFormat):
    """Raster formats that require 8-bit file-compatible image arrays."""

    format_key = "eight_bit_raster"
    suffixes = (".bmp", ".gif", ".jpeg", ".jpg")

    def prepare(self, payload: Any) -> Any:
        return image_payload_as_uint8(payload)


class PngImageFileSerializationFormat(ImageFileSerializationFormat):
    """PNG preserves uint8/uint16 images but cannot encode float image modes."""

    format_key = "png"
    suffixes = (".png",)

    def prepare(self, payload: Any) -> Any:
        array = np.asarray(payload)
        if array.dtype == np.uint8 or array.dtype == np.uint16:
            return array
        return image_payload_as_uint8(array)


class ImagePayloadUint8Strategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for dtype-specific uint8 image conversion."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[str | None] = None

    @classmethod
    def for_dtype(cls, dtype: Any) -> "ImagePayloadUint8Strategy":
        strategy_type = cls.__registry__[_UINT8_STRATEGY_KEYS.get(
            np.dtype(dtype),
            NumericImagePayloadUint8Strategy.strategy_key,
        )]
        return strategy_type()

    @abstractmethod
    def prepare(self, array: np.ndarray) -> np.ndarray:
        """Return a uint8-compatible image array."""


class NativeUint8ImagePayloadStrategy(ImagePayloadUint8Strategy):
    """Uint8 arrays are already compatible with 8-bit raster formats."""

    strategy_key = "uint8"

    def prepare(self, array: np.ndarray) -> np.ndarray:
        return array


class BoolImagePayloadUint8Strategy(ImagePayloadUint8Strategy):
    """Boolean masks serialize as black/white 8-bit images."""

    strategy_key = "bool"

    def prepare(self, array: np.ndarray) -> np.ndarray:
        return array.astype(np.uint8) * np.uint8(255)


class NumericImagePayloadUint8Strategy(ImagePayloadUint8Strategy):
    """Numeric images serialize through explicit clipping/scaling semantics."""

    strategy_key = "numeric"

    def prepare(self, array: np.ndarray) -> np.ndarray:
        values = array.astype(np.float64, copy=False)
        if _is_unit_interval(values):
            values = values * 255.0

        sanitized = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
        return np.rint(np.clip(sanitized, 0.0, 255.0)).astype(np.uint8)


_UINT8_STRATEGY_KEYS = {
    np.dtype(np.uint8): NativeUint8ImagePayloadStrategy.strategy_key,
    np.dtype(np.bool_): BoolImagePayloadUint8Strategy.strategy_key,
}


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
        ImageFileSerializationFormat.for_path(path).prepare(payload)
        for payload, path in zip(payloads, paths)
    ]


def image_payload_as_uint8(payload: Any) -> np.ndarray:
    """Convert numeric image payloads to uint8 using explicit file semantics."""
    array = np.asarray(payload)
    return ImagePayloadUint8Strategy.for_dtype(array.dtype).prepare(array)


def _is_unit_interval(values: np.ndarray) -> bool:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return True
    return float(finite_values.min()) >= 0.0 and float(finite_values.max()) <= 1.0
