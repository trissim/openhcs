"""
Converted from CellProfiler: SaveImages
Original: SaveImages module

Note: SaveImages is fundamentally an I/O operation that saves images to disk.
In OpenHCS, this is handled by the pipeline's materialization system rather than
as a processing function. This conversion provides a pass-through function that
can be used with materialization decorators to save images.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import skimage.util
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


class BitDepth(Enum):
    BIT_8 = "8-bit integer"
    BIT_16 = "16-bit integer"
    BIT_FLOAT = "32-bit floating point"
    RAW = "No conversion"


class FileFormat(Enum):
    JPEG = "jpeg"
    NPY = "npy"
    PNG = "png"
    TIFF = "tiff"
    H5 = "h5"


class ImageType(Enum):
    IMAGE = "Image"
    MASK = "Mask"
    CROPPING = "Cropping"


@dataclass
class SaveMetadata:
    """Metadata about saved image."""
    slice_index: int
    filename: str
    bit_depth: str
    file_format: str
    shape_d: int
    shape_h: int
    shape_w: int
    dtype: str
    min_value: float
    max_value: float


@dataclass(frozen=True, slots=True)
class SaveImagesRequest:
    """Shared SaveImages conversion and metadata context."""

    image: np.ndarray
    filename_prefix: str
    file_format: FileFormat
    bit_depth: BitDepth
    use_compression: bool

    @property
    def filename(self) -> str:
        return f"{self.filename_prefix}.{self.file_format.value}"

    @property
    def conversion_strategy(self) -> "BitDepthConversionStrategy":
        return BitDepthConversionStrategy.for_bit_depth(self.bit_depth)

    def converted_image(self) -> np.ndarray:
        return self.conversion_strategy.convert(self.image)

    def converted_binary_image(self) -> np.ndarray:
        return self.conversion_strategy.binary_mask(self.converted_image())

    def metadata_for(self, output: np.ndarray) -> SaveMetadata:
        shape = output.shape
        return SaveMetadata(
            slice_index=0,
            filename=self.filename,
            bit_depth=self.bit_depth.value,
            file_format=self.file_format.value,
            shape_d=shape[0] if output.ndim == 3 else 1,
            shape_h=shape[-2],
            shape_w=shape[-1],
            dtype=str(output.dtype),
            min_value=float(np.min(output)),
            max_value=float(np.max(output)),
        )


class BitDepthConversionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal image conversion policy for one SaveImages bit depth."""

    __registry_key__ = "bit_depth_label"
    __skip_if_no_key__ = True

    bit_depth: ClassVar[BitDepth | None] = None
    bit_depth_label: ClassVar[str | None] = None

    @classmethod
    def for_bit_depth(cls, bit_depth: BitDepth) -> "BitDepthConversionStrategy":
        strategy_type = cls.__registry__.get(bit_depth.value)
        if strategy_type is None:
            raise ValueError(f"Unsupported SaveImages bit depth: {bit_depth.value!r}")
        return strategy_type()

    @abstractmethod
    def convert(self, image: np.ndarray) -> np.ndarray:
        """Return image converted to this strategy's output bit depth."""

    @abstractmethod
    def binary_mask(self, image: np.ndarray) -> np.ndarray:
        """Return binary mask/cropping output in this strategy's bit depth."""


class IntegerBitDepthConversionStrategy(BitDepthConversionStrategy):
    """Shared conversion policy for unsigned integer SaveImages outputs."""

    true_value: ClassVar[int]
    dtype: ClassVar[type[np.integer]]
    converter: ClassVar[Callable[[np.ndarray], np.ndarray]]

    def convert(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.bool_:
            return (image * self.true_value).astype(self.dtype)
        return self.converter(image)

    def binary_mask(self, image: np.ndarray) -> np.ndarray:
        return (image > 0).astype(self.dtype) * self.true_value


class EightBitConversionStrategy(IntegerBitDepthConversionStrategy):
    bit_depth = BitDepth.BIT_8
    bit_depth_label = bit_depth.value
    true_value = 255
    dtype = np.uint8
    converter = staticmethod(skimage.util.img_as_ubyte)


class SixteenBitConversionStrategy(IntegerBitDepthConversionStrategy):
    bit_depth = BitDepth.BIT_16
    bit_depth_label = bit_depth.value
    true_value = 65535
    dtype = np.uint16
    converter = staticmethod(skimage.util.img_as_uint)


class FloatBitConversionStrategy(BitDepthConversionStrategy):
    bit_depth = BitDepth.BIT_FLOAT
    bit_depth_label = bit_depth.value

    def convert(self, image: np.ndarray) -> np.ndarray:
        return skimage.util.img_as_float32(image)

    def binary_mask(self, image: np.ndarray) -> np.ndarray:
        return (image > 0).astype(np.float32)


class RawBitConversionStrategy(BitDepthConversionStrategy):
    bit_depth = BitDepth.RAW
    bit_depth_label = bit_depth.value

    def convert(self, image: np.ndarray) -> np.ndarray:
        return image.copy()

    def binary_mask(self, image: np.ndarray) -> np.ndarray:
        return (image > 0).astype(np.float32)


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("save_metadata", csv_materializer(
    fields=["slice_index", "filename", "bit_depth", "file_format", 
            "shape_d", "shape_h", "shape_w", "dtype", "min_value", "max_value"],
    analysis_type="save_images"
)))
def save_images(
    image: np.ndarray,
    filename_prefix: str = "saved_image",
    file_format: FileFormat = FileFormat.TIFF,
    bit_depth: BitDepth = BitDepth.BIT_16,
    image_type: ImageType = ImageType.IMAGE,
    use_compression: bool = True,
) -> Tuple[np.ndarray, SaveMetadata]:
    """
    Prepare image for saving with specified format and bit depth.
    
    In OpenHCS, actual file I/O is handled by the materialization system.
    This function converts the image to the appropriate bit depth and
    returns metadata about the conversion.
    
    Args:
        image: Input image array (H, W)
        filename_prefix: Prefix for output filename
        file_format: Output file format (tiff, png, jpeg, npy, h5)
        bit_depth: Bit depth for output (8-bit, 16-bit, 32-bit float, or raw)
        image_type: Type of image data (Image, Mask, Cropping)
        use_compression: Whether to use lossless compression for TIFF
    
    Returns:
        Tuple of (converted_image, save_metadata)
    """
    request = SaveImagesRequest(
        image=image,
        filename_prefix=filename_prefix,
        file_format=file_format,
        bit_depth=bit_depth,
        use_compression=use_compression,
    )
    output = request.converted_image()
    
    # Handle mask/cropping types - ensure binary output
    if image_type == ImageType.MASK or image_type == ImageType.CROPPING:
        output = request.converted_binary_image()
    
    # Generate metadata
    metadata = request.metadata_for(output)
    
    return output, metadata


@numpy(contract=ProcessingContract.PURE_3D)
@special_outputs(("save_metadata", csv_materializer(
    fields=["slice_index", "filename", "bit_depth", "file_format",
            "shape_d", "shape_h", "shape_w", "dtype", "min_value", "max_value"],
    analysis_type="save_images_3d"
)))
def save_images_3d(
    image: np.ndarray,
    filename_prefix: str = "saved_stack",
    file_format: FileFormat = FileFormat.TIFF,
    bit_depth: BitDepth = BitDepth.BIT_16,
    use_compression: bool = True,
) -> Tuple[np.ndarray, SaveMetadata]:
    """
    Prepare 3D image stack for saving.
    
    Handles volumetric data (D, H, W) for formats that support 3D:
    TIFF, NPY, and H5.
    
    Args:
        image: Input 3D image array (D, H, W)
        filename_prefix: Prefix for output filename
        file_format: Output file format (tiff, npy, h5 for 3D)
        bit_depth: Bit depth for output
        use_compression: Whether to use compression
    
    Returns:
        Tuple of (converted_image, save_metadata)
    """
    # Validate format supports 3D
    volumetric_formats = [FileFormat.TIFF, FileFormat.NPY, FileFormat.H5]
    if file_format not in volumetric_formats:
        raise ValueError(
            f"Format {file_format.value} does not support 3D. "
            f"Use one of: {[f.value for f in volumetric_formats]}"
        )
    
    request = SaveImagesRequest(
        image=image,
        filename_prefix=filename_prefix,
        file_format=file_format,
        bit_depth=bit_depth,
        use_compression=use_compression,
    )
    output = request.converted_image()

    metadata = request.metadata_for(output)

    return output, metadata
