"""
Converted from CellProfiler: SaveImages
Original: SaveImages module

Note: SaveImages is fundamentally an I/O operation that saves images to disk.
In OpenHCS, this is handled by the pipeline's materialization system rather than
as a processing function. This conversion provides a pass-through function that
can be used with materialization decorators to save images.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
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
    import skimage.util
    
    # Convert image based on bit depth
    if bit_depth == BitDepth.BIT_8:
        # Convert to 8-bit unsigned integer
        if image.dtype == np.bool_:
            output = (image * 255).astype(np.uint8)
        else:
            output = skimage.util.img_as_ubyte(image)
    elif bit_depth == BitDepth.BIT_16:
        # Convert to 16-bit unsigned integer
        if image.dtype == np.bool_:
            output = (image * 65535).astype(np.uint16)
        else:
            output = skimage.util.img_as_uint(image)
    elif bit_depth == BitDepth.BIT_FLOAT:
        # Convert to 32-bit float
        output = skimage.util.img_as_float32(image)
    else:  # RAW - no conversion
        output = image.copy()
    
    # Handle mask/cropping types - ensure binary output
    if image_type == ImageType.MASK or image_type == ImageType.CROPPING:
        if bit_depth == BitDepth.BIT_8:
            output = (output > 0).astype(np.uint8) * 255
        elif bit_depth == BitDepth.BIT_16:
            output = (output > 0).astype(np.uint16) * 65535
        else:
            output = (output > 0).astype(np.float32)
    
    # Generate metadata
    metadata = SaveMetadata(
        slice_index=0,
        filename=f"{filename_prefix}.{file_format.value}",
        bit_depth=bit_depth.value,
        file_format=file_format.value,
        shape_d=1,
        shape_h=output.shape[0],
        shape_w=output.shape[1],
        dtype=str(output.dtype),
        min_value=float(np.min(output)),
        max_value=float(np.max(output))
    )
    
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
    import skimage.util
    
    # Validate format supports 3D
    volumetric_formats = [FileFormat.TIFF, FileFormat.NPY, FileFormat.H5]
    if file_format not in volumetric_formats:
        raise ValueError(
            f"Format {file_format.value} does not support 3D. "
            f"Use one of: {[f.value for f in volumetric_formats]}"
        )
    
    # Convert based on bit depth
    if bit_depth == BitDepth.BIT_8:
        output = skimage.util.img_as_ubyte(image)
    elif bit_depth == BitDepth.BIT_16:
        output = skimage.util.img_as_uint(image)
    elif bit_depth == BitDepth.BIT_FLOAT:
        output = skimage.util.img_as_float32(image)
    else:  # RAW
        output = image.copy()
    
    metadata = SaveMetadata(
        slice_index=0,
        filename=f"{filename_prefix}.{file_format.value}",
        bit_depth=bit_depth.value,
        file_format=file_format.value,
        shape_d=output.shape[0],
        shape_h=output.shape[1],
        shape_w=output.shape[2],
        dtype=str(output.dtype),
        min_value=float(np.min(output)),
        max_value=float(np.max(output))
    )
    
    return output, metadata