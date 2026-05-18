"""Focused tests for CellProfiler SaveImages request binding."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from openhcs.core.callable_contract import CallableContract, CallableRequestBinding
from openhcs.interop.cellprofiler.image_export import (
    BitDepth,
    FileFormat,
    ImageType,
    save_images,
    save_images_3d,
)


def test_save_images_public_signature_exposes_legacy_kwargs() -> None:
    signature = inspect.signature(save_images)

    assert tuple(signature.parameters) == (
        "image",
        "filename_prefix",
        "file_format",
        "bit_depth",
        "use_compression",
        "image_type",
        "slice_by_slice",
    )
    assert isinstance(
        CallableContract.from_callable(save_images).request_binding,
        CallableRequestBinding,
    )


def test_save_images_converts_2d_image_and_metadata() -> None:
    image = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    output, metadata = save_images(
        image,
        filename_prefix="example",
        file_format=FileFormat.PNG,
        bit_depth=BitDepth.BIT_8,
    )

    np.testing.assert_array_equal(output, np.array([[0, 128], [255, 64]]))
    assert metadata.filename == "example.png"
    assert metadata.dtype == "uint8"
    assert metadata.shape_d == 1
    assert metadata.shape_h == 2
    assert metadata.shape_w == 2


def test_save_images_mask_outputs_binary_bit_depth() -> None:
    image = np.array([[0.0, 0.2], [0.0, 0.8]], dtype=np.float32)

    output, _metadata = save_images(
        image,
        bit_depth=BitDepth.BIT_16,
        image_type=ImageType.MASK,
    )

    assert set(np.unique(output)) == {0, 65535}
    assert _metadata.dtype == "uint16"


def test_save_images_3d_validates_volumetric_formats() -> None:
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="does not support 3D"):
        save_images_3d(image, file_format=FileFormat.JPEG)


def test_save_images_3d_metadata_uses_stack_shape() -> None:
    image = np.zeros((2, 3, 4), dtype=np.float32)

    output, metadata = save_images_3d(
        image,
        filename_prefix="stack",
        file_format=FileFormat.TIFF,
        bit_depth=BitDepth.RAW,
    )

    assert output.shape == (2, 3, 4)
    assert metadata.filename == "stack.tiff"
    assert metadata.shape_d == 2
    assert metadata.shape_h == 3
    assert metadata.shape_w == 4
