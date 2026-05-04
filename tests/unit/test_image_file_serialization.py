import numpy as np

from openhcs.core.image_file_serialization import prepare_disk_image_payloads
from openhcs.core.runtime_values import ImageMetadataPayload, ImagePayloadMetadata


def test_jpeg_disk_serialization_scales_unit_float_image_to_uint8() -> None:
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.JPG",))

    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared, np.array([[0, 128, 255]], dtype=np.uint8))


def test_png_disk_serialization_preserves_float32_quantization() -> None:
    image = np.array([[np.float32(2.5 / 255.0)]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.png",))

    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared, np.array([[2]], dtype=np.uint8))


def test_jpeg_disk_serialization_clips_non_unit_float_image_to_uint8() -> None:
    image = np.array([[-5.0, 12.2, 300.0]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.jpg",))

    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared, np.array([[0, 12, 255]], dtype=np.uint8))


def test_png_disk_serialization_preserves_uint16_image() -> None:
    image = np.array([[0, 1024]], dtype=np.uint16)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.png",))

    assert prepared is image


def test_png_disk_serialization_collapses_singleton_color_stack() -> None:
    image = np.zeros((1, 3, 4, 3), dtype=np.uint8)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.png",))

    assert prepared.shape == (3, 4, 3)
    assert prepared.dtype == np.uint8


def test_jpeg_disk_serialization_collapses_singleton_grayscale_stack() -> None:
    image = np.ones((1, 3, 5), dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.jpg",))

    assert prepared.shape == (3, 5)
    assert prepared.dtype == np.uint8


def test_png_disk_serialization_preserves_one_row_color_slice() -> None:
    image = np.zeros((1, 4, 3), dtype=np.uint8)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.png",))

    assert prepared.shape == (1, 4, 3)
    assert prepared.dtype == np.uint8


def test_tiff_disk_serialization_preserves_float_payload() -> None:
    image = np.array([[0.25]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.tif",))

    assert prepared is image


def test_native_disk_serialization_unwraps_image_metadata_payload() -> None:
    image = np.array([[0.25]], dtype=np.float32)
    payload = ImageMetadataPayload(
        image,
        ImagePayloadMetadata(source_dtype="float32"),
    )

    (prepared,) = prepare_disk_image_payloads((payload,), ("out.tif",))

    assert prepared is image


def test_png_disk_serialization_unwraps_image_metadata_payload() -> None:
    image = np.array([[[0, 1024]]], dtype=np.uint16)
    payload = ImageMetadataPayload(
        image,
        ImagePayloadMetadata(source_dtype="uint16"),
    )

    (prepared,) = prepare_disk_image_payloads((payload,), ("out.png",))

    assert prepared.shape == (1, 2)
    assert prepared.dtype == np.uint16
