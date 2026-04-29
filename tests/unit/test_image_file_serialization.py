import numpy as np

from openhcs.core.image_file_serialization import prepare_disk_image_payloads


def test_jpeg_disk_serialization_scales_unit_float_image_to_uint8() -> None:
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.JPG",))

    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared, np.array([[0, 128, 255]], dtype=np.uint8))


def test_jpeg_disk_serialization_clips_non_unit_float_image_to_uint8() -> None:
    image = np.array([[-5.0, 12.2, 300.0]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.jpg",))

    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared, np.array([[0, 12, 255]], dtype=np.uint8))


def test_png_disk_serialization_preserves_uint16_image() -> None:
    image = np.array([[0, 1024]], dtype=np.uint16)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.png",))

    assert prepared is image


def test_tiff_disk_serialization_preserves_float_payload() -> None:
    image = np.array([[0.25]], dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.tif",))

    assert prepared is image
