import numpy as np
import pytest

from openhcs.core.image_file_serialization import (
    ImageFileFormat,
    NumpyImageFileFormat,
    PngImageFileFormat,
    TiffImageFileFormat,
    prepare_disk_image_payloads,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis


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


def test_png_disk_serialization_does_not_infer_singleton_color_stack() -> None:
    image = np.zeros((1, 3, 4, 3), dtype=np.uint8)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.png",))

    assert prepared.shape == image.shape
    assert prepared.dtype == np.uint8


def test_jpeg_disk_serialization_does_not_infer_singleton_grayscale_stack() -> None:
    image = np.ones((1, 3, 5), dtype=np.float32)

    (prepared,) = prepare_disk_image_payloads((image,), ("out.jpg",))

    assert prepared.shape == image.shape
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


def test_registered_tiff_format_round_trips_grayscale_pixels(tmp_path) -> None:
    path = tmp_path / "image.tif"
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)

    image_format = ImageFileFormat.require_path(path)
    image_format.write(path, image)

    np.testing.assert_array_equal(image_format.read(path), image)


def test_tiff_source_metadata_reads_dtype_and_declared_scale_without_imageio(
    tmp_path,
    monkeypatch,
) -> None:
    import imageio.v3 as iio
    import tifffile

    path = tmp_path / "source.tif"
    tifffile.imwrite(
        path,
        np.array([[0, 4095]], dtype=np.uint16),
        extratags=((281, "H", 1, 4095, False),),
    )
    monkeypatch.setattr(
        iio,
        "improps",
        lambda *_args, **_kwargs: pytest.fail("TIFF metadata reopened through ImageIO"),
    )

    metadata = TiffImageFileFormat().source_metadata(path)

    assert metadata.source_dtype == np.dtype(np.uint16)
    assert metadata.intensity_scale == 4095.0
    assert metadata.pixel_semantics.channel_axis is None
    assert metadata.pixel_semantics.channel_count is None


def test_tiff_source_metadata_reads_rgb_semantics_without_generic_reopen(
    tmp_path,
    monkeypatch,
) -> None:
    import tifffile

    path = tmp_path / "rgb.tiff"
    tifffile.imwrite(path, np.zeros((4, 5, 3), dtype=np.uint8), photometric="rgb")
    monkeypatch.setattr(
        TiffImageFileFormat,
        "pixel_semantics",
        lambda *_args, **_kwargs: pytest.fail(
            "TIFF source metadata reopened inherited pixel semantics"
        ),
    )

    metadata = TiffImageFileFormat().source_metadata(path)

    assert metadata.source_dtype == np.dtype(np.uint8)
    assert metadata.intensity_scale == 255.0
    assert metadata.pixel_semantics.channel_axis == -1
    assert metadata.pixel_semantics.channel_count == 3
    assert metadata.pixel_semantics.validated_channel_axis(
        tifffile.imread(path)
    ) == -1


def test_tiff_source_metadata_uses_declared_planar_sample_axis(tmp_path) -> None:
    import tifffile

    path = tmp_path / "planar-rgb.tiff"
    tifffile.imwrite(
        path,
        np.zeros((3, 4, 5), dtype=np.uint8),
        photometric="rgb",
        planarconfig="separate",
    )

    metadata = TiffImageFileFormat().source_metadata(path)

    assert metadata.pixel_semantics.channel_axis == 0
    assert metadata.pixel_semantics.channel_count == 3
    assert metadata.pixel_semantics.validated_channel_axis(tifffile.imread(path)) == 0


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

    assert prepared.shape == (1, 1, 2)
    assert prepared.dtype == np.uint16


def test_png_disk_serialization_rejects_declared_unprojected_plane_axis() -> None:
    payload = ImageMetadataPayload(
        np.zeros((1, 3, 5), dtype=np.uint8),
        ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )

    with pytest.raises(ValueError, match="projected off.*runtime_slice"):
        prepare_disk_image_payloads((payload,), ("out.png",))


def test_numpy_image_format_has_explicit_registered_suffix() -> None:
    assert isinstance(
        ImageFileFormat.require_path("out.npy"),
        NumpyImageFileFormat,
    )


@pytest.mark.parametrize("path", ("out.tif", "out.tiff"))
def test_tiff_image_format_has_explicit_registered_suffixes(path) -> None:
    assert isinstance(ImageFileFormat.require_path(path), TiffImageFileFormat)


def test_png_image_format_uses_registered_png_leaf() -> None:
    assert isinstance(
        ImageFileFormat.require_path("out.png"),
        PngImageFileFormat,
    )


@pytest.mark.parametrize("path", ("out.h5", "out.hdf5", "out.unknown"))
def test_unknown_image_serialization_suffix_fails_loudly(path) -> None:
    with pytest.raises(ValueError, match="image|suffix|format"):
        ImageFileFormat.require_path(path)
