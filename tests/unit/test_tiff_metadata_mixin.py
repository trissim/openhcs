from pathlib import Path

import pytest
import tifffile

from openhcs.microscopes.exceptions import MicroscopePixelSizeUnavailableError
from openhcs.microscopes.tiff_metadata_mixin import TiffPixelSizeMixin


class _FileManager:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def list_image_files(self, plate_path, backend, *, extensions, recursive):
        del plate_path, backend, extensions, recursive
        return [self.image_path]


def test_tiff_pixel_size_missing_metadata_raises_typed_exception(tmp_path: Path):
    image_path = tmp_path / "A14_s001_w1_z001_t001.tif"
    tifffile.imwrite(image_path, [[1]])

    with pytest.raises(MicroscopePixelSizeUnavailableError) as exc_info:
        TiffPixelSizeMixin()._pixel_size_from_tiff(
            tmp_path,
            _FileManager(image_path),
        )

    assert exc_info.value.image_path == image_path
    assert str(image_path) in str(exc_info.value)
