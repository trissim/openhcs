"""
Shared helper for reading pixel size (and optional channel name) from TIFF tags.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import re
import tifffile

from openhcs.microscopes.exceptions import MicroscopePixelSizeUnavailableError

class TiffPixelSizeMixin:
    """Utility mixin to extract pixel size and channel name from TIFF metadata."""

    def _first_tiff(self, plate_path, filemanager, extensions=None) -> Path:
        exts = extensions or {".tif", ".tiff"}
        images = filemanager.list_image_files(plate_path, "disk", extensions=exts, recursive=True)
        if not images:
            raise FileNotFoundError(f"No TIFF images found in {plate_path}")
        return Path(images[0])

    def _pixel_size_from_tiff(self, plate_path, filemanager) -> float:
        img = self._first_tiff(plate_path, filemanager)
        with tifffile.TiffFile(img) as tif:
            page = tif.pages[0]
            uic = page.tags.get("UIC1tag")
            if uic:
                data = uic.value
                if "XCalibration" in data:
                    return float(data["XCalibration"])
            desc = page.tags.get("ImageDescription")
            if desc:
                text = desc.value
                if isinstance(text, bytes):
                    text = text.decode(errors="ignore")
                m = re.search(r"spatial[- ]calibration[- ]x[^0-9]*([0-9.]+)", text, re.IGNORECASE)
                if m:
                    return float(m.group(1))
        raise MicroscopePixelSizeUnavailableError(img)

    def _channel_from_tiff(self, plate_path, filemanager) -> Optional[Dict[str, Optional[str]]]:
        img = self._first_tiff(plate_path, filemanager)
        with tifffile.TiffFile(img) as tif:
            page = tif.pages[0]
            uic = page.tags.get("UIC1tag")
            if uic and "Name" in uic.value:
                return {"1": str(uic.value["Name"])}
            desc = page.tags.get("ImageDescription")
            if desc:
                text = desc.value
                if isinstance(text, bytes):
                    text = text.decode(errors="ignore")
                m = re.search(r"Name:\\s*([A-Za-z0-9_ +-]+)", text)
                if m:
                    return {"1": m.group(1).strip()}
        return None
