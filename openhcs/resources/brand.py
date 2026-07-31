"""Authoritative access to the packaged OpenHCS brand assets."""

from __future__ import annotations

import argparse
from enum import Enum
from importlib.resources import files
from pathlib import Path


class BrandAsset(str, Enum):
    """Closed set of mechanically equivalent OpenHCS mark encodings."""

    SCALABLE = "openhcs-mark.svg"
    RASTER = "openhcs-mark.png"
    WINDOWS_ICON = "openhcs.ico"
    MACOS_ICON = "openhcs.icns"


def brand_asset_path(asset: BrandAsset) -> Path:
    """Return the filesystem path for one installed OpenHCS brand asset."""

    return Path(str(files("openhcs.resources") / "assets" / asset.value))


def brand_asset_bytes(asset: BrandAsset) -> bytes:
    """Read one packaged OpenHCS brand asset."""

    return brand_asset_path(asset).read_bytes()


def main() -> int:
    """Print an installed brand asset path for native launcher integration."""

    parser = argparse.ArgumentParser(description=__doc__)
    choices = tuple(asset.name.lower() for asset in BrandAsset)
    parser.add_argument("asset", choices=choices)
    arguments = parser.parse_args()
    asset = BrandAsset[arguments.asset.upper()]
    print(brand_asset_path(asset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
