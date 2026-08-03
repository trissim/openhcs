"""Authoritative access to the packaged OpenHCS brand assets."""

from __future__ import annotations

import argparse
from enum import Enum
from importlib.resources import files
from pathlib import Path


BRAND_PRIMARY_COLOR = "#00AAFF"


class BrandAsset(str, Enum):
    """Closed family of official OpenHCS logo variants and native encodings."""

    SOURCE = "openhcs-logo-source.svg"
    MARK = "openhcs-mark.svg"
    MARK_MONO = "openhcs-mark-mono.svg"
    LOCKUP_HORIZONTAL = "openhcs-lockup-horizontal.svg"
    LOCKUP_STACKED = "openhcs-lockup-stacked.svg"
    ICON_SQUARE = "openhcs-icon-square.svg"
    FAVICON = "openhcs-favicon.svg"
    ICON_RASTER = "openhcs-icon-square.png"
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
