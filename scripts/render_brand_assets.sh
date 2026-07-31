#!/bin/bash

set -euo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
asset_directory="$repository_root/openhcs/resources/assets"
source_svg="$asset_directory/openhcs-mark.svg"
raster_png="$asset_directory/openhcs-mark.png"
windows_icon="$asset_directory/openhcs.ico"
macos_icon="$asset_directory/openhcs.icns"

for executable in rsvg-convert python; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Required brand renderer is unavailable: %s\n' "$executable" >&2
        exit 2
    fi
done

rsvg-convert --width 1024 --height 1024 "$source_svg" --output "$raster_png"
python - "$raster_png" "$windows_icon" "$macos_icon" <<'PY'
from pathlib import Path
import sys

from PIL import Image

source = Path(sys.argv[1])
windows_destination = Path(sys.argv[2])
macos_destination = Path(sys.argv[3])
with Image.open(source) as image:
    image.save(
        windows_destination,
        format="ICO",
        sizes=((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)),
    )
    image.save(macos_destination, format="ICNS")
PY
