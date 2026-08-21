#!/bin/bash

set -euo pipefail

script_directory=$(cd "$(dirname "$0")" && pwd)
contract_path=${1:-"$script_directory/../../../openhcs/resources/installer_contract.json"}
output_path=${2:-"$script_directory/dist/OpenHCS Installer.app"}
brand_icon_path="$script_directory/../../../openhcs/resources/assets/openhcs.icns"

if [[ ! -f "$contract_path" ]]; then
    printf 'Installer contract not found: %s\n' "$contract_path" >&2
    exit 2
fi
if [[ ! -f "$brand_icon_path" ]]; then
    printf 'OpenHCS brand icon not found: %s\n' "$brand_icon_path" >&2
    exit 2
fi
if ! command -v xcrun >/dev/null 2>&1; then
    printf 'Xcode command-line tools are required to build the macOS installer app.\n' >&2
    exit 2
fi
if ! command -v lipo >/dev/null 2>&1; then
    printf 'lipo is required to build the universal macOS installer app.\n' >&2
    exit 2
fi

output_parent=$(dirname "$output_path")
/bin/mkdir -p "$output_parent"
if [[ -e "$output_path" ]]; then
    printf 'Refusing to replace existing output: %s\n' "$output_path" >&2
    exit 2
fi

temporary_directory=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/openhcs-app-build.XXXXXX")
temporary_app="$temporary_directory/OpenHCS Installer.app"
cleanup() {
    /bin/rm -rf "$temporary_directory"
}
trap cleanup EXIT HUP INT TERM

/bin/mkdir -p "$temporary_app/Contents/MacOS" "$temporary_app/Contents/Resources"
sdk_path=$(/usr/bin/xcrun --sdk macosx --show-sdk-path)
for architecture in x86_64 arm64; do
    /usr/bin/xcrun --sdk macosx swiftc \
        -O \
        -sdk "$sdk_path" \
        -target "$architecture-apple-macosx12.0" \
        "$script_directory/OpenHCSInstaller.swift" \
        -o "$temporary_directory/OpenHCSInstaller-$architecture"
done
/usr/bin/lipo -create \
    "$temporary_directory/OpenHCSInstaller-x86_64" \
    "$temporary_directory/OpenHCSInstaller-arm64" \
    -output "$temporary_app/Contents/MacOS/OpenHCSInstaller"

/bin/cat >"$temporary_app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key><string>OpenHCS Installer</string>
  <key>CFBundleExecutable</key><string>OpenHCSInstaller</string>
  <key>CFBundleIdentifier</key><string>org.openhcs.installer</string>
  <key>CFBundleIconFile</key><string>OpenHCS.icns</string>
  <key>CFBundleName</key><string>OpenHCS Installer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

/bin/cp "$script_directory/install-openhcs.sh" \
    "$temporary_app/Contents/Resources/install-openhcs.sh"
/bin/cp "$contract_path" \
    "$temporary_app/Contents/Resources/installer_contract.json"
/bin/cp "$brand_icon_path" \
    "$temporary_app/Contents/Resources/OpenHCS.icns"
/bin/chmod 755 "$temporary_app/Contents/MacOS/OpenHCSInstaller"
/bin/chmod 755 "$temporary_app/Contents/Resources/install-openhcs.sh"
/usr/bin/plutil -lint "$temporary_app/Contents/Info.plist"
/bin/mv "$temporary_app" "$output_path"

printf 'Built %s\n' "$output_path"
