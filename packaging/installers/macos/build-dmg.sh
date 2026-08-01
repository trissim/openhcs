#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s <installer-app> <output-dmg>\n' "$0" >&2
  exit 2
fi

installer_app=$1
output_dmg=$2

if [[ ! -d "$installer_app" ]]; then
  printf 'Installer application does not exist: %s\n' "$installer_app" >&2
  exit 1
fi

installer_icon="$installer_app/Contents/Resources/OpenHCS.icns"
if [[ ! -f "$installer_icon" ]]; then
  printf 'Installer application icon does not exist: %s\n' "$installer_icon" >&2
  exit 1
fi

build_root=$(mktemp -d "${TMPDIR:-/tmp}/openhcs-dmg-build.XXXXXX")
source_root="$build_root/source"
mount_point="$build_root/mount"
writable_dmg="$build_root/OpenHCS-macOS-Installer-writable.dmg"
mounted=false

cleanup() {
  if [[ "$mounted" == true ]]; then
    hdiutil detach "$mount_point" || true
  fi
  rm -rf "$build_root"
}
trap cleanup EXIT

mkdir -p "$source_root" "$mount_point" "$(dirname "$output_dmg")"
ditto "$installer_app" "$source_root/OpenHCS Installer.app"
ditto "$installer_icon" "$source_root/.VolumeIcon.icns"

hdiutil create \
  -volname "OpenHCS Installer" \
  -srcfolder "$source_root" \
  -format UDRW \
  "$writable_dmg"
hdiutil attach \
  -nobrowse \
  -readwrite \
  -mountpoint "$mount_point" \
  "$writable_dmg"
mounted=true

# Finder reads the custom-icon attribute from the mounted volume itself. A flag
# applied only to the source directory is not preserved when hdiutil creates an
# APFS image.
xcrun SetFile -a V "$mount_point/.VolumeIcon.icns"
xcrun SetFile -a C "$mount_point"
xcrun GetFileInfo -a "$mount_point" | grep -q C

hdiutil detach "$mount_point"
mounted=false

hdiutil convert \
  "$writable_dmg" \
  -format UDZO \
  -ov \
  -o "$output_dmg"
hdiutil verify "$output_dmg"
