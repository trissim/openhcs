#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s <installer-app> <output-dmg>\n' "$0" >&2
  exit 2
fi

installer_app=$1
output_dmg=$2
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$script_directory/dmg-lifecycle.sh"

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
mount_point="$build_root/mount"
writable_dmg="$build_root/OpenHCS-macOS-Installer-writable.dmg"
mounted_device=
mounted_volume=

cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$mounted_device" ]]; then
    if ! openhcs_detach_disk_image "$mounted_device" "$mounted_volume"; then
      printf 'Could not detach %s; preserving build directory at %s.\n' \
        "$mounted_device" "$build_root" >&2
      exit 1
    fi
  fi
  rm -rf "$build_root"
  exit "$exit_code"
}
trap cleanup EXIT

mkdir -p "$mount_point" "$(dirname "$output_dmg")"
payload_size_kib=$(du -sk "$installer_app" | awk '{print $1}')
image_size_kib=$((payload_size_kib + payload_size_kib / 4 + 32768))
if ((image_size_kib < 65536)); then
  image_size_kib=65536
fi

hdiutil create \
  -volname "OpenHCS Installer" \
  -size "${image_size_kib}k" \
  -fs APFS \
  -type UDIF \
  "$writable_dmg"
attachment=$(openhcs_attach_writable_disk_image \
  "$writable_dmg" \
  "$mount_point")
IFS=$'\t' read -r mounted_device mounted_volume <<< "$attachment"

ditto "$installer_app" "$mount_point/OpenHCS Installer.app"
ditto "$installer_icon" "$mount_point/.VolumeIcon.icns"

# Finder reads the custom-icon attribute from the mounted volume itself. A flag
# applied only to a staging directory is not preserved in the final image.
xcrun SetFile -a V "$mount_point/.VolumeIcon.icns"
xcrun SetFile -a C "$mount_point"
xcrun GetFileInfo -a "$mount_point" | grep -q C

openhcs_detach_disk_image "$mounted_device" "$mounted_volume"
mounted_device=
mounted_volume=

hdiutil convert \
  "$writable_dmg" \
  -format UDZO \
  -ov \
  -o "$output_dmg"
hdiutil verify "$output_dmg"
