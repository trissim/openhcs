#!/usr/bin/env bash

# Shared macOS disk-image mount lifecycle for installer builds and verification.

_openhcs_attach_disk_image() {
  local access_mode=$1
  local image_path=$2
  local mount_point=$3
  local attach_plist
  local mounted_device
  local mounted_volume

  if ! attach_plist=$(/usr/bin/hdiutil attach \
    -plist \
    -nobrowse \
    "$access_mode" \
    -mountpoint "$mount_point" \
    "$image_path"); then
    return 1
  fi
  printf '%s\n' "$attach_plist" >&2
  if ! mounted_device=$(printf '%s\n' "$attach_plist" | \
    /usr/bin/plutil -extract 'system-entities.0.dev-entry' raw -o - -); then
    printf 'Could not resolve the backing device attached from %s.\n' \
      "$image_path" >&2
    return 1
  fi
  if ! mounted_volume=$(/usr/sbin/diskutil info -plist "$mount_point" | \
    /usr/bin/plutil -extract DeviceNode raw -o - -); then
    printf 'Could not resolve the volume mounted at %s.\n' "$mount_point" >&2
    /usr/bin/hdiutil detach -force "$mounted_device" >/dev/null 2>&1 || true
    return 1
  fi
  printf '%s\t%s\n' "$mounted_device" "$mounted_volume"
}

openhcs_attach_writable_disk_image() {
  _openhcs_attach_disk_image -readwrite "$1" "$2"
}

openhcs_attach_readonly_disk_image() {
  _openhcs_attach_disk_image -readonly "$1" "$2"
}

openhcs_detach_disk_image() {
  local mounted_device=$1
  local mounted_volume=$2
  local detach_attempt_limit=10
  local attempt

  /bin/sync
  # hdiutil owns the image attachment and uses Disk Arbitration to coordinate
  # filesystem unmount with device teardown.  Unmounting the APFS volume as a
  # separate first step can strand its synthesized device while the backing
  # image remains busy.
  for ((attempt = 1; attempt <= detach_attempt_limit; attempt += 1)); do
    if /usr/bin/hdiutil detach "$mounted_device"; then
      return 0
    fi
    if ! /usr/sbin/diskutil info "$mounted_device" >/dev/null 2>&1; then
      return 0
    fi
    if ((attempt < detach_attempt_limit)); then
      /bin/sleep 1
    fi
  done
  if /usr/sbin/diskutil info "$mounted_volume" >/dev/null 2>&1; then
    /usr/sbin/diskutil unmount force "$mounted_volume"
  fi
  if /usr/bin/hdiutil detach -force "$mounted_device"; then
    return 0
  fi
  if ! /usr/sbin/diskutil info "$mounted_device" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}
