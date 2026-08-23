#!/usr/bin/env bash

# Shared macOS disk-image mount lifecycle for installer builds and verification.

_openhcs_attach_disk_image() {
  local access_mode=$1
  local image_path=$2
  local mount_point=$3
  local attach_plist
  local mounted_device

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
  printf '%s\n' "$mounted_device"
}

openhcs_attach_writable_disk_image() {
  _openhcs_attach_disk_image -readwrite "$1" "$2"
}

openhcs_attach_readonly_disk_image() {
  _openhcs_attach_disk_image -readonly "$1" "$2"
}

openhcs_detach_disk_image() {
  local mounted_device=$1
  local attempt

  for attempt in 1 2 3; do
    if /usr/bin/hdiutil detach "$mounted_device"; then
      return 0
    fi
    if ! /usr/sbin/diskutil info "$mounted_device" >/dev/null 2>&1; then
      return 0
    fi
    /bin/sleep 1
  done
  /usr/bin/hdiutil detach -force "$mounted_device"
}
