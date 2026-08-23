#!/usr/bin/env bash

# Shared macOS disk-image mount lifecycle for installer builds and verification.

_openhcs_attach_disk_image() {
  local access_mode=$1
  local image_path=$2
  local mount_point=$3
  local mounted_identifier

  /usr/bin/hdiutil attach \
    -nobrowse \
    "$access_mode" \
    -mountpoint "$mount_point" \
    "$image_path" >&2
  if ! mounted_identifier=$(/usr/sbin/diskutil info -plist "$mount_point" | \
    /usr/bin/plutil -extract DeviceIdentifier raw -o - -); then
    printf 'Could not resolve the device mounted from %s at %s.\n' \
      "$image_path" "$mount_point" >&2
    /usr/bin/hdiutil detach "$mount_point" || true
    return 1
  fi
  printf '/dev/%s\n' "$mounted_identifier"
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
