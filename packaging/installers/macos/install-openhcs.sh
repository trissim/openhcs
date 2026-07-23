#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /path/to/installer_contract.json\n' "$0" >&2
    exit 2
fi

contract_path=$1
if [[ ! -f "$contract_path" ]]; then
    printf 'Installer contract not found: %s\n' "$contract_path" >&2
    exit 2
fi

contract_value() {
    /usr/bin/plutil -extract "$1" raw -o - "$contract_path"
}

schema_version=$(contract_value schema_version)
product_name=$(contract_value product_name)
python_version=$(contract_value python_version)
package_requirement=$(contract_value package_requirement)
entry_point=$(contract_value entry_point)
uv_installer_url=$(contract_value uv_installer_urls.macos)

if [[ "$schema_version" != 'openhcs.installer.v1' ]]; then
    printf 'Unsupported installer contract schema: %s\n' "$schema_version" >&2
    exit 2
fi
if [[ ! "$product_name" =~ ^[A-Za-z0-9][A-Za-z0-9._\ -]*$ ]]; then
    printf 'Unsafe product_name in installer contract.\n' >&2
    exit 2
fi
if [[ ! "$python_version" =~ ^3\.[0-9]+$ ]]; then
    printf 'python_version must select one Python 3 minor.\n' >&2
    exit 2
fi
if [[ ! "$package_requirement" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*(\[[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*\])?([\<\>\=\!\~]=?[A-Za-z0-9.*+!_-]+)?$ ]]; then
    printf 'Unsafe package_requirement in installer contract.\n' >&2
    exit 2
fi
if [[ ! "$entry_point" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    printf 'Unsafe entry_point in installer contract.\n' >&2
    exit 2
fi
if [[ ! "$uv_installer_url" =~ ^https://astral\.sh/[^[:space:]]+$ ]]; then
    printf 'The uv installer URL must use the official Astral HTTPS host.\n' >&2
    exit 2
fi

application_root="$HOME/Library/Application Support/$product_name"
environment_root="$application_root/environments"
bootstrap_root="$application_root/bootstrap"
uv_root="$bootstrap_root/uv"
python_root="$application_root/python"
log_root="$HOME/Library/Logs/$product_name"
log_path="$log_root/installer.log"
applications_root="$HOME/Applications"
desktop_root="$HOME/Desktop"
launcher_app="$applications_root/$product_name.app"
desktop_link="$desktop_root/$product_name.app"
current_environment="$application_root/current"
current_candidate="$application_root/.current.new.$$"
install_id=$(/bin/date -u '+%Y%m%dT%H%M%SZ')-$$
new_environment="$environment_root/$install_id"
temporary_uv_installer=$(
    /usr/bin/mktemp "${TMPDIR:-/tmp}/openhcs-uv-installer.XXXXXX"
)
new_launcher_app="$applications_root/.$product_name.app.new.$$"
launcher_created=false
install_succeeded=false

/bin/mkdir -p "$application_root" "$environment_root" "$bootstrap_root" \
    "$uv_root" "$python_root" "$log_root" "$applications_root" "$desktop_root"
/usr/bin/touch "$log_path"
exec >>"$log_path" 2>&1

cleanup() {
    /bin/rm -f "$temporary_uv_installer" "$current_candidate"
    /bin/rm -rf "$new_launcher_app"
    if [[ "$install_succeeded" != true && -L "$current_environment" ]] && \
        [[ "$(/usr/bin/readlink "$current_environment")" == "$new_environment" ]]; then
        install_succeeded=true
    fi
    if [[ "$install_succeeded" != true ]]; then
        /bin/rm -rf "$new_environment"
        if [[ "$launcher_created" == true ]]; then
            /bin/rm -rf "$launcher_app"
        fi
    fi
}
trap cleanup EXIT HUP INT TERM

printf '%s Starting %s installation.\n' \
    "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')" "$product_name"
printf 'Downloading the official uv installer.\n'
/usr/bin/curl --fail --location --retry 3 \
    --proto '=https' --tlsv1.2 --output "$temporary_uv_installer" \
    "$uv_installer_url"

export UV_INSTALL_DIR="$uv_root"
export UV_NO_MODIFY_PATH=1
export UV_NO_CONFIG=1
export UV_PYTHON_INSTALL_DIR="$python_root"
/bin/sh "$temporary_uv_installer"

uv_executable="$uv_root/uv"
if [[ ! -x "$uv_executable" ]]; then
    printf 'uv installer did not create %s.\n' "$uv_executable" >&2
    exit 1
fi

"$uv_executable" --no-config python install "$python_version"
"$uv_executable" --no-config venv --python "$python_version" "$new_environment"
environment_python="$new_environment/bin/python"
"$uv_executable" --no-config pip install --python "$environment_python" \
    --upgrade "$package_requirement"
"$uv_executable" --no-config pip check --python "$environment_python"

installed_entry="$new_environment/bin/$entry_point"
if [[ ! -x "$installed_entry" ]]; then
    printf 'Installation did not create the declared GUI entry point.\n' >&2
    exit 1
fi

environment_launcher="$new_environment/launch-openhcs.sh"
/bin/cat >"$environment_launcher" <<LAUNCHER
#!/bin/bash
set -euo pipefail
export OPENHCS_CPU_ONLY=true
environment_path=\$(cd "\$(dirname "\$0")" && pwd)
exec "\$environment_path/bin/$entry_point" "\$@"
LAUNCHER
/bin/chmod 755 "$environment_launcher"

if [[ -e "$current_environment" && ! -L "$current_environment" ]]; then
    printf 'Refusing to replace non-link current environment path: %s\n' \
        "$current_environment" >&2
    exit 1
fi

if [[ -e "$launcher_app" && ! -d "$launcher_app" ]]; then
    printf 'Refusing to replace non-application launcher path: %s\n' \
        "$launcher_app" >&2
    exit 1
fi

if [[ ! -e "$launcher_app" ]]; then
    /bin/rm -rf "$new_launcher_app"
    /bin/mkdir -p "$new_launcher_app/Contents/MacOS"
    /bin/cat >"$new_launcher_app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key><string>$product_name</string>
  <key>CFBundleExecutable</key><string>launch-openhcs</string>
  <key>CFBundleIdentifier</key><string>org.openhcs.desktop</string>
  <key>CFBundleName</key><string>$product_name</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
</dict>
</plist>
PLIST
    /bin/cat >"$new_launcher_app/Contents/MacOS/launch-openhcs" <<LAUNCH_APP
#!/bin/bash
exec "\$HOME/Library/Application Support/$product_name/current/launch-openhcs.sh" "\$@"
LAUNCH_APP
    /bin/chmod 755 "$new_launcher_app/Contents/MacOS/launch-openhcs"
    /usr/bin/plutil -lint "$new_launcher_app/Contents/Info.plist"
    /bin/mv "$new_launcher_app" "$launcher_app"
    launcher_created=true
fi

/bin/ln -s "$new_environment" "$current_candidate"
if ! /bin/mv -fh "$current_candidate" "$current_environment"; then
    exit 1
fi
install_succeeded=true

if [[ -L "$desktop_link" ]]; then
    /bin/rm -f "$desktop_link"
fi
if [[ ! -e "$desktop_link" ]]; then
    /bin/ln -s "$launcher_app" "$desktop_link" || \
        printf 'WARNING: Could not create Desktop shortcut.\n'
else
    printf 'WARNING: Desktop shortcut path already exists; leaving it unchanged.\n'
fi

printf '%s Installation completed successfully.\n' \
    "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')"
