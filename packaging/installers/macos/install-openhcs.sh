#!/bin/bash

set -euo pipefail

installer_state_directory=${OPENHCS_INSTALLER_STATE_DIRECTORY:-}
register_mcp_clients=${OPENHCS_INSTALLER_REGISTER_MCP_CLIENTS:-1}

write_installer_state() {
    local state_name=$1
    local state_value=$2
    local temporary_state

    if [[ -z "$installer_state_directory" ]]; then
        return
    fi
    if [[ ! -d "$installer_state_directory" || -L "$installer_state_directory" ]]; then
        printf 'Unsafe installer state directory: %s\n' \
            "$installer_state_directory" >&2
        exit 2
    fi
    case "$state_name" in
        progress | log-path | launcher-path | agent-registration-status | \
            agent-registration-summary) ;;
        *)
            printf 'Unsupported installer state name: %s\n' "$state_name" >&2
            exit 2
            ;;
    esac

    temporary_state="$installer_state_directory/.$state_name.$$"
    printf '%s\n' "$state_value" >"$temporary_state"
    /bin/chmod 600 "$temporary_state"
    /bin/mv -f "$temporary_state" "$installer_state_directory/$state_name"
}

report_progress() {
    printf '%s\n' "$1"
    write_installer_state progress "$1"
}

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /path/to/installer_contract.json\n' "$0" >&2
    exit 2
fi
if [[ "$register_mcp_clients" != 0 && "$register_mcp_clients" != 1 ]]; then
    printf 'OPENHCS_INSTALLER_REGISTER_MCP_CLIENTS must be 0 or 1.\n' >&2
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
uv_version=$(contract_value uv_release.version)
uv_base_url=$(contract_value uv_release.base_url)

if [[ "$schema_version" != 'openhcs.installer.v2' ]]; then
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
if [[ ! "$uv_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf 'uv_release.version must be stable SemVer.\n' >&2
    exit 2
fi
if [[ "$uv_base_url" != 'https://astral.sh/uv' ]]; then
    printf 'uv_release.base_url must use the official Astral uv endpoint.\n' >&2
    exit 2
fi
uv_installer_url="$uv_base_url/$uv_version/install.sh"

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
agent_registration_report="$application_root/agent-registration.json"
agent_registration_candidate="$application_root/.agent-registration.new.$$"
launcher_created=false
install_succeeded=false
active_child_pid=

/bin/mkdir -p "$application_root" "$environment_root" "$bootstrap_root" \
    "$uv_root" "$python_root" "$log_root" "$applications_root" "$desktop_root"
if [[ -L "$log_path" ]]; then
    printf 'Refusing symbolic-link installer log path: %s\n' "$log_path" >&2
    exit 2
fi
/usr/bin/touch "$log_path"
if [[ ! -f "$log_path" ]]; then
    printf 'Installer log is not a regular file: %s\n' "$log_path" >&2
    exit 2
fi
write_installer_state log-path "$log_path"
exec >>"$log_path" 2>&1

cleanup() {
    /bin/rm -f "$temporary_uv_installer" "$current_candidate" \
        "$agent_registration_candidate"
    /bin/rm -rf "$new_launcher_app"
    if [[ "$install_succeeded" != true && -L "$current_environment" ]] && \
        [[ "$(/usr/bin/readlink "$current_environment")" == "$new_environment" ]]; then
        install_succeeded=true
    fi
    if [[ "$install_succeeded" == true ]]; then
        write_installer_state launcher-path "$launcher_app"
    fi
    if [[ "$install_succeeded" != true ]]; then
        /bin/rm -rf "$new_environment"
        if [[ "$launcher_created" == true ]]; then
            /bin/rm -rf "$launcher_app"
        fi
    fi
}

run_cancellable() {
    local child_status

    "$@" &
    active_child_pid=$!
    if wait "$active_child_pid"; then
        child_status=0
    else
        child_status=$?
    fi
    active_child_pid=
    return "$child_status"
}

child_is_running() {
    local child_pid=$1
    local process_state

    process_state=$(/bin/ps -o state= -p "$child_pid" 2>/dev/null) || return 1
    [[ -n "$process_state" && "$process_state" != *Z* ]]
}

terminate_active_child() {
    local child_pid=$1
    local poll_attempt

    /bin/kill -TERM "$child_pid" 2>/dev/null || true
    for poll_attempt in {1..20}; do
        if ! child_is_running "$child_pid"; then
            break
        fi
        /bin/sleep 0.1
    done
    if child_is_running "$child_pid"; then
        /bin/kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
}

cancel_install() {
    local child_pid=$active_child_pid

    trap - HUP INT TERM
    if [[ -n "$child_pid" ]] && /bin/kill -0 "$child_pid" 2>/dev/null; then
        terminate_active_child "$child_pid"
    fi
    exit 130
}

trap cleanup EXIT
trap cancel_install HUP INT TERM

printf '%s Starting %s installation.\n' \
    "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')" "$product_name"
report_progress 'Downloading the secure installer components…'
printf 'Using pinned official uv %s.\n' "$uv_version"
run_cancellable /usr/bin/curl --fail --location --retry 3 \
    --proto '=https' --tlsv1.2 --output "$temporary_uv_installer" \
    "$uv_installer_url"

export UV_INSTALL_DIR="$uv_root"
export UV_NO_MODIFY_PATH=1
export UV_NO_CONFIG=1
export UV_PYTHON_INSTALL_DIR="$python_root"
report_progress 'Preparing the private package manager…'
run_cancellable /bin/sh "$temporary_uv_installer"

uv_executable="$uv_root/uv"
if [[ ! -x "$uv_executable" ]]; then
    printf 'uv installer did not create %s.\n' "$uv_executable" >&2
    exit 1
fi

report_progress 'Installing a private Python environment…'
run_cancellable "$uv_executable" --no-config python install "$python_version"
report_progress 'Creating the application environment…'
run_cancellable "$uv_executable" --no-config venv \
    --python "$python_version" "$new_environment"
environment_python="$new_environment/bin/python"
report_progress "Installing $product_name and its desktop features…"
run_cancellable "$uv_executable" --no-config pip install \
    --python "$environment_python" \
    --upgrade "$package_requirement"
report_progress 'Verifying the installed application…'
run_cancellable "$uv_executable" --no-config pip check \
    --python "$environment_python"

installed_entry="$new_environment/bin/$entry_point"
if [[ ! -x "$installed_entry" ]]; then
    printf 'Installation did not create the declared GUI entry point.\n' >&2
    exit 1
fi

stable_mcp_launcher="$current_environment/launch-openhcs.sh"
stable_launch_command_json=$(
    "$environment_python" -c \
        'import json,sys; print(json.dumps([sys.argv[1], "mcp"], separators=(",", ":")))' \
        "$stable_mcp_launcher"
)
printf -v stable_launch_command_shell '%q' "$stable_launch_command_json"
printf -v installation_pointer_shell '%q' "$current_environment"
environment_launcher="$new_environment/launch-openhcs.sh"
/bin/cat >"$environment_launcher" <<LAUNCHER
#!/bin/bash
set -euo pipefail
export OPENHCS_CPU_ONLY=true
export OPENHCS_MCP_STABLE_LAUNCH_COMMAND_JSON=$stable_launch_command_shell
export OPENHCS_MCP_INSTALLATION_POINTER=$installation_pointer_shell
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

report_progress 'Preparing Applications and Desktop shortcuts…'
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
write_installer_state launcher-path "$launcher_app"

if [[ "$register_mcp_clients" == 1 ]]; then
    report_progress 'Connecting OpenHCS to local AI agent apps…'
    registration_executable="$new_environment/bin/openhcs-mcp-register"
    if [[ ! -x "$registration_executable" ]]; then
        printf 'WARNING: Agent registration entry point is unavailable: %s\n' \
            "$registration_executable"
        write_installer_state agent-registration-status warning
    else
        if "$registration_executable" \
            --command "$current_environment/launch-openhcs.sh" \
            --args-json '["mcp"]' \
            --register codex \
            --register-detected \
            --json >"$agent_registration_candidate"; then
            registration_status=0
        else
            registration_status=$?
        fi
        /bin/chmod 600 "$agent_registration_candidate"
        /bin/cat "$agent_registration_candidate"
        registration_ok=$(
            "$environment_python" -c \
                'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["ok"])).lower())' \
                "$agent_registration_candidate" 2>/dev/null || printf 'false'
        )
        registration_summary=$(
            "$environment_python" -c \
                'import json,sys; payload=json.load(open(sys.argv[1])); print(", ".join(str(result["display_name"]) for result in payload["results"] if result["status"] != "failed"))' \
                "$agent_registration_candidate" 2>/dev/null || true
        )
        if [[ -n "$registration_summary" ]]; then
            write_installer_state agent-registration-summary "$registration_summary"
        fi
        /bin/mv -f "$agent_registration_candidate" "$agent_registration_report"
        if [[ "$registration_status" -ne 0 || "$registration_ok" != true ]]; then
            printf 'WARNING: One or more agent client registrations did not complete '
            printf '(exit code %s). ' \
                "$registration_status"
            printf '%s itself remains installed.\n' "$product_name"
            write_installer_state agent-registration-status warning
        else
            write_installer_state agent-registration-status connected
        fi
    fi
fi

if [[ -L "$desktop_link" ]]; then
    /bin/rm -f "$desktop_link"
fi
if [[ ! -e "$desktop_link" ]]; then
    /bin/ln -s "$launcher_app" "$desktop_link" || \
        printf 'WARNING: Could not create Desktop shortcut.\n'
else
    printf 'WARNING: Desktop shortcut path already exists; leaving it unchanged.\n'
fi

report_progress 'Installation complete.'
printf '%s Installation completed successfully.\n' \
    "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')"
