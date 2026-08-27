#!/bin/bash
set -euo pipefail

installer_app=${1:?A native installer application path is required}
expected_title=${2:?An expected installer window title is required}
evidence_directory=${3:?An evidence directory is required}
completion_log=${4:?The durable installer log path is required}

script_directory=$(cd "$(dirname "$0")" && pwd)
installer_executable="$installer_app/Contents/MacOS/OpenHCSInstaller"
probe_source="$script_directory/macos_installer_window_probe.swift"

if [[ ! -x "$installer_executable" ]]; then
    printf 'Native installer executable is unavailable: %s\n' \
        "$installer_executable" >&2
    exit 2
fi
if [[ ! -f "$probe_source" ]]; then
    printf 'Native installer window probe is unavailable: %s\n' \
        "$probe_source" >&2
    exit 2
fi

/bin/mkdir -p "$evidence_directory"
probe_executable="$evidence_directory/macos-installer-window-probe"
installer_stdout="$evidence_directory/installer-stdout.log"
installer_stderr="$evidence_directory/installer-stderr.log"

/usr/bin/xcrun --sdk macosx swiftc \
    -O \
    -framework CoreGraphics \
    "$probe_source" \
    -o "$probe_executable"

"$installer_executable" >"$installer_stdout" 2>"$installer_stderr" &
installer_pid=$!
cleanup() {
    if /bin/kill -0 "$installer_pid" 2>/dev/null; then
        /bin/kill -TERM "$installer_pid" 2>/dev/null || true
        wait "$installer_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

window_ready=false
for _ in {1..120}; do
    if ! /bin/kill -0 "$installer_pid" 2>/dev/null; then
        printf 'Native installer exited before showing %s.\n' \
            "$expected_title" >&2
        /bin/cat "$installer_stdout" >&2
        /bin/cat "$installer_stderr" >&2
        exit 1
    fi
    if "$probe_executable" \
        "$installer_pid" \
        "$expected_title" \
        inspect \
        >"$evidence_directory/installer-window.json"; then
        window_ready=true
        break
    fi
    /bin/sleep 0.25
done
if [[ "$window_ready" != true ]]; then
    printf 'Native installer did not show %s within 30 seconds.\n' \
        "$expected_title" >&2
    exit 1
fi

capture_installer_window() {
    local evidence_name=$1
    local screenshot_name=$2
    local window_evidence="$evidence_directory/$evidence_name.json"
    local installer_screenshot="$evidence_directory/$screenshot_name.png"
    "$probe_executable" \
        "$installer_pid" \
        "$expected_title" \
        inspect \
        >"$window_evidence"
    local window_id
    window_id=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["window_id"])' \
        "$window_evidence")
    /usr/sbin/screencapture -x -l "$window_id" "$installer_screenshot"
    if [[ ! -s "$installer_screenshot" ]]; then
        printf 'Native installer screenshot is empty: %s\n' \
            "$installer_screenshot" >&2
        exit 1
    fi
}

wait_for_installer_log() {
    local expected_line=$1
    local description=$2
    local deadline=$3
    local timeout_description=$4
    while (( SECONDS < deadline )); do
        if ! /bin/kill -0 "$installer_pid" 2>/dev/null; then
            printf 'Native installer exited before %s.\n' "$description" >&2
            /bin/cat "$installer_stdout" >&2
            /bin/cat "$installer_stderr" >&2
            exit 1
        fi
        if [[ -f "$completion_log" ]] && \
            /usr/bin/grep -Fq "$expected_line" "$completion_log"; then
            return
        fi
        /bin/sleep 0.25
    done
    printf 'Native installer did not reach %s within %s.\n' \
        "$description" "$timeout_description" >&2
    exit 1
}

capture_installer_window installer-window installer-welcome
"$probe_executable" \
    "$installer_pid" \
    "$expected_title" \
    press-primary \
    >/dev/null

interaction_timeout_seconds=30
interaction_deadline=$((SECONDS + interaction_timeout_seconds))
wait_for_installer_log \
    " Starting " \
    "visible installation progress" \
    "$interaction_deadline" \
    "$interaction_timeout_seconds seconds"
/bin/sleep 0.5
capture_installer_window installer-progress installer-progress

installation_timeout_seconds=1200
installation_deadline=$((SECONDS + installation_timeout_seconds))
wait_for_installer_log \
    " Installation completed successfully." \
    "successful completion" \
    "$installation_deadline" \
    "$((installation_timeout_seconds / 60)) minutes"
/bin/sleep 1
capture_installer_window installer-finished installer-finished
/bin/cp "$completion_log" "$evidence_directory/installer.log"

/bin/kill -TERM "$installer_pid"
set +e
wait "$installer_pid"
installer_status=$?
set -e
trap - EXIT HUP INT TERM
if [[ "$installer_status" -ne 0 && "$installer_status" -ne 143 ]]; then
    printf 'Native installer exited with status %s after its window closed.\n' \
        "$installer_status" >&2
    exit 1
fi
