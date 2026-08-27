#!/bin/bash
set -euo pipefail

installer_app=${1:?A native installer application path is required}
expected_title=${2:?An expected installer window title is required}
evidence_directory=${3:?An evidence directory is required}

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
window_evidence="$evidence_directory/installer-window.json"
installer_stdout="$evidence_directory/installer-stdout.log"
installer_stderr="$evidence_directory/installer-stderr.log"
installer_screenshot="$evidence_directory/installer-welcome.png"

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
    if "$probe_executable" "$installer_pid" "$expected_title" \
        >"$window_evidence"; then
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

window_id=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["window_id"])' \
    "$window_evidence")
/usr/sbin/screencapture -x -l "$window_id" "$installer_screenshot"
if [[ ! -s "$installer_screenshot" ]]; then
    printf 'Native installer screenshot is empty: %s\n' \
        "$installer_screenshot" >&2
    exit 1
fi

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
