#!/bin/bash

set -euo pipefail

script_directory=$(cd "$(dirname "$0")" && pwd)
contract_path=${1:-"$script_directory/../installer_contract.json"}
output_path=${2:-"$script_directory/dist/OpenHCS Installer.app"}

if [[ ! -f "$contract_path" ]]; then
    printf 'Installer contract not found: %s\n' "$contract_path" >&2
    exit 2
fi
if ! command -v osacompile >/dev/null 2>&1; then
    printf 'osacompile is required to build the macOS installer app.\n' >&2
    exit 2
fi

output_parent=$(dirname "$output_path")
/bin/mkdir -p "$output_parent"
if [[ -e "$output_path" ]]; then
    printf 'Refusing to replace existing output: %s\n' "$output_path" >&2
    exit 2
fi

/usr/bin/osacompile -o "$output_path" "$script_directory/Install-OpenHCS.applescript"
/bin/cp "$script_directory/install-openhcs.sh" \
    "$output_path/Contents/Resources/install-openhcs.sh"
/bin/cp "$contract_path" "$output_path/Contents/Resources/installer_contract.json"
/bin/chmod 755 "$output_path/Contents/Resources/install-openhcs.sh"

printf 'Built %s\n' "$output_path"
