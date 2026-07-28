#!/bin/bash

set -euo pipefail

usage() {
    printf 'Usage: %s sign-app|sign-dmg-and-notarize ARTIFACT\n' "$0" >&2
    exit 2
}

require_command() {
    local command_path=$1
    if [[ ! -x "$command_path" ]]; then
        printf 'Required macOS trust tool not found: %s\n' "$command_path" >&2
        exit 2
    fi
}

require_environment() {
    local name=$1
    if [[ -z ${!name:-} ]]; then
        printf 'Required macOS installer signing secret is missing: %s\n' \
            "$name" >&2
        exit 2
    fi
}

temporary_directory=
signing_keychain=
cleanup() {
    if [[ -n "$signing_keychain" && -e "$signing_keychain" ]]; then
        /usr/bin/security delete-keychain "$signing_keychain" \
            >/dev/null 2>&1 || true
    fi
    if [[ -n "$temporary_directory" && -d "$temporary_directory" ]]; then
        /bin/rm -rf "$temporary_directory"
    fi
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

create_signing_keychain() {
    require_environment OPENHCS_MACOS_SIGNING_CERTIFICATE_BASE64
    require_environment OPENHCS_MACOS_SIGNING_CERTIFICATE_PASSWORD
    require_environment OPENHCS_MACOS_DEVELOPER_IDENTITY
    case "$OPENHCS_MACOS_DEVELOPER_IDENTITY" in
        "Developer ID Application: "*)
            ;;
        *)
            printf '%s\n' \
                'The macOS signing identity must be Developer ID Application.' \
                >&2
            exit 2
            ;;
    esac

    temporary_directory=$(
        /usr/bin/mktemp -d "${TMPDIR:-/tmp}/openhcs-signing.XXXXXX"
    )
    local certificate_path="$temporary_directory/developer-id.p12"
    signing_keychain="$temporary_directory/openhcs-signing.keychain-db"
    local keychain_password
    keychain_password=$(/usr/bin/uuidgen)

    if ! printf '%s' "$OPENHCS_MACOS_SIGNING_CERTIFICATE_BASE64" |
        /usr/bin/base64 -D >"$certificate_path"; then
        printf '%s\n' \
            'OPENHCS_MACOS_SIGNING_CERTIFICATE_BASE64 is not valid base64.' >&2
        exit 2
    fi
    /bin/chmod 600 "$certificate_path"

    /usr/bin/security create-keychain \
        -p "$keychain_password" "$signing_keychain"
    /usr/bin/security set-keychain-settings \
        -lut 21600 "$signing_keychain"
    /usr/bin/security unlock-keychain \
        -p "$keychain_password" "$signing_keychain"
    /usr/bin/security import "$certificate_path" \
        -k "$signing_keychain" \
        -P "$OPENHCS_MACOS_SIGNING_CERTIFICATE_PASSWORD" \
        -T /usr/bin/codesign
    /usr/bin/security set-key-partition-list \
        -S apple-tool:,apple: \
        -s \
        -k "$keychain_password" \
        "$signing_keychain" >/dev/null

    if ! /usr/bin/security find-identity \
        -v \
        -p codesigning \
        "$signing_keychain" |
        /usr/bin/grep -F \
            "\"$OPENHCS_MACOS_DEVELOPER_IDENTITY\"" >/dev/null; then
        printf 'Developer ID identity is absent from the supplied certificate.\n' \
            >&2
        exit 2
    fi
}

verify_timestamped_signature() {
    local artifact=$1
    /usr/bin/codesign --verify --deep --strict --verbose=2 "$artifact"

    local signature_details
    signature_details=$(
        /usr/bin/codesign --display --verbose=4 "$artifact" 2>&1
    )
    printf '%s\n' "$signature_details"
    if ! printf '%s\n' "$signature_details" |
        /usr/bin/grep -q '^Timestamp='; then
        printf 'The Developer ID signature has no secure timestamp.\n' >&2
        exit 1
    fi
}

sign_app() {
    local app_path=$1
    if [[ ! -d "$app_path" ]]; then
        printf 'macOS installer app not found: %s\n' "$app_path" >&2
        exit 2
    fi

    create_signing_keychain
    /usr/bin/codesign \
        --force \
        --options runtime \
        --timestamp \
        --keychain "$signing_keychain" \
        --sign "$OPENHCS_MACOS_DEVELOPER_IDENTITY" \
        "$app_path"
    verify_timestamped_signature "$app_path"

    local signature_details
    signature_details=$(
        /usr/bin/codesign --display --verbose=4 "$app_path" 2>&1
    )
    if ! printf '%s\n' "$signature_details" |
        /usr/bin/grep -q 'flags=.*runtime'; then
        printf 'The installer app signature does not enable hardened runtime.\n' \
            >&2
        exit 1
    fi
}

sign_dmg_and_notarize() {
    local dmg_path=$1
    if [[ ! -f "$dmg_path" ]]; then
        printf 'macOS installer disk image not found: %s\n' "$dmg_path" >&2
        exit 2
    fi
    require_environment OPENHCS_MACOS_NOTARY_KEY_BASE64
    require_environment OPENHCS_MACOS_NOTARY_KEY_ID
    require_environment OPENHCS_MACOS_NOTARY_ISSUER_ID

    create_signing_keychain
    /usr/bin/codesign \
        --force \
        --timestamp \
        --keychain "$signing_keychain" \
        --sign "$OPENHCS_MACOS_DEVELOPER_IDENTITY" \
        "$dmg_path"
    verify_timestamped_signature "$dmg_path"
    /usr/bin/hdiutil verify "$dmg_path"

    local notary_key_path="$temporary_directory/AuthKey.p8"
    local notary_result_path="$temporary_directory/notary-result.json"
    local notary_log_path="$temporary_directory/notary-log.json"
    if ! printf '%s' "$OPENHCS_MACOS_NOTARY_KEY_BASE64" |
        /usr/bin/base64 -D >"$notary_key_path"; then
        printf '%s\n' \
            'OPENHCS_MACOS_NOTARY_KEY_BASE64 is not valid base64.' >&2
        exit 2
    fi
    /bin/chmod 600 "$notary_key_path"

    if ! /usr/bin/xcrun notarytool submit "$dmg_path" \
        --key "$notary_key_path" \
        --key-id "$OPENHCS_MACOS_NOTARY_KEY_ID" \
        --issuer "$OPENHCS_MACOS_NOTARY_ISSUER_ID" \
        --wait \
        --output-format json >"$notary_result_path"; then
        /bin/cat "$notary_result_path" >&2
        exit 1
    fi
    /bin/cat "$notary_result_path"

    local notary_status
    notary_status=$(
        /usr/bin/plutil \
            -extract status raw \
            -o - \
            "$notary_result_path"
    )
    if [[ "$notary_status" != Accepted ]]; then
        printf 'Apple notarization was not accepted: %s\n' "$notary_status" >&2
        exit 1
    fi

    local notary_submission_id
    notary_submission_id=$(
        /usr/bin/plutil \
            -extract id raw \
            -o - \
            "$notary_result_path"
    )
    if [[ -z "$notary_submission_id" ]]; then
        printf 'Apple notarization returned no submission identifier.\n' >&2
        exit 1
    fi
    if ! /usr/bin/xcrun notarytool log "$notary_submission_id" \
        --key "$notary_key_path" \
        --key-id "$OPENHCS_MACOS_NOTARY_KEY_ID" \
        --issuer "$OPENHCS_MACOS_NOTARY_ISSUER_ID" \
        "$notary_log_path"; then
        printf 'Could not retrieve the Apple notarization log.\n' >&2
        exit 1
    fi
    /bin/cat "$notary_log_path"

    local notary_log_status
    notary_log_status=$(
        /usr/bin/plutil \
            -extract status raw \
            -o - \
            "$notary_log_path"
    )
    if [[ "$notary_log_status" != Accepted ]]; then
        printf 'Apple notarization log was not accepted: %s\n' \
            "$notary_log_status" >&2
        exit 1
    fi

    /usr/bin/xcrun stapler staple "$dmg_path"
    /usr/bin/xcrun stapler validate "$dmg_path"
    verify_timestamped_signature "$dmg_path"
    /usr/bin/hdiutil verify "$dmg_path"
    /usr/sbin/spctl \
        --assess \
        --type open \
        --context context:primary-signature \
        --verbose=4 \
        "$dmg_path"
}

require_command /usr/bin/base64
require_command /usr/bin/codesign
require_command /usr/bin/hdiutil
require_command /usr/bin/plutil
require_command /usr/bin/security
require_command /usr/bin/xcrun
require_command /usr/sbin/spctl

if [[ $# -ne 2 ]]; then
    usage
fi
mode=$1
artifact_path=$2
case "$mode" in
    sign-app)
        sign_app "$artifact_path"
        ;;
    sign-dmg-and-notarize)
        sign_dmg_and_notarize "$artifact_path"
        ;;
    *)
        usage
        ;;
esac
