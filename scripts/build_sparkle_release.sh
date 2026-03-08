#!/bin/zsh

set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
project_file="$project_root/VoiceInk.xcodeproj/project.pbxproj"

tag_name="${TAG_NAME:?TAG_NAME is required}"
github_repository="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
sparkle_feed_url="${SPARKLE_FEED_URL:?SPARKLE_FEED_URL is required}"
sparkle_private_key="${SPARKLE_PRIVATE_KEY:?SPARKLE_PRIVATE_KEY is required}"
apple_team_id="${APPLE_TEAM_ID:-}"
keychain_path="${KEYCHAIN_PATH:?KEYCHAIN_PATH is required}"

release_version="${tag_name#v}"
current_marketing="$(sed -n 's/.*MARKETING_VERSION = \([0-9.]*\);/\1/p' "$project_file" | head -1)"

if [[ "$current_marketing" != "$release_version" ]]; then
    echo "Tag version $release_version does not match project MARKETING_VERSION $current_marketing" >&2
    exit 1
fi

current_build="$(sed -n 's/.*CURRENT_PROJECT_VERSION = \([0-9]*\);/\1/p' "$project_file" | head -1)"
repo_name="${github_repository##*/}"
build_root="${BUILD_ROOT:-$project_root/.codex-build/sparkle-release}"
derived_data_path="$build_root/deriveddata"
spm_path="$build_root/spm"
archive_path="$build_root/VoiceInk.xcarchive"
dist_path="$build_root/dist"
pages_path="$build_root/pages"
release_entitlements="${VOICEINK_RELEASE_ENTITLEMENTS:-$project_root/VoiceInk/VoiceInk.local.entitlements}"
release_notes_path="$dist_path/VoiceInk ${release_version}.txt"
archive_name="VoiceInk ${release_version}.zip"
archive_output_path="$dist_path/$archive_name"
sparkle_key_account="${SPARKLE_KEY_ACCOUNT:-$repo_name}"
codesign_identity="${APPLE_CODESIGN_IDENTITY:-Developer ID Application}"

rm -rf "$build_root"
mkdir -p "$dist_path" "$pages_path"

xcodebuild \
    -project "$project_root/VoiceInk.xcodeproj" \
    -scheme VoiceInk \
    -resolvePackageDependencies \
    -clonedSourcePackagesDirPath "$spm_path" >/dev/null

sparkle_bin_dir="$(find "$spm_path" -path '*/artifacts/sparkle/Sparkle/bin' -type d | head -1)"
if [[ -z "$sparkle_bin_dir" ]]; then
    echo "Could not locate Sparkle binaries under $spm_path" >&2
    exit 1
fi

sparkle_private_key_file="$build_root/sparkle_private_key.txt"
printf '%s' "$sparkle_private_key" > "$sparkle_private_key_file"

"$sparkle_bin_dir/generate_keys" --account "$sparkle_key_account" -f "$sparkle_private_key_file" >/dev/null
sparkle_public_key="$("$sparkle_bin_dir/generate_keys" --account "$sparkle_key_account" -p | tr -d '\n')"

if [[ -z "$sparkle_public_key" ]]; then
    echo "Could not derive Sparkle public key" >&2
    exit 1
fi

xcodebuild_args=(
    -project "$project_root/VoiceInk.xcodeproj"
    -scheme VoiceInk
    -configuration Release
    -archivePath "$archive_path"
    -derivedDataPath "$derived_data_path"
    -clonedSourcePackagesDirPath "$spm_path"
    CODE_SIGN_STYLE=Manual
    CODE_SIGN_IDENTITY="$codesign_identity"
    CODE_SIGN_ENTITLEMENTS="$release_entitlements"
    OTHER_CODE_SIGN_FLAGS="--keychain $keychain_path"
    SPARKLE_FEED_URL="$sparkle_feed_url"
    SPARKLE_PUBLIC_ED_KEY="$sparkle_public_key"
    'SWIFT_ACTIVE_COMPILATION_CONDITIONS=$(inherited) OPEN_SOURCE_DISTRIBUTION'
    archive
)

if [[ -n "$apple_team_id" ]]; then
    xcodebuild_args+=(DEVELOPMENT_TEAM="$apple_team_id")
fi

xcodebuild "${xcodebuild_args[@]}"

app_path="$archive_path/Products/Applications/VoiceInk.app"
if [[ ! -d "$app_path" ]]; then
    echo "Expected archived app at $app_path" >&2
    exit 1
fi

if [[ -n "${APPLE_NOTARY_KEY_ID:-}" && -n "${APPLE_NOTARY_ISSUER_ID:-}" && -n "${APPLE_NOTARY_API_PRIVATE_KEY_BASE64:-}" ]]; then
    api_key_path="$build_root/AuthKey_${APPLE_NOTARY_KEY_ID}.p8"
    printf '%s' "$APPLE_NOTARY_API_PRIVATE_KEY_BASE64" | base64 --decode > "$api_key_path"

    notarization_zip="$build_root/notarization.zip"
    ditto -c -k --sequesterRsrc --keepParent "$app_path" "$notarization_zip"

    xcrun notarytool submit "$notarization_zip" \
        --wait \
        --key "$api_key_path" \
        --key-id "$APPLE_NOTARY_KEY_ID" \
        --issuer "$APPLE_NOTARY_ISSUER_ID"

    xcrun stapler staple "$app_path"
fi

ditto -c -k --sequesterRsrc --keepParent "$app_path" "$archive_output_path"

if [[ -n "${RELEASE_NOTES_FILE:-}" && -f "${RELEASE_NOTES_FILE}" ]]; then
    cat "$RELEASE_NOTES_FILE" > "$release_notes_path"
elif [[ -n "${RELEASE_NOTES_TEXT:-}" ]]; then
    printf '%s\n' "$RELEASE_NOTES_TEXT" > "$release_notes_path"
else
    cat > "$release_notes_path" <<EOF
VoiceInk $release_version

Build $current_build
EOF
fi

current_appcast_url="${sparkle_feed_url}"
if curl --fail --silent --location "$current_appcast_url" --output "$dist_path/appcast.xml"; then
    :
else
    rm -f "$dist_path/appcast.xml"
fi

"$sparkle_bin_dir/generate_appcast" \
    --ed-key-file "$sparkle_private_key_file" \
    --download-url-prefix "https://github.com/${github_repository}/releases/download/${tag_name}/" \
    --link "https://github.com/${github_repository}" \
    --embed-release-notes \
    "$dist_path"

cp "$dist_path/appcast.xml" "$pages_path/appcast.xml"

cat <<EOF
ARCHIVE_PATH=$archive_output_path
APPCAST_PATH=$pages_path/appcast.xml
DIST_PATH=$dist_path
SPARKLE_PUBLIC_ED_KEY=$sparkle_public_key
EOF
