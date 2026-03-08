#!/bin/zsh

set -euo pipefail

project_root="${1:-$(pwd)}"
project_file="$project_root/VoiceInk.xcodeproj/project.pbxproj"

if [[ ! -f "$project_file" ]]; then
    echo "Could not find project file at $project_file" >&2
    exit 1
fi

current_marketing="$(
    sed -n 's/.*MARKETING_VERSION = \([0-9.]*\);/\1/p' "$project_file" | head -1
)"
current_build="$(
    sed -n 's/.*CURRENT_PROJECT_VERSION = \([0-9]*\);/\1/p' "$project_file" | head -1
)"

if [[ -z "$current_marketing" || -z "$current_build" ]]; then
    echo "Could not read current marketing/build versions from $project_file" >&2
    exit 1
fi

IFS='.' read -r major minor patch <<< "$current_marketing"
patch="${patch:-0}"

next_patch=$((patch + 1))
next_marketing="${major}.${minor}.${next_patch}"
next_build=$((current_build + 1))

NEXT_MARKETING="$next_marketing" NEXT_BUILD="$next_build" perl -0pi -e '
    my $marketing = $ENV{NEXT_MARKETING};
    my $build = $ENV{NEXT_BUILD};
    my $marketing_seen = 0;
    s/MARKETING_VERSION = [0-9.]+;/++$marketing_seen <= 2 ? "MARKETING_VERSION = $marketing;" : $&/ge;
    s/CURRENT_PROJECT_VERSION = [0-9]+;/CURRENT_PROJECT_VERSION = $build;/g;
' "$project_file"

printf 'Version bumped: %s (%s) -> %s (%s)\n' "$current_marketing" "$current_build" "$next_marketing" "$next_build"
