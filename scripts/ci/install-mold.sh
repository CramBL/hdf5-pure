#!/usr/bin/env bash
set -euo pipefail

version=3.0.0
sha256=6c90d4a474c7c0409dfb575be03a5345878ac14fdba18de8b40fa58c60121189

if [[ ${1:-} == --identity && $# == 1 ]]; then
    printf 'mold-%s-%s\n' "$version" "$sha256"
    exit 0
fi
if [[ $# != 1 ]]; then
    echo 'usage: install-mold.sh PREFIX | --identity' >&2
    exit 1
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'mold installation requires a Linux AMD64 host' >&2
    exit 1
fi

prefix=$1
archive="mold-$version-x86_64-linux.tar.gz"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl --fail --silent --show-error --location \
    "https://github.com/rui314/mold/releases/download/v$version/$archive" \
    --output "$work/$archive"
printf '%s  %s\n' "$sha256" "$work/$archive" | sha256sum --check
mkdir -p "$prefix"
tar --extract --gzip --file "$work/$archive" --strip-components=1 --directory "$prefix"
"$prefix/bin/mold" --version | grep -F "mold $version "
"$prefix/bin/ld.mold" --version | grep -F "mold $version "
