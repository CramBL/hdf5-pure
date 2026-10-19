#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp --parents ci/images.toml scripts/ci/*.sh docker/ci/cross/* "$work"
cp --parents ci/matio.env rust-toolchain.toml docker/ci/matio/* "$work"
cd "$work"
i686=$(bash scripts/ci/cross-image.sh fingerprint i686)
s390x=$(bash scripts/ci/cross-image.sh fingerprint s390x)
matio=$(bash scripts/ci/matio-image.sh fingerprint)

printf '[published]\ndigest = "sha256:record"\n' >> ci/images.toml
printf 'image = "sha256:consumed"\n' > Cross.toml
printf '\n' >> scripts/ci/select-cross-image.sh
printf '\n' >> scripts/ci/run-matio.sh
test "$i686" = "$(bash scripts/ci/cross-image.sh fingerprint i686)"
test "$s390x" = "$(bash scripts/ci/cross-image.sh fingerprint s390x)"
test "$matio" = "$(bash scripts/ci/matio-image.sh fingerprint)"

printf '\n' >> docker/ci/matio/linux-64.lock
test "$matio" != "$(bash scripts/ci/matio-image.sh fingerprint)"
test "$s390x" = "$(bash scripts/ci/cross-image.sh fingerprint s390x)"

sed -i '/^\[images.i686.build\]$/,/^\[/s/parent = "/parent = "changed-/' ci/images.toml
test "$i686" != "$(bash scripts/ci/cross-image.sh fingerprint i686)"
test "$s390x" = "$(bash scripts/ci/cross-image.sh fingerprint s390x)"

printf '\n' >> scripts/ci/install-mold.sh
test "$s390x" != "$(bash scripts/ci/cross-image.sh fingerprint s390x)"
