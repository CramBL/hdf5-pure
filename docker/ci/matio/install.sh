#!/usr/bin/env bash
set -euo pipefail
cd /tmp/matio-build
source matio.env
apt-get update
xargs apt-get install --assume-yes --no-install-recommends < packages.txt
rm -rf /var/lib/apt/lists/*
rustup toolchain install --profile minimal
rustup component list --installed > /opt/rust-components.txt
mkdir -p /opt/ci-tools/bin
curl -fsSL "https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64" -o micromamba
printf '%s  micromamba\n' "$MICROMAMBA_SHA256" | sha256sum --check
chmod +x micromamba
export MAMBA_ROOT_PREFIX=/tmp/matio-packages
./micromamba create --yes --prefix /opt/matio --file /opt/matio-linux-64.lock
mkdir -p /opt/matio-licenses
while IFS= read -r -d '' licenses; do
    package=${licenses%/info/licenses}
    cp -r "$licenses" "/opt/matio-licenses/${package##*/}"
done < <(find "$MAMBA_ROOT_PREFIX/pkgs" -type d -path '*/info/licenses' -print0)
./micromamba clean --all --yes
for tool in just nextest; do
    if [[ $tool == just ]]; then
        url="https://github.com/casey/just/releases/download/$JUST_VERSION/just-$JUST_VERSION-x86_64-unknown-linux-musl.tar.gz"
        checksum=$JUST_SHA256
        binary=just
    else
        url="https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-$NEXTEST_VERSION/cargo-nextest-$NEXTEST_VERSION-x86_64-unknown-linux-gnu.tar.gz"
        checksum=$NEXTEST_SHA256
        binary=cargo-nextest
    fi
    curl -fsSL "$url" -o tool.tar.gz
    printf '%s  tool.tar.gz\n' "$checksum" | sha256sum --check
    tar -xzf tool.tar.gz -C /opt/ci-tools/bin "$binary"
done
bash install-mold.sh /opt/mold
dpkg-query --show > /opt/system-packages.txt
