#!/usr/bin/env bash
set -euo pipefail
cd /tmp/miri-build
source miri.env
nightly=$(cat nightly.txt)
apt-get update
apt-get install --assume-yes --no-install-recommends ca-certificates curl gcc libc6-dev git
rm -rf /var/lib/apt/lists/*
curl -fsSL "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/x86_64-unknown-linux-gnu/rustup-init" -o rustup-init
printf '%s  rustup-init\n' "$RUSTUP_SHA256" | sha256sum --check
chmod +x rustup-init
./rustup-init --yes --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install --no-self-update --profile minimal "$nightly" --component miri,rust-src
rustup default "$nightly"
rustup component list --installed > /opt/rust-components.txt
curl -fsSL "https://github.com/casey/just/releases/download/$JUST_VERSION/just-$JUST_VERSION-x86_64-unknown-linux-musl.tar.gz" -o just.tar.gz
printf '%s  just.tar.gz\n' "$JUST_SHA256" | sha256sum --check
mkdir -p /opt/ci-tools/bin
tar -xzf just.tar.gz -C /opt/ci-tools/bin just
cargo +"$nightly" miri setup
rm -rf "$CARGO_HOME/registry" "$CARGO_HOME/git"
rustc --version --verbose
cargo --version
just --version
dpkg-query --show > /opt/system-packages.txt
