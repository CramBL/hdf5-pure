ARG RUST_VERSION
FROM rust:${RUST_VERSION}-slim-trixie

ARG RUST_VERSION
ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/CramBL/hdf5-pure" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

COPY ci/matio.env rust-toolchain.toml docker/ci/matio/packages.txt /tmp/matio-build/
COPY docker/ci/matio/linux-64.lock /opt/matio-linux-64.lock
COPY scripts/ci/install-mold.sh docker/ci/matio/install.sh /tmp/matio-build/
RUN bash /tmp/matio-build/install.sh && rm -rf /tmp/matio-build

ENV RUSTUP_TOOLCHAIN="${RUST_VERSION}" \
    PATH="/opt/mold/bin:/opt/ci-tools/bin:${PATH}" \
    LIBRARY_PATH="/opt/matio/lib" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="cc" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C force-frame-pointers=yes -C link-arg=-fuse-ld=mold -C link-arg=-Wl,-rpath,/opt/matio/lib"
