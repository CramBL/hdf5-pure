ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SOURCE_REVISION
ARG INPUT_FINGERPRINT
LABEL org.opencontainers.image.source="https://github.com/CramBL/hdf5-pure" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.hdf5-pure.inputs="${INPUT_FINGERPRINT}"

COPY ci/matio.env rust-toolchain.toml docker/ci/matio/packages.txt /tmp/matio-build/
COPY docker/ci/matio/linux-64.lock /opt/matio-linux-64.lock
COPY scripts/ci/install-mold.sh docker/ci/matio/install.sh /tmp/matio-build/
COPY docker/ci/matio/verify.sh /usr/local/libexec/verify-matio.sh
RUN bash /tmp/matio-build/install.sh && rm -rf /tmp/matio-build

ENV PATH="/opt/mold/bin:/opt/ci-tools/bin:${PATH}" \
    LIBRARY_PATH="/opt/matio/lib" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="cc" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C force-frame-pointers=yes -C link-arg=-fuse-ld=mold -C link-arg=-Wl,-rpath,/opt/matio/lib"
RUN bash /usr/local/libexec/verify-matio.sh
