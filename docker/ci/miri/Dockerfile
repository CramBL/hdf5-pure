ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/CramBL/hdf5-pure" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

ENV RUSTUP_HOME="/opt/rustup" \
    CARGO_HOME="/opt/cargo" \
    MIRI_SYSROOT="/opt/miri-sysroot" \
    PATH="/opt/cargo/bin:/opt/ci-tools/bin:${PATH}"
COPY ci/miri.env scripts/nightly.txt docker/ci/miri/install.sh /tmp/miri-build/
RUN bash /tmp/miri-build/install.sh && rm -rf /tmp/miri-build
