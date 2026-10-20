ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CMAKE_VERSION
ARG CMAKE_URL
ARG CMAKE_SHA256
ARG SOURCE_REVISION
ARG INPUT_FINGERPRINT

LABEL org.opencontainers.image.source="https://github.com/CramBL/hdf5-pure" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.hdf5-pure.inputs="${INPUT_FINGERPRINT}"

COPY docker/ci/cross/verify-cmake.sh /usr/local/libexec/
COPY scripts/ci/install-mold.sh /usr/local/libexec/install-mold.sh

RUN test -n "$CMAKE_VERSION" && test -n "$CMAKE_SHA256" \
    && curl --fail --silent --show-error --location "$CMAKE_URL" --output /tmp/cmake.tar.gz \
    && printf '%s  /tmp/cmake.tar.gz\n' "$CMAKE_SHA256" | sha256sum --check \
    && mkdir -p /opt/cmake \
    && tar --extract --gzip --file /tmp/cmake.tar.gz --strip-components=1 --directory /opt/cmake \
    && rm -rf /opt/cmake/bin/cmake-gui /opt/cmake/bin/ccmake \
        /opt/cmake/doc/cmake/html /opt/cmake/doc/cmake/CMake.qch /opt/cmake/man \
    && rm /tmp/cmake.tar.gz \
    && ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && ln -sf /opt/cmake/bin/ctest /usr/local/bin/ctest \
    && ln -sf /opt/cmake/bin/cpack /usr/local/bin/cpack \
    && bash /usr/local/libexec/verify-cmake.sh "$CMAKE_VERSION" \
    && bash /usr/local/libexec/install-mold.sh /opt/mold \
    && ln -s /opt/mold/bin/mold "/usr/local/bin/${CROSS_TOOLCHAIN_PREFIX}ld.mold" \
    && dpkg-query --show > /usr/local/libexec/packages.txt

ENV PATH="/opt/mold/bin:${PATH}"
