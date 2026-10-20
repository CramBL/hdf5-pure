ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS prepared

ARG CMAKE_VERSION
ARG CMAKE_URL
ARG CMAKE_SHA256
COPY docker/ci/cross/verify-cmake.sh /usr/local/libexec/
COPY scripts/ci/install-mold.sh /usr/local/libexec/install-mold.sh

RUN apt-get purge --assume-yes libclang-dev libclang-18-dev \
        gfortran gfortran-13 gfortran-x86-64-linux-gnu gfortran-13-x86-64-linux-gnu \
        "gfortran-${CROSS_TOOLCHAIN_PREFIX%-}" "gfortran-13-${CROSS_TOOLCHAIN_PREFIX%-}" \
    && rm -rf /usr/local/bin/cmake /usr/local/bin/cmake-gui /usr/local/bin/ccmake \
        /usr/local/bin/ctest /usr/local/bin/cpack /usr/local/share/cmake-3.23 \
        /usr/local/doc/cmake /usr/local/man/man1/cmake* /usr/local/man/man1/ccmake* \
        /usr/local/man/man1/ctest* /usr/local/man/man1/cpack* /usr/local/man/man7/cmake* \
    && test -n "$CMAKE_VERSION" && test -n "$CMAKE_SHA256" \
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

FROM scratch
COPY --from=prepared / /

ARG SOURCE_REVISION

LABEL org.opencontainers.image.source="https://github.com/CramBL/hdf5-pure" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

ENV PATH="/opt/mold/bin:${PATH}"
