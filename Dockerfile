# syntax=docker/dockerfile:1.7
#
# Stages:
#   dev       - Toolchain + dev-only tools (gdb, clang-tidy, clang-format,
#               cppcheck, ccache). No project source. Bind-mount the project
#               at run time.
#               Build with: `docker build --target dev -t kinectfusion-dev .`
#   runtime   - Minimal Ubuntu with the stripped release `intro` binary and
#               its runtime libraries.
#               Build with: `docker build -t kinectfusion .`
FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget \
        git \
        build-essential \
        gdb \
        clang-tidy \
        clang-format \
        cppcheck \
        ccache \
        libasan8 \
        libubsan1; \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main" \
        > /etc/apt/sources.list.d/kitware.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Unnamed, hence not a `--target` candidate; referenced from `runtime` as `COPY --from=1`.
FROM dev

COPY . /workspace

# Test image: static and dynamic analysis.
# Compiles and runs the test suite.
#
# Top-level project defaults cover:
#   Static, compile-time:
#     - WARNINGS_AS_ERRORS  every g++ warning becomes a build error
#     - CLANG_TIDY          clang-tidy runs on every TU per .clang-tidy
#     - CPPCHECK            cppcheck runs on every TU
#   Dynamic, runtime:
#     - SANITIZER_ADDRESS   AddressSanitizer
#     - SANITIZER_UNDEFINED UndefinedBehaviorSanitizer
# 
# Any failure aborts `docker build`, so no image is produced.
RUN cmake -S . -B build-debug \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_TESTING=ON \
    && cmake --build build-debug -j"$(nproc)" \
    && ctest --test-dir build-debug --output-on-failure --no-tests=error

# KINECTFUSION_ENABLE_HARDENING=ON left at default, which via bakes the following
# into the shipped binary (via cmake/Hardening.cmake):
#   - UBSan minimal runtime    (-fsanitize=undefined -fsanitize-minimal-runtime)
#                              same UB checks as full UBSan, aborts via
#                              __builtin_trap instead of libubsan.
#   - _GLIBCXX_ASSERTIONS      bounds checks std::vector::operator[],
#                              std::string, std::span, libstdc++ iterators.
#   - _FORTIFY_SOURCE=3        runtime overflow checks on memcpy/strcpy/
#                              snprintf/... (Release-only, needs -O > 0).
#   - -fstack-protector-strong stack canaries.
#   - -fcf-protection          Intel CET indirect-branch tracking.
#   - -fstack-clash-protection page-by-page stack growth guard.
#   - WARNINGS_AS_ERRORS       treat compile warnings as errors
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DKINECTFUSION_ENABLE_CLANG_TIDY=OFF \
        -DKINECTFUSION_ENABLE_CPPCHECK=OFF \
        -DKINECTFUSION_ENABLE_SANITIZER_ADDRESS=OFF \
        -DKINECTFUSION_ENABLE_SANITIZER_UNDEFINED=OFF \
    && cmake --build build --target intro -j"$(nproc)" \
    && strip --strip-unneeded build/src/app/intro

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

COPY --from=1 /workspace/build/src/app/intro /usr/local/bin/kinectfusion

USER app
WORKDIR /home/app

ENTRYPOINT ["/usr/local/bin/kinectfusion"]
