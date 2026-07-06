# syntax=docker/dockerfile:1.7
#
# Stages:
#   dev       - Toolchain + dev-only tools (gdb, clang-tidy, clang-format,
#               cppcheck, ccache). No project source. Bind-mount the project
#               at run time.
#               Build with: `docker build --target dev -t kinectfusion-dev .`
#   runtime   - Minimal Ubuntu with the stripped release `kinectfusion` binary and
#               its runtime libraries.
#               Build with: `docker build -t kinectfusion .`
FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 24.04's default GCC is 13, which is what we build with. We do pull in
# Clang 21 (clangd, clang-tidy, clang-format) from apt.llvm.org so IntelliSense
# and the tidy pass agree with libstdc++ on C++23 concept-gated headers like
# <expected> (libstdc++'s <expected> is gated on __cpp_concepts >= 202002L,
# which Clang < 19 doesn't advertise).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget \
        git \
        build-essential \
        gdb \
        cppcheck \
        ccache \
        zlib1g-dev \
        libasan8 \
        libubsan1; \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main" \
        > /etc/apt/sources.list.d/kitware.list; \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
        > /etc/apt/sources.list.d/llvm.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cmake \
        clangd-21 \
        clang-tidy-21 \
        clang-format-21; \
    update-alternatives \
        --install /usr/bin/clangd       clangd       /usr/bin/clangd-21       210 \
        --slave   /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-21 \
        --slave   /usr/bin/clang-format clang-format /usr/bin/clang-format-21; \
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
RUN cmake --preset ci-checks \
    && cmake --build --preset ci-checks --parallel "$(nproc)" \
    && ctest --preset ci-checks

# KINECTFUSION_ENABLE_HARDENING=ON is set by the release preset, which bakes
# the following into the shipped binary (via cmake/Hardening.cmake):
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
RUN cmake --preset release \
    && cmake --build --preset release --parallel "$(nproc)" \
    && strip --strip-unneeded build/release/src/app/kinectfusion

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

COPY --from=1 /workspace/build/release/src/app/kinectfusion /usr/local/bin/kinectfusion

USER app
WORKDIR /home/app

ENTRYPOINT ["/usr/local/bin/kinectfusion"]
