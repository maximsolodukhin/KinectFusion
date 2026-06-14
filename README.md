# KinectFusion

The repository ships a single multi-stage `Dockerfile` that produces both the
development image and the production image. All stages are based on
`ubuntu:24.04`.

| Stage               | Purpose                                                                                                                     | Image name         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `dev`               | Toolchain + dev-only tools (gdb, clang-tidy, cppcheck, ccache). No source baked in; you bind-mount the project at run time. | `kinectfusion-dev` |
| `runtime` (default) | Minimal Ubuntu with the compiled `kinectfusion` binary and its runtime libs.                                                | `kinectfusion`     |

Building `runtime` transparently runs an unnamed intermediate stage that
performs two passes over the source: (1) a Debug build with `clang-tidy`,
`cppcheck`, ASan and UBSan all ON plus the `ctest` suite — any failure aborts
the whole `docker build` — and (2) a Release build of the `intro` binary.

## Dev Container

The intended flow is: bring the dev container up in the background with the
project bind-mounted, then have your editor attach to the running container
(named `kinectfusion-dev`).

### Build and start

```bash
docker compose up -d dev
```

This builds the `dev` stage on first run, starts a container called
`kinectfusion-dev`, bind-mounts the repository at `/workspace`, and keeps it
alive with `sleep infinity` so the editor has something to attach to. After
attaching to the container with **Dev Containers: Attach to Running Container**
in the IDE:

```bash
cd /workspace
cmake -S . -B build
cmake --build build -j
```

`build/` lives in the bind-mounted workspace, so it's visible from the host
filesystem too.

### Stop / clean up

```bash
docker compose down
```

## Build (host toolchain)

```bash
git clone git@github.com:maximsolodukhin/KinectFusion.git
cd KinectFusion
mkdir build
cmake -S . -B build
cmake --build build -j 5
```

## Production image

```bash
# Build only (runs the full validation pass; aborts on any failure).
docker compose build kinectfusion

# Run as a one-shot CLI. Compose creates an ephemeral container per call.
docker compose run --rm kinectfusion --version
docker compose run --rm kinectfusion --help
```

To pass input data (e.g. RGB-D sequences) into the container, add a volume flag
to `compose run`:

```bash
docker compose run --rm \
    -v "$PWD/data:/data:ro" \
    kinectfusion <args> /data/<input>
```
