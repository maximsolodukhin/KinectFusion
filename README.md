# KinectFusion

The repository ships a single multi-stage `Dockerfile` that produces both the
development image and the production image. All stages are based on
`ubuntu:24.04`.

| Stage               | Purpose                                                                                                                     | Image name         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `dev`               | Toolchain + dev-only tools (gdb, clang-tidy, cppcheck, ccache). No source baked in; you bind-mount the project at run time. | `kinectfusion-dev` |
| `runtime` (default) | Minimal Ubuntu with the compiled `kinectfusion` binary and its runtime libs.                                                | `kinectfusion`     |

Build modes are centralized in `CMakePresets.json`. Building `runtime`
transparently runs an unnamed intermediate stage that performs two preset-driven
passes over the source: (1) `ci-checks`, a Debug build with `clang-tidy`,
`cppcheck`, ASan and UBSan all ON plus the `ctest` suite — any failure aborts
the whole `docker build` — and (2) `release`, a Release build of the
`kinectfusion` binary.

## Dev Container

The intended flow is: bring the dev container up in the background with the
project bind-mounted, then have your editor attach to the running container
(named `kinectfusion-dev`).

### Build and start

```bash
docker compose up -d dev
```

This builds the `dev` stage on first run, starts a container called
`kinectfusion-dev`, bind-mounts the repository at `/workspace`, runs
`scripts/fetch_tum_freiburg1_xyz.sh`, and keeps it alive with `sleep infinity`
so the editor has something to attach to. The script downloads the TUM
`freiburg1_xyz` dataset into `data/` on first start and exits immediately on
later starts when the dataset is already present. First startup requires network
access unless the dataset has already been fetched. After attaching to the
container with **Dev Containers: Attach to Running Container** in the IDE:

```bash
cd /workspace
cmake --workflow --preset dev-strict
```

`build/` lives in the bind-mounted workspace, so it's visible from the host
filesystem too.

### Build presets

The common build modes are available as CMake presets:

| Preset        | Purpose                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `dev-strict`  | Debug build, tests, warnings as errors, clang-tidy, cppcheck, ASan/UBSan |
| `dev-relaxed` | Debug build, tests, warnings allowed, static analysis off, ASan/UBSan    |
| `ci-checks`   | Docker validation build, equivalent to the strict checked path           |
| `release`     | Optimized production binary with tests, analyzers, sanitizers, and progress logging off |

Inside the dev container:

```bash
# Full checked build + tests.
cmake --workflow --preset dev-strict

# Faster local build that still runs tests but does not fail on warnings.
cmake --workflow --preset dev-relaxed

# Optimized binary.
cmake --preset release
cmake --build --preset release --parallel
```

### Stop / clean up

```bash
docker compose down
```

## Build (host toolchain)

```bash
git clone git@github.com:maximsolodukhin/KinectFusion.git
cd KinectFusion
cmake --workflow --preset dev-strict
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

# Running KinectFusion

## 1. Build

```bash
cmake --preset dev-relaxed
cmake --build --preset dev-relaxed --parallel
```

The executable is `build/dev-relaxed/src/app/kinectfusion`. The release preset
writes the optimized executable to `build/release/src/app/kinectfusion`.

## 2. Get a dataset

A TUM RGB-D sequence (the `freiburg1_xyz` desk scene) is expected. The directory
must contain `depth.txt`, `rgb.txt`, and the `depth/` + `rgb/` folders. Download
from <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>.

This repo expects it in the `data/` folder at the project root, e.g.
`data/rgbd_dataset_freiburg1_xyz/`.

## 3. Run

```bash
./build/dev-relaxed/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 5 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

This writes one raycast render per frame to `outputs/`:

- `frame_XXXXXX_raycast.png` — color raycast of the fused TSDF surface
- `frame_XXXXXX_raycast.ply` — surface point cloud (position + normal + color)

Single frame only (just integrate + raycast, no tracking):

```bash
./build/dev-relaxed/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 1 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

## Options

| Flag                        | Default                              | Meaning                                   |
| --------------------------- | ------------------------------------ | ----------------------------------------- |
| `dataset` (positional)      | `../data/rgbd_dataset_freiburg1_xyz` | TUM RGB-D dataset directory               |
| `--frames`                  | `30`                                 | Frames to process (`-1` for all)          |
| `--volume-resolution`       | `512`                                | Cubic TSDF resolution in voxels           |
| `--voxel-size`              | `0.01`                               | Voxel size in meters                      |
| `--truncation-distance`     | `0.05`                               | TSDF truncation distance in meters        |
| `--volume-camera-margin`    | `2.56`                               | Space behind the initial camera in meters |
| `--output-dir`              | `kinectfusion_output`                | Output directory                          |
| `--no-write-raycast-images` | —                                    | Skip PNG output                           |
| `--no-write-point-clouds`   | —                                    | Skip PLY output                           |

`./build/dev-relaxed/src/app/kinectfusion --help` lists everything.

## Notes

- **Use a small `--volume-camera-margin`** (e.g. `0.5`). The default `2.56`
  equals the full volume extent, which places the volume _behind_ the camera
  (which looks down +z) and reconstructs nothing.
- **Performance**: the integrator sweeps the whole grid every frame on a single
  CPU thread, so `--volume-resolution 256` is ~20–40 s/frame. Use `128` for
  quick iteration, higher for more detail.
- The volume is fixed in world space at the first frame's pose; keep sequences
  short enough that the camera stays inside it.

# Running with Docker

The `Dockerfile` (in the repository root, on the `main` branch) builds the app
in a multi-stage build and ships a minimal runtime image containing only the
stripped `kinectfusion` binary and its runtime libraries. Run all `docker`
commands from the repository root so the build context contains the
`Dockerfile`.

## 1. Build the image

```bash
docker build -t kinectfusion .
```

The intermediate stage compiles and runs the test suite (with all warnings,
clang-tidy, cppcheck, and the sanitizers enabled), so any failure aborts the
build. The final image's entrypoint is the binary, exposed as `kinectfusion`.

## 2. Run

The binary is the image entrypoint, so anything after the image name is passed
straight to it (same flags as the native build). Mount the dataset read-only and
an output directory you can write to:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /abs/path/to/rgbd_dataset_freiburg1_xyz:/data:ro \
  -v "$PWD/outputs:/out" \
  kinectfusion /data --frames 5 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir /out
```

Renders land in `./outputs` on the host. Notes:

- `--user "$(id -u):$(id -g)"` makes the container write `/out` as your user.
  Without it the container runs as uid `10001` and the mounted directory must be
  writable by that uid.
- The dataset path passed to the binary (`/data`) is the path **inside** the
  container, i.e. the mount target — not the host path.
- See `kinectfusion --help`, or the [Options](#options) table above, for the
  full flag list.

```bash
docker run --rm kinectfusion --help
```

## 3. Development image (optional)

A `dev` target provides the full toolchain (gdb, clang-tidy, clang-format,
cppcheck, ccache, cmake) without any project source — bind-mount the working
tree at run time:

```bash
docker build --target dev -t kinectfusion-dev .
docker run --rm -it -v "$PWD:/workspace" -w /workspace kinectfusion-dev bash
# then, inside the container:
cmake --workflow --preset dev-strict
```
