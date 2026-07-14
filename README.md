# KinectFusion

## Usage

### 1. Build

```bash
cmake --preset release
cmake --build --preset release --parallel
```

The executable is `build/release/src/app/kinectfusion`.

### 2. Get a dataset

A TUM RGB-D sequence (the `freiburg1_xyz` desk scene) is expected. The directory
must contain `depth.txt`, `rgb.txt`, and the `depth/` + `rgb/` folders. Download
from <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>.

This repo expects it in the `data/` folder at the project root, e.g.
`data/rgbd_dataset_freiburg1_xyz/`.

### 3. Run

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 5 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

This writes one raycast render per frame to `outputs/`:

- `frame_XXXXXX_raycast.png` — color raycast of the fused TSDF surface
- `frame_XXXXXX_raycast.ply` — surface point cloud (position + normal + color)

Single frame only (just integrate + raycast, no tracking):

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 1 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

#### Options

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

`./build/release/src/app/kinectfusion --help` lists everything.

#### Notes

- **Use a small `--volume-camera-margin`** (e.g. `0.5`). The default `2.56`
  equals the full volume extent, which places the volume _behind_ the camera
  (which looks down +z) and reconstructs nothing.
- **Performance**: the integrator sweeps the whole grid every frame on a single
  CPU thread, so `--volume-resolution 256` is ~20–40 s/frame. Use `128` for
  quick iteration, higher for more detail.
- The volume is fixed in world space at the first frame's pose; keep sequences
  short enough that the camera stays inside it.

### Ablation pipelines

`--pipelines <file.toml>` runs several independently configured TSDF pipelines
on the same input frames and compares each against a reference pipeline:

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 5 \
  --volume-resolution 128 --voxel-size 0.02 --volume-camera-margin 0.5 \
  --pipelines configs/ablation_example.toml --output-dir outputs
```

Per-frame deviation statistics against the reference are logged and appended to
`<output-dir>/ablation_stats.csv`; each pipeline writes its raycasts to
`<output-dir>/<pipeline-name>/`.

The TOML file has two optional top-level keys and one `[[pipeline]]` table per
pipeline:

```toml
reference = "baseline"     # defaults to the first pipeline
compare-every-n-frames = 1 # defaults to --compare-every; <= 0 disables

[[pipeline]]
name = "baseline"
tsdf-variant = "angle-weighted"

[[pipeline]]
name = "classic"
tsdf-variant = "classic"
```

Every pipeline starts from the CLI-derived configuration and overrides only
what it ablates:

| Key                          | Type   | Meaning                                            |
| ---------------------------- | ------ | -------------------------------------------------- |
| `name`                       | string | Required, unique; names the output subdirectory    |
| `space`                      | string | `"cpu"` or `"cuda"` (falls back to cpu with a warning) |
| `tsdf-variant`               | string | `"angle-weighted"` or `"classic"`                  |
| `projective-distance`        | bool   | Projective instead of euclidean surface distance   |
| `distance-scaled-truncation` | bool   | Scale truncation with measured depth               |
| `truncation-distance-scale`  | float  | Factor for the scaled truncation                   |
| `observation-weight`         | float  | Weight of one observation                          |
| `max-weight`                 | float  | Running-average weight cap                         |

The volume geometry is deliberately not overridable — it stays global (CLI
flags) so cross-pipeline comparison is defined on identical grids. Unknown keys
throw `std::invalid_argument`: ablation configs fail loud rather than silently
running a default.

## Usage with Docker

The repo's `compose.yaml` defines a `kinectfusion` service that builds the
production image and runs the binary as a one-shot CLI (mirroring the `dev`
service used for the [dev container](#development-container)).

### 1. Build the image

```bash
docker compose build kinectfusion
```

The intermediate stage compiles and runs the test suite (with all warnings,
clang-tidy, cppcheck, and the sanitizers enabled), so any failure aborts the
build. The final image's entrypoint is the binary.

### 2. Run

`docker compose run --rm kinectfusion` starts an ephemeral container and passes
everything after the service name straight to the binary (same flags as the
native build):

```bash
docker compose run --rm kinectfusion --version
docker compose run --rm kinectfusion --help
```

Mount the dataset read-only and an output directory you can write to:

```bash
docker compose run --rm \
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
- See `docker compose run --rm kinectfusion --help`, or the [Options](#options)
  table above, for the full flag list.

## Development

### Development Container

Bring the dev container up in the background with the project bind-mounted, then
attach to the running container (named `kinectfusion-dev`) from the editor, and
work there.

```bash
docker compose up -d dev
```

This builds the `dev` stage on first run, starts a container called
`kinectfusion-dev`, bind-mounts the repository at `/workspace`, runs
`scripts/fetch_tum_freiburg1_xyz.sh`, and keeps it alive with `sleep infinity`
so the editor has something to attach to. The script downloads the TUM
`freiburg1_xyz` dataset into `data/` on first start if necessary.

The project source is bind-mounted into the container at `/workspace`. The build
directory `build/` lives in the bind-mounted workspace, so it's visible from the
host filesystem too.

To work with the project, navigate to the `/workspace` directory, and:

- install the editor extensions you need
- work on the code
- build and run

### Conda Environment

As an alternative to the development container, you can use a Conda environment.

`environment.yml` provides a development environment with a GCC 15 toolchain,
CMake, debugger, analyzers, cache, and LLVM 22 tools. Project libraries continue
to be pinned and downloaded by CPM during CMake configuration.

Create and activate the environment:

```bash
conda env create --file environment.yml
conda activate kinectfusion-dev
```

Download the development dataset and run the build:

```bash
./scripts/fetch_tum_freiburg1_xyz.sh
cmake --preset release
cmake --build --preset release --parallel
```

When `environment.yml` changes, update an existing environment with:

```bash
conda env update --file environment.yml --prune
```

### Build and run (regardless of the development container)

#### CMake Presets

| Preset        | Purpose                                                                                 |
| ------------- | --------------------------------------------------------------------------------------- |
| `dev-strict`  | Debug build, tests, warnings as errors, clang-tidy, cppcheck, ASan/UBSan                |
| `dev-relaxed` | Debug build, tests, warnings allowed, static analysis off, ASan/UBSan                   |
| `ci-checks`   | Docker validation build, equivalent to the strict checked path                          |
| `release`     | Optimized production binary with tests, analyzers, sanitizers, and progress logging off |

Example:

```bash
cmake --preset release
cmake --build --preset release --parallel
```

or a one-liner without parallelization:

```bash
cmake --workflow --preset release
```

**Note:** for heavy builds, extensive parallelization may exhaust the memory
available to the development container or the workstation (you will notice it by
connection hanging). You can specify the number of parallel jobs with
`--parallel 2`.

### Formatting

If the extensions under `.vscode/extensions.json` are installed, the formatting
will applied on save.

To format all files of the project, run:

```bash
find include src test -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cc' -o -name '*.cxx' -o -name '*.c' \) -print0   | xargs -0 -P "$(nproc)" -n 20 clang-format -i --style=file
```
