# KinectFusion

## Quick Start

```bash
conda env create --file environment.yml
conda activate kinectfusion-dev
./scripts/fetch_tum_freiburg1_xyz.sh
cmake --preset release
cmake --build --preset release --parallel
./build/release/src/app/kinectfusion --help
```

## Setup

Use Conda to setup an environment for build and development: GCC 15 toolchain,
CMake, debugger, analyzers, cache, and LLVM 22 tools. Project libraries continue
to be pinned and downloaded by CPM during CMake configuration.

Create and activate the environment:

```bash
conda env create --file environment.yml
conda activate kinectfusion-dev
```

When `environment.yml` changes, update an existing environment with:

```bash
conda env update --file environment.yml --prune
```

## Dataset

A TUM RGB-D sequence (the `freiburg1_xyz` desk scene) is expected. The directory
must contain `depth.txt`, `rgb.txt`, and the `depth/` + `rgb/` folders. Download
from <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>.

This repo expects it in the `data/` folder at the project root, e.g.
`data/rgbd_dataset_freiburg1_xyz/`.

Use the script to download the dataset into the `data/` folder:

```bash
./scripts/fetch_tum_freiburg1_xyz.sh
```

## Build

```bash
cmake --preset release
cmake --build --preset release --parallel
```

The executable is `build/release/src/app/kinectfusion`.

## Run

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 5 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

This writes one raycast render per frame to `outputs/`, plus one mesh at the end
of the run:

- `frame_XXXXXX_raycast.png` — color raycast of the fused TSDF surface
- `frame_XXXXXX_raycast.ply` — surface point cloud (position + normal + color)
- `mesh.ply` — triangle mesh of the whole fused TSDF surface (marching cubes;
  welded vertices with normals and colors). `--mesh-min-weight` removes
  low-confidence cells: a higher value needs more agreeing observations and
  gives a cleaner mesh. Sparse volumes mesh block by block, so large resolutions
  (for example 2048) export without a dense copy.

Single frame only (just integrate + raycast, no tracking):

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 1 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

### Options

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
| `--no-write-mesh`           | —                                    | Skip the final mesh.ply                   |
| `--mesh-min-weight`         | `2`                                  | Minimum TSDF weight for meshed cells      |

`./build/release/src/app/kinectfusion --help` lists everything.

### Notes

- **Use a small `--volume-camera-margin`** (e.g. `0.5`). The default `2.56`
  equals the full volume extent, which places the volume _behind_ the camera
  (which looks down +z) and reconstructs nothing.
- **Use `--space cuda`** when a CUDA GPU is present. The CPU path is many times
  slower.
- **Performance (CPU)**: the integrator sweeps the whole grid every frame on a
  single CPU thread, so `--volume-resolution 256` is ~20–40 s/frame. Use `128`
  for quick iteration, higher for more detail.
- The volume is fixed in world space at the first frame's pose; keep sequences
  short enough that the camera stays inside it.

## Ablations

The program has one baseline algorithm and a set of ablation switches. One
switch changes one part of the algorithm. You run the program with a switch and
you compare the quality and the speed with the baseline.

Each switch is a CLI flag. The first value in the table is the default.

| Flag                                | Values                                | What it does                                                                                                                              |
| ----------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `--space`                           | `cpu`, `cuda`                         | Select the processor that runs the pipeline. `cuda` needs a GPU.                                                                          |
| `--tsdf-variant`                    | `angle-weighted`, `classic`           | TSDF update rule. `angle-weighted` weights each observation by cos(theta)/depth. `classic` uses a constant weight.                        |
| `--voxel`                           | `float`, `quantized`, `bf16`          | Storage of one TSDF voxel. `float` uses 8 bytes. `quantized` (int16) and `bf16` use 4 bytes.                                              |
| `--color`                           | `float`, `none`                       | Volume color storage. `none` stores no color and renders shaded geometry.                                                                 |
| `--storage`                         | `dense`, `sparse`                     | Volume layout. `dense` stores all voxels. `sparse` allocates 8x8x8 voxel blocks only near the surface.                                    |
| `--integration`                     | `full`, `band`                        | Voxels that one frame updates. `full` updates all voxels and carves free space. `band` updates only blocks near the surface; it is lossy. |
| `--raycast`                         | `march`, `bitmap-march`, `band-march` | Empty-space skip for the ray march. `bitmap-march` gives output identical to `march`. `band-march` skips more; output is approximate.     |
| `--projective-tsdf-distance`        | on (`--no-projective-tsdf-distance`)  | Signed-distance measure. On: distance along the pixel ray. Off: camera z distance.                                                        |
| `--distance-scaled-truncation`      | off                                   | Widen the truncation band linearly with measured depth. Set the rate with `--truncation-distance-scale`.                                  |
| `--cell-normals`                    | off                                   | Compute raycast normals from the 8 corners of the final sample. The default takes six extra trilinear samples.                            |
| `--raycast-seed-previous`           | off                                   | Start each ray just in front of the surface from the last frame, not at the minimum depth.                                                |
| `--raycast-tsdf-from-valid-corners` | off                                   | Interpolate the raycast TSDF only where all 8 corners are valid. The default reweights the valid corners.                                 |
| `--icp-device-solve`                | off                                   | Run the full ICP Gauss-Newton loop on the GPU, with one synchronization for each pyramid level.                                           |
| `--icp-capture-graph`               | off                                   | Record the device ICP graph with stream capture. The default builds it with the explicit node API.                                        |

### Which ablations change the output?

- **Identical output.** `--raycast bitmap-march` and `--icp-capture-graph` give
  output that is equal to the baseline, bit for bit.
- **Approximate output.** The differences are small and bounded.
  - `--voxel quantized` and `--voxel bf16`: the voxel storage rounds each TSDF
    value to its quantization step.
  - `--raycast band-march`: the interpolation endpoints shift by less than one
    voxel.
  - `--cell-normals`: a different normal estimate; the surface points do not
    change.
  - `--raycast-seed-previous`: a ray can miss a new surface that appears much
    closer than the surface from the last frame.
  - `--icp-device-solve`: the loop runs a fixed iteration count; the poses
    differ at drift level.
- **Lossy output.** Information is lost.
  - `--integration band`: free space is not carved. A surface that moves away
    leaves stale voxels behind.
  - `--storage sparse`: the same band loss, and blocks past the pool capacity
    are dropped.
  - `--color none`: no color is stored. The geometry stays identical.
- **Different algorithm.** `--tsdf-variant`, `--projective-tsdf-distance`,
  `--distance-scaled-truncation`, and `--raycast-tsdf-from-valid-corners` change
  the rule itself. The result is different by design, not degraded.
- **A note on `--space cuda`**: float rounding differs from the cpu path, and
  the ICP reduction order makes GPU runs not repeat bit for bit.

Rules:

- `--storage sparse` needs `--integration band` and `--raycast march`. The
  program rejects other combinations.
- Set `--volume-camera-margin 0.5` when `--volume-resolution` is below 512 (see
  [Notes](#notes)).

Run one ablation:

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames -1 \
  --space cuda --voxel bf16 --output-dir outputs
```

Measure speed. `--preload` decodes the dataset into memory first, so the frame
loop measures only the pipeline:

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames -1 \
  --space cuda --preload --no-write-raycast-images --no-write-point-clouds \
  --output-dir outputs
```

### CUDA speed by configuration

Measured on one RTX 5080 with the command above (TUM `freiburg1_desk`, 595
frames, default 512^3 volume). Each row changes one switch against the baseline.
The numbers are rounded; run-to-run noise is about ±5%. Warm up the GPU with one
run and measure with the next runs.

| Configuration                       | fps | Why this speed                                                                                   |
| ----------------------------------- | --: | ------------------------------------------------------------------------------------------------ |
| baseline (all defaults)             | 545 | Reference: full integration, plain march, 8-byte float voxels, color on.                         |
| `--cell-normals`                    | 840 | Fastest. One 8-corner gather per hit pixel replaces six extra trilinear samples.                 |
| `--integration band`                | 680 | Each frame updates only blocks near the surface, far fewer voxels than the full sweep.           |
| `--icp-device-solve`                | 570 | The ICP loop synchronizes once per pyramid level, not once per iteration.                        |
| `--tsdf-variant classic`            | 565 | The constant weight needs less math per observation.                                             |
| `--color none`                      | 560 | No color buffer, so less memory traffic in integration and raycast.                              |
| `--distance-scaled-truncation`      | 550 | A slightly wider band; the extra cost is small.                                                  |
| `--no-projective-tsdf-distance`     | 550 | Same memory traffic, similar math.                                                               |
| `--voxel bf16`                      | 550 | Half the voxel bytes, and the decode is one shift: same speed as float.                          |
| `--voxel quantized`                 | 545 | Half the voxel bytes, but the int16 decode adds convert instructions; the two cancel.            |
| `--icp-capture-graph`               | 540 | Changes only how the ICP graph is built, not what runs.                                          |
| `--raycast-seed-previous`           | 540 | Rays start near the last surface, but the seed logic costs about the same as it saves.           |
| `--raycast-tsdf-from-valid-corners` | 515 | The raycast rejects more samples, so rays march farther before they hit.                         |
| `--storage sparse` + `band`         | 505 | Band speed minus the block allocation pass that runs each frame.                                 |
| `--raycast bitmap-march`            | 140 | Output is identical to `march`, but the per-frame bitmap rebuild costs more than the skip saves. |
| `--raycast band-march`              | 135 | Same rebuild cost; the larger skip does not pay for it either.                                   |

The two skip backends are a negative result. They stay in the code as ablations,
not as optimizations.

### Ablation pipeline sets

`--pipelines <file.toml>` runs several independently configured pipelines on the
same input frames and compares each against a reference pipeline:

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames 60 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --pipelines configs/ablation_example.toml --compare-every 10 --output-dir outputs
```

Per-frame deviation statistics against the reference are logged and appended to
`<output-dir>/ablation_stats.csv` (truncated at the start of each run). The
reference pipeline writes its raycasts to `<output-dir>/` every frame; the other
pipelines write theirs to `<output-dir>/<pipeline-name>/` on comparison frames.
At the end of the run each pipeline also writes its own `mesh.ply`, so you can
compare mesh quality across voxel storages.

The TOML file has two optional top-level keys and one `[[pipeline]]` table per
pipeline. See `configs/ablation_example.toml` for a commented example, and
`configs/ablate_all.toml` for a set with one pipeline per ablatable parameter
(run it at `--volume-resolution 256` so all volumes fit in GPU memory):

```bash
./build/release/src/app/kinectfusion data/rgbd_dataset_freiburg1_xyz --frames -1 \
  --pipelines configs/ablate_all.toml --volume-resolution 256 \
  --volume-camera-margin 0.5 --space cuda --compare-every 10 --output-dir outputs
```

```toml
reference = "baseline"     # defaults to the first pipeline
compare-every-n-frames = 1 # defaults to --compare-every; <= 0 disables

[[pipeline]]
name = "baseline"

[[pipeline]]
name = "bitmap"
raycast = "bitmap-march"   # must compare equal to the baseline, bit for bit

[[pipeline]]
name = "sparse"
storage = "sparse"
integration = "band"
```

Every pipeline starts from the CLI-derived configuration and overrides only what
it ablates:

| Key                          | Type   | Meaning                                                 |
| ---------------------------- | ------ | ------------------------------------------------------- |
| `name`                       | string | Required, unique; names the output subdirectory         |
| `space`                      | string | `"cpu"` or `"cuda"`                                     |
| `tsdf-variant`               | string | `"angle-weighted"` or `"classic"`                       |
| `voxel`                      | string | `"float"`, `"quantized"`, or `"bf16"`                   |
| `color`                      | string | `"float"` or `"none"`                                   |
| `storage`                    | string | `"dense"` or `"sparse"`                                 |
| `sparse-capacity`            | int    | Sparse block pool size; `0` = one quarter of the blocks |
| `integration`                | string | `"full"` or `"band"`                                    |
| `raycast`                    | string | `"march"`, `"bitmap-march"`, or `"band-march"`          |
| `projective-distance`        | bool   | Projective instead of euclidean surface distance        |
| `distance-scaled-truncation` | bool   | Scale truncation with measured depth                    |
| `truncation-distance-scale`  | float  | Factor for the scaled truncation                        |
| `observation-weight`         | float  | Weight of one observation                               |
| `max-weight`                 | float  | Running-average weight cap                              |

The volume geometry is deliberately not overridable — it stays global (CLI
flags) so cross-pipeline comparison is defined on identical grids. Unknown keys
throw `std::invalid_argument`: ablation configs fail loud rather than silently
running a default.

## Development

### Formatting

If the extensions under `.vscode/extensions.json` are installed, the formatting
will applied on save.

To format all files of the project, run:

```bash
find include src test -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cc' -o -name '*.cxx' -o -name '*.c' \) -print0   | xargs -0 -P "$(nproc)" -n 20 clang-format -i --style=file
```

## Dockerization

> **Note:** Docker was only configured for the CPU implementation and is not
> supported with the current CUDA implementation yet.

The repo's `compose.yaml` defines a `kinectfusion` service that builds the
production image and runs the binary as a one-shot CLI (mirroring the `dev`
service used for the [dev container](#development-container)).

When using Dockerized setup, Conda environment is not needed.

### Release

#### Build

```bash
docker compose build kinectfusion
```

The intermediate stage compiles and runs the test suite (with all warnings,
clang-tidy, cppcheck, and the sanitizers enabled), so any failure aborts the
build. The final image's entrypoint is the binary.

#### Run

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

### Build and run

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
