# KinectFusion
Build instructions for KinectFusion:
```bash
git clone git@github.com:maximsolodukhin/KinectFusion.git
cd KinectFusion
mkdir build
cmake -S . -B build
cmake --build build -j 5
```

# Running KinectFusion

## 1. Build

```bash
cmake -S . -B build
cmake --build build --target intro -j 5
```

The executable is `build/src/app/intro`.

## 2. Get a dataset

A TUM RGB-D sequence (the `freiburg1_xyz` desk scene) is expected. The
directory must contain `depth.txt`, `rgb.txt`, and the `depth/` + `rgb/`
folders. Download from
<https://vision.in.tum.de/data/datasets/rgbd-dataset/download>.

This repo expects it at the project root, e.g. `rgbd_dataset_freiburg1_xyz/`.

## 3. Run

```bash
./build/src/app/intro rgbd_dataset_freiburg1_xyz --frames 5 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

This writes one raycast render per frame to `outputs/`:

- `frame_XXXXXX_raycast.png` — color raycast of the fused TSDF surface
- `frame_XXXXXX_raycast.ply` — surface point cloud (position + normal + color)

Single frame only (just integrate + raycast, no tracking):

```bash
./build/src/app/intro rgbd_dataset_freiburg1_xyz --frames 1 \
  --volume-resolution 256 --voxel-size 0.01 --volume-camera-margin 0.5 \
  --output-dir outputs
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `dataset` (positional) | `../data/rgbd_dataset_freiburg1_xyz` | TUM RGB-D dataset directory |
| `--frames` | `30` | Frames to process (`-1` for all) |
| `--volume-resolution` | `512` | Cubic TSDF resolution in voxels |
| `--voxel-size` | `0.01` | Voxel size in meters |
| `--truncation-distance` | `0.05` | TSDF truncation distance in meters |
| `--volume-camera-margin` | `2.56` | Space behind the initial camera in meters |
| `--output-dir` | `kinectfusion_output` | Output directory |
| `--no-write-raycast-images` | — | Skip PNG output |
| `--no-write-point-clouds` | — | Skip PLY output |

`./build/src/app/intro --help` lists everything.

## Notes

- **Use a small `--volume-camera-margin`** (e.g. `0.5`). The default `2.56`
  equals the full volume extent, which places the volume *behind* the camera
  (which looks down +z) and reconstructs nothing.
- **Performance**: the integrator sweeps the whole grid every frame on a single
  CPU thread, so `--volume-resolution 256` is ~20–40 s/frame. Use `128` for
  quick iteration, higher for more detail.
- The volume is fixed in world space at the first frame's pose; keep sequences
  short enough that the camera stays inside it.

