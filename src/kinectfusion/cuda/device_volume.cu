#include <algorithm>
#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/device_volume.cuh>

namespace kinectfusion {

namespace {

// Grid-stride count; each thread accumulates locally and contributes one
// atomic add.
__global__ void count_observed_kernel(ConstDeviceVolumeView volume,
                                      unsigned long long* result) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

  unsigned long long count = 0;
  for (std::size_t index = first; index < volume.voxel_count();
       index += stride) {
    if (volume.voxels[index].weight > 0.0F) {
      ++count;
    }
  }
  if (count > 0) {
    atomicAdd(result, count);
  }
}

}  // namespace

std::size_t DeviceVolumeReduction::observed_voxel_count(
    const ConstDeviceVolumeView& volume) {
  cuda::DeviceBuffer<unsigned long long> result{1};

  constexpr unsigned int kBlock = 256;
  constexpr unsigned int kMaxGrid = 256;
  const unsigned int grid =
      std::min(cuda::ceil_div(volume.voxel_count(), kBlock), kMaxGrid);

  count_observed_kernel<<<grid, kBlock>>>(volume, result.data());
  cuda::check(cudaGetLastError(), "count_observed_kernel launch");

  unsigned long long count = 0;
  result.copy_to_host(&count, 1);
  return static_cast<std::size_t>(count);
}

}  // namespace kinectfusion
