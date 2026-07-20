#include <algorithm>
#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/registered_storages.hpp>

namespace kinectfusion {

namespace {

// Grid-stride count; each thread accumulates locally and contributes one
// atomic add.
template <DenseVoxelGridView VolumeViewT>
__global__ void count_observed_kernel(VolumeViewT volume,
                                      unsigned long long* result) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

  unsigned long long count = 0;
  for (std::size_t index = first; index < volume.voxel_count();
       index += stride) {
    if (volume.voxels[index].weight_value() > 0.0F) {
      ++count;
    }
  }
  if (count > 0) {
    atomicAdd(result, count);
  }
}

}  // namespace

template <DenseVoxelGridView VolumeViewT>
std::size_t DeviceVolumeReduction::observed_voxel_count(
    const VolumeViewT& volume) {
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

// One instantiation per registered storage combination.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KINECTFUSION_INSTANTIATE(GeomVoxel, Color)                  \
  template std::size_t DeviceVolumeReduction::observed_voxel_count( \
      const ConstVolumeView<MemorySpace::kDevice, GeomVoxel, Color>&);
KINECTFUSION_FOR_EACH_REGISTERED_STORAGE(KINECTFUSION_INSTANTIATE)
#undef KINECTFUSION_INSTANTIATE

}  // namespace kinectfusion
