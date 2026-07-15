#include <algorithm>
#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/icp_correspondence.hpp>

namespace kinectfusion {

namespace {

static_assert(sizeof(std::size_t) == sizeof(unsigned long long));

// Grid-stride over the live image; each thread accumulates its matches
// locally and merges once into the global system.
__global__ void reduce_correspondences_kernel(DeviceCorrespondenceSearch search,
                                              IcpNormalEquations* result) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

  const std::size_t width = search.live().vertices.width;
  const std::size_t pixel_count = width * search.live().vertices.height;
  IcpNormalEquations local;
  for (std::size_t index = first; index < pixel_count; index += stride) {
    if (const auto match = search.match(index % width, index / width)) {
      local.accumulate(*match);
    }
  }
  if (local.count == 0) {
    return;
  }

  for (std::size_t entry = 0; entry < kIcpUpperTriangleSize; ++entry) {
    atomicAdd(&result->jtj[entry], local.jtj[entry]);
  }
  for (std::size_t entry = 0; entry < kIcpDof; ++entry) {
    atomicAdd(&result->jtr[entry], local.jtr[entry]);
  }
  atomicAdd(&result->distance_sum, local.distance_sum);
  atomicAdd(reinterpret_cast<unsigned long long*>(&result->count),
            static_cast<unsigned long long>(local.count));
}

}  // namespace

IcpNormalEquations DeviceCorrespondenceSweep::run(
    const DeviceCorrespondenceSearch& search) {
  const std::size_t pixel_count =
      search.live().vertices.width * search.live().vertices.height;
  if (pixel_count == 0) {
    return {};
  }

  cuda::DeviceBuffer<IcpNormalEquations> result{1};

  constexpr unsigned int kBlock = 256;
  constexpr unsigned int kMaxGrid = 256;
  const unsigned int grid =
      std::min(cuda::ceil_div(pixel_count, kBlock), kMaxGrid);

  reduce_correspondences_kernel<<<grid, kBlock>>>(search, result.data());
  cuda::check(cudaGetLastError(), "reduce_correspondences_kernel launch");
  cuda::check(cudaDeviceSynchronize(), "reduce_correspondences_kernel");

  IcpNormalEquations equations;
  result.copy_to_host(&equations, 1);
  return equations;
}

}  // namespace kinectfusion
