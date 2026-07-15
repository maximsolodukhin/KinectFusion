#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/raycasting.cuh>
#include <kinectfusion/volume_sampler.hpp>

namespace kinectfusion {

namespace {

// One thread per pixel.
__global__ void raycast_kernel(DeviceSurfaceRaycast raycast,
                               DeviceSurfaceMapsView maps) {
  const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (col >= maps.points.width || row >= maps.points.height) {
    return;
  }
  maps.points.at(col, row) = invalid_vec3f();
  maps.normals.at(col, row) = invalid_vec3f();
  maps.colors.at(col, row) = 0;
  raycast.render_pixel(maps, col, row);
}

}  // namespace

DeviceSurfaceMaps::DeviceSurfaceMaps(std::size_t width, std::size_t height)
    : points_(width, height), normals_(width, height), colors_(width, height) {}

DeviceSurfaceMapsView DeviceSurfaceMaps::view() {
  return DeviceSurfaceMapsView{.points = points_.view(),
                               .normals = normals_.view(),
                               .colors = colors_.view()};
}

ConstDeviceSurfaceMapsView DeviceSurfaceMaps::view() const {
  return ConstDeviceSurfaceMapsView{.points = points_.view(),
                                    .normals = normals_.view(),
                                    .colors = colors_.view()};
}

SurfaceMaps DeviceSurfaceMaps::download() const {
  SurfaceMaps maps = SurfaceMaps::allocate(points_.width(), points_.height());
  points_.copy_to(maps.points.view());
  normals_.copy_to(maps.normals.view());
  colors_.copy_to(maps.colors.view());
  return maps;
}

void DeviceRaycastSweep::run(const DeviceSurfaceRaycast& raycast,
                             const DeviceSurfaceMapsView& maps) {
  constexpr dim3 kBlock{16, 16};
  const dim3 grid{cuda::ceil_div(maps.points.width, kBlock.x),
                  cuda::ceil_div(maps.points.height, kBlock.y)};
  raycast_kernel<<<grid, kBlock>>>(raycast, maps);
  cuda::check(cudaGetLastError(), "raycast_kernel launch");
  cuda::check(cudaDeviceSynchronize(), "raycast_kernel");
}

}  // namespace kinectfusion
