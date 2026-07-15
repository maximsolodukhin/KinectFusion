#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/tsdf_integration.cuh>
#include <variant>

namespace kinectfusion {

namespace {

// One thread per (x, y) column sweeping z
template <TsdfUpdateRule<MemorySpace::kDevice> Rule>
__global__ void integrate_kernel(DeviceVolumeView volume,
                                 DeviceIntegrationContext context, Rule rule) {
  const std::size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const Size3 resolution = volume.resolution();
  if (x >= resolution.x || y >= resolution.y) {
    return;
  }
  for (std::size_t z = 0; z < resolution.z; ++z) {
    rule.update(volume, context, x, y, z);
  }
}

}  // namespace

DeviceDepthFrame DeviceDepthFrame::upload(const DepthFrame& frame) {
  using DeviceDepthImg = image_proc::DeviceDepthImage;
  using DeviceColorImg = image_proc::DeviceColorImage;
  using DeviceVec3fImg = image_proc::DeviceVector3fImage;
  const HostDepthFrameView host = frame.view();

  DeviceDepthFrame device;
  device.intrinsics_ = host.intrinsics;
  device.world_to_camera_ = host.world_to_camera;
  if (host.depth.data != nullptr) {
    device.depth_ = DeviceDepthImg::uploaded(host.depth);
  }
  if (host.color.data != nullptr) {
    device.color_ = DeviceColorImg::uploaded(host.color);
  }
  if (host.normals.data != nullptr) {
    device.normals_ = DeviceVec3fImg::uploaded(host.normals);
  }
  return device;
}

DeviceDepthFrameView DeviceDepthFrame::view() const {
  return DeviceDepthFrameView{.depth = depth_.view(),
                              .color = color_.view(),
                              .normals = normals_.view(),
                              .intrinsics = intrinsics_,
                              .world_to_camera = world_to_camera_};
}

void DeviceIntegrationSweep::run(const DeviceVolumeView& volume,
                                 const DeviceIntegrationContext& context,
                                 const TsdfRuleVariant& rule) {
  constexpr dim3 kBlock{16, 16};
  const dim3 grid{cuda::ceil_div(volume.resolution().x, kBlock.x),
                  cuda::ceil_div(volume.resolution().y, kBlock.y)};
  std::visit(
      [&](const auto& concrete) {
        integrate_kernel<<<grid, kBlock>>>(volume, context, concrete);
      },
      rule);
  cuda::check(cudaGetLastError(), "integrate_kernel launch");
  cuda::check(cudaDeviceSynchronize(), "integrate_kernel");
}

}  // namespace kinectfusion
