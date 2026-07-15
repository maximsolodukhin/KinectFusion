#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/tsdf_integration.cuh>
#include <variant>

namespace kinectfusion {

namespace {

// One thread per (x, y) column sweeping z
template <TsdfUpdateRule<MemorySpace::kDevice> Rule>
__global__ void integrate_kernel(
    VolumeView<MemorySpace::kDevice> volume,
    IntegrationContext<MemorySpace::kDevice> context, Rule rule) {
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

[[nodiscard]] unsigned int ceil_div(std::size_t count, unsigned int divisor) {
  return static_cast<unsigned int>((count + divisor - 1) / divisor);
}

}  // namespace

DeviceDepthFrame DeviceDepthFrame::upload(const DepthFrame& frame) {
  const HostDepthFrameView host = frame.view();

  DeviceDepthFrame device;
  device.intrinsics_ = host.intrinsics;
  device.rotation_ = host.rotation;
  device.translation_ = host.translation;
  if (host.depth.data != nullptr) {
    device.depth_ =
        image_proc::DeviceDepthImage{host.depth.width, host.depth.height};
    device.depth_.copy_from(host.depth);
  }
  if (host.color.data != nullptr) {
    device.color_ =
        image_proc::DeviceColorImage{host.color.width, host.color.height};
    device.color_.copy_from(host.color);
  }
  if (host.normals.data != nullptr) {
    device.normals_ = image_proc::DeviceVector3fImage{host.normals.width,
                                                      host.normals.height};
    device.normals_.copy_from(host.normals);
  }
  return device;
}

DeviceDepthFrameView DeviceDepthFrame::view() const {
  return DeviceDepthFrameView{.depth = depth_.view(),
                              .color = color_.view(),
                              .normals = normals_.view(),
                              .intrinsics = intrinsics_,
                              .rotation = rotation_,
                              .translation = translation_};
}

void DeviceIntegrationSweep::run(
    const VolumeView<MemorySpace::kDevice>& volume,
    const IntegrationContext<MemorySpace::kDevice>& context,
    const TsdfRuleVariant& rule) {
  constexpr dim3 kBlock{16, 16};
  const dim3 grid{ceil_div(volume.resolution().x, kBlock.x),
                  ceil_div(volume.resolution().y, kBlock.y)};
  std::visit(
      [&](const auto& concrete) {
        integrate_kernel<<<grid, kBlock>>>(volume, context, concrete);
      },
      rule);
  cuda::check(cudaGetLastError(), "integrate_kernel launch");
  cuda::check(cudaDeviceSynchronize(), "integrate_kernel");
}

}  // namespace kinectfusion
