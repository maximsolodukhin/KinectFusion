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

// A null-data source empties the image.
template <typename Img, typename View>
  requires image_proc::RefillableFrom<Img, View>
void assign_image(Img& image, const View& source) {
  if (source.data == nullptr) {
    image = {};
    return;
  }
  image.ensure_extent(source.width, source.height);
  image.copy_from(source);
}

}  // namespace

void DeviceDepthFrame::assign(const DepthFrame& frame) {
  const HostDepthFrameView host = frame.view();
  intrinsics_ = host.intrinsics;
  world_to_camera_ = host.world_to_camera;
  assign_image(depth_, host.depth);
  assign_image(color_, host.color);
  assign_image(normals_, host.normals);
}

void DeviceDepthFrame::assign_from_pyramid(const DepthFrame& frame,
                                           const DeviceDepthImg& raw_depth,
                                           const DeviceVec3fImg& normals) {
  intrinsics_ = frame.intrinsics;
  world_to_camera_ = from_eigen(frame.world_to_camera);
  assign_image(depth_, raw_depth.view());
  if (frame.color != nullptr) {
    assign_image(color_, frame.color->view());
  } else {
    color_ = {};
  }
  assign_image(normals_, normals.view());
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
}

}  // namespace kinectfusion
