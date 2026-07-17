#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/depth_processing.cuh>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/tsdf_integration.cuh>
#include <memory>
#include <utility>

namespace kinectfusion {

namespace {

using image_proc::DeviceImageView;

constexpr dim3 kBlock{16, 16};

// Every worker exposes operator()(col, row) -> optional value, with
// `fallback` filling the rest; one kernel and one launcher cover the whole
// pyramid.
template <typename Worker, typename PixelT>
__global__ void per_pixel_kernel(Worker worker, PixelT fallback,
                                 DeviceImageView<PixelT> output) {
  const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (col >= output.width || row >= output.height) {
    return;
  }
  output.at(col, row) = worker(col, row).value_or(fallback);
}

template <typename Worker, typename PixelT>
void launch_per_pixel(const Worker& worker, PixelT fallback,
                      DeviceImageView<PixelT> output, const char* name) {
  const dim3 grid{cuda::ceil_div(output.width, kBlock.x),
                  cuda::ceil_div(output.height, kBlock.y)};
  per_pixel_kernel<<<grid, kBlock>>>(worker, fallback, output);
  cuda::check(cudaGetLastError(), name);
}

using DeviceLevel = DepthProcessingLevel<MemorySpace::kDevice>;

// Level slot with images of the requested dimensions, reusing allocations from
// the previous frame when they still fit.
void ensure_level(SurfacePyramidFor<MemorySpace::kDevice>& pyramid,
                  std::size_t level, std::size_t width, std::size_t height,
                  const CameraIntrinsics& intrinsics) {
  if (pyramid.size() == level) {
    pyramid.emplace_back(DeviceDepthImg::uninitialized(width, height),
                         intrinsics,
                         DeviceVec3fImg::uninitialized(width, height),
                         DeviceVec3fImg::uninitialized(width, height));
    return;
  }
  DeviceLevel& slot = pyramid.at(level);
  slot.depth_image.ensure_extent(width, height);
  slot.surface.vertices.ensure_extent(width, height);
  slot.surface.normals.ensure_extent(width, height);
  slot.intrinsics = intrinsics;
}

}  // namespace

DepthProcessor<MemorySpace::kDevice>::DepthProcessor(
    DepthProcessingOptions options)
    : options_(DepthProcessingOptions::validated(options)) {}

void DepthProcessor<MemorySpace::kDevice>::build_surface_pyramid(
    image_proc::DeviceImageView<const std::uint16_t> depth,
    const CameraIntrinsics& intrinsics, SurfacePyramid& pyramid) const {
  // All vector growth happens before any level reference is taken.
  std::size_t width = depth.width;
  std::size_t height = depth.height;
  std::size_t level_count = 0;
  for (unsigned int level = 0; level < options_.levels; ++level) {
    if (width == 0U || height == 0U) {
      break;
    }
    ensure_level(pyramid, level, width, height, intrinsics.scaled(level));
    ++level_count;
    width /= kDownsampleFactor;
    height /= kDownsampleFactor;
  }
  pyramid.erase(pyramid.begin() + static_cast<std::ptrdiff_t>(level_count),
                pyramid.end());

  for (std::size_t level = 0; level < level_count; ++level) {
    DeviceLevel& current = pyramid.at(level);

    if (level == 0U) {
      if (!options_.bilateral_filter || options_.bilateral_radius == 0) {
        current.depth_image.copy_from(depth);
      } else {
        launch_per_pixel(DeviceBilateralFilter{depth, options_},
                         std::uint16_t{0}, current.depth_image.view(),
                         "bilateral filter kernel");
      }
    }

    launch_per_pixel(
        DeviceVertexProjection{current.depth_image.view(), current.intrinsics,
                               RigidTransform::identity(), options_},
        invalid_vec3f(), current.surface.vertices.view(),
        "vertex projection kernel");

    launch_per_pixel(
        DeviceNormalEstimation{current.surface.vertices.view(), options_},
        invalid_vec3f(), current.surface.normals.view(),
        "normal estimation kernel");

    if (level + 1U < level_count) {
      launch_per_pixel(
          DeviceBlockDownsample{current.depth_image.view(), options_},
          std::uint16_t{0}, pyramid.at(level + 1U).depth_image.view(),
          "block downsample kernel");
    }
  }
}

namespace {

class DevicePyramidSource final : public PyramidSource {
 public:
  explicit DevicePyramidSource(const DepthProcessingOptions& options)
      : processor_(options) {}

  std::size_t build(const image_proc::DepthImage& raw_depth,
                    const CameraIntrinsics& intrinsics) override {
    raw_depth_.ensure_extent(raw_depth.width(), raw_depth.height());
    raw_depth_.copy_from(raw_depth.view());
    processor_.build_surface_pyramid(raw_depth_.view(), intrinsics, pyramid_);
    return pyramid_.size();
  }

  [[nodiscard]] PyramidLevel level(std::size_t index) const override {
    const auto& source = pyramid_.at(index);
    return {.surface = view(source.surface), .intrinsics = source.intrinsics};
  }

  [[nodiscard]] const image_proc::Vector3fImage* host_normals() const override {
    return nullptr;
  }

  [[nodiscard]] const DeviceDepthFrame* device_frame(
      const DepthFrame& frame) override {
    if (pyramid_.empty()) {
      return nullptr;
    }
    frame_.assign_from_pyramid(frame, raw_depth_,
                               pyramid_.front().surface.normals);
    return &frame_;
  }

 private:
  DeviceDepthProcessor processor_;
  DeviceDepthImg raw_depth_;
  SurfacePyramidFor<MemorySpace::kDevice> pyramid_;
  DeviceDepthFrame frame_;
};

}  // namespace

std::unique_ptr<PyramidSource> PyramidSource::create_device(
    const DepthProcessingOptions& options) {
  return std::make_unique<DevicePyramidSource>(options);
}

}  // namespace kinectfusion
