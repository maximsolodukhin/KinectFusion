#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <memory>
#include <utility>

namespace kinectfusion {
namespace {

// every worker exposes operator()(col, row) -> optional value, with `fallback`
// filling the rest.
template <typename Worker, typename PixelT>
void fill_pixels(const Worker& worker, image_proc::HostImageView<PixelT> output,
                 PixelT fallback) {
  for (const auto [col, row] : PixelIndices{output.width, output.height}) {
    output.at(col, row) = worker(col, row).value_or(fallback);
  }
}

class HostPyramidSource final : public PyramidSource {
 public:
  explicit HostPyramidSource(const DepthProcessingOptions& options)
      : processor_(options) {}

  std::size_t build(const image_proc::DepthImage& raw_depth,
                    const CameraIntrinsics& intrinsics) override {
    pyramid_ = processor_.build_surface_pyramid(raw_depth, intrinsics);
    return pyramid_.size();
  }

  [[nodiscard]] PyramidLevel level(std::size_t index) const override {
    const auto& source = pyramid_.at(index);
    return {.surface = view(source.surface), .intrinsics = source.intrinsics};
  }

  [[nodiscard]] const image_proc::Vector3fImage* host_normals() const override {
    return pyramid_.empty() ? nullptr : &pyramid_.front().surface.normals;
  }

  [[nodiscard]] const DeviceDepthFrame* device_frame(
      const DepthFrame& /*frame*/) override {
    return nullptr;
  }

 private:
  DepthProcessor<MemorySpace::kHost> processor_;
  SurfacePyramid pyramid_;
};

}  // namespace

PyramidSource::Creation PyramidSource::create(
    MemorySpace space, const DepthProcessingOptions& options) {
  if (space == MemorySpace::kDevice) {
#ifdef KINECTFUSION_HAS_CUDA
    return {.source = create_device(options), .fallback_reason = {}};
#else
    return {.source = std::make_unique<HostPyramidSource>(options),
            .fallback_reason =
                "device memory space unavailable; processing depth on host"};
#endif
  }
  return {.source = std::make_unique<HostPyramidSource>(options),
          .fallback_reason = {}};
}

DepthProcessingOptions DepthProcessingOptions::validated(
    DepthProcessingOptions options) {
  require(options.levels > 0U, "Depth pyramid must have at least one level");
  require(options.depth_scale > 0.0F, "Depth scale must be positive");
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Depth range is invalid");
  require(options.max_normal_depth_jump >= 0.0F,
          "Normal depth jump threshold must be non-negative");
  require(options.max_downsample_depth_jump >= 0.0F,
          "Downsample depth jump threshold must be non-negative");
  require(options.bilateral_radius >= 0,
          "Bilateral filter radius must be non-negative");
  require(options.bilateral_spatial_sigma > 0.0F,
          "Bilateral spatial sigma must be positive");
  require(options.bilateral_depth_sigma > 0.0F,
          "Bilateral depth sigma must be positive");
  return options;
}

DepthProcessor<MemorySpace::kHost>::DepthProcessor(
    DepthProcessingOptions options)
    : options_(DepthProcessingOptions::validated(options)) {}

auto DepthProcessor<MemorySpace::kHost>::bilateral_filter(
    const DepthImg& depth_image) const -> DepthImg {
  if (!options_.bilateral_filter || options_.bilateral_radius == 0) {
    return depth_image;
  }

  DepthImg filtered{depth_image.width(), depth_image.height()};
  fill_pixels(HostBilateralFilter{depth_image.view(), options_},
              filtered.view(), std::uint16_t{0});
  return filtered;
}

auto DepthProcessor<MemorySpace::kHost>::downsample(
    const DepthImg& depth_image) const -> DepthImg {
  const std::size_t width = depth_image.width() / kDownsampleFactor;
  const std::size_t height = depth_image.height() / kDownsampleFactor;

  DepthImg downsampled{width, height};
  fill_pixels(HostBlockDownsample{depth_image.view(), options_},
              downsampled.view(), std::uint16_t{0});
  return downsampled;
}

auto DepthProcessor<MemorySpace::kHost>::project_to_vertices(
    const DepthImg& depth_image, const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose) const -> Vec3fImg {
  Vec3fImg vertices{depth_image.width(), depth_image.height()};
  fill_pixels(HostVertexProjection{depth_image.view(), intrinsics,
                                   from_eigen(camera_pose), options_},
              vertices.view(), invalid_vec3f());
  return vertices;
}

auto DepthProcessor<MemorySpace::kHost>::compute_normals(
    const Vec3fImg& vertices) const -> Vec3fImg {
  Vec3fImg normals{vertices.width(), vertices.height()};
  fill_pixels(HostNormalEstimation{vertices.view(), options_}, normals.view(),
              invalid_vec3f());
  return normals;
}

SurfacePyramid DepthProcessor<MemorySpace::kHost>::build_surface_pyramid(
    const DepthImg& depth_image, const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose) const {
  SurfacePyramid pyramid;
  pyramid.reserve(options_.levels);
  DepthImg current = bilateral_filter(depth_image);
  for (unsigned int level = 0; level < options_.levels; ++level) {
    if (current.width() == 0U || current.height() == 0U) {
      break;
    }
    const CameraIntrinsics level_intrinsics = intrinsics.scaled(level);
    Vec3fImg vertices =
        project_to_vertices(current, level_intrinsics, camera_pose);
    Vec3fImg normals = compute_normals(vertices);
    // Downsample before `current` is moved into the pyramid; the last level
    // leaves `current` empty, ending the loop.
    DepthImg next;
    if (level + 1U < options_.levels) {
      next = downsample(current);
    }
    pyramid.emplace_back(std::move(current), level_intrinsics,
                         std::move(vertices), std::move(normals));
    current = std::move(next);
  }
  return pyramid;
}

}  // namespace kinectfusion
