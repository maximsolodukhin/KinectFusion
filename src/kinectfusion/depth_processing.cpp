#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/validation.hpp>
#include <limits>
#include <optional>
#include <utility>

namespace kinectfusion {
namespace {

// Linear resolution reduction per pyramid level: each level halves width and
// height, downsampling over a kDownsampleFactor x kDownsampleFactor block.
constexpr std::size_t kDownsampleFactor = 2U;

[[nodiscard]] DepthProcessingOptions validated(DepthProcessingOptions options) {
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

// Exponent scale of a Gaussian falloff: exp(-d^2 / (2 sigma^2)).
[[nodiscard]] constexpr float gaussian_exponent_scale(float sigma) {
  return -0.5F / (sigma * sigma);
}

}  // namespace

DepthProcessor<MemorySpace::kHost>::DepthProcessor(
    DepthProcessingOptions options)
    : options_(validated(options)),
      spatial_scale_(gaussian_exponent_scale(options_.bilateral_spatial_sigma)),
      range_scale_(gaussian_exponent_scale(options_.bilateral_depth_sigma)) {}

std::optional<float> DepthProcessor<MemorySpace::kHost>::usable_depth(
    std::uint16_t raw) const {
  return depth_in_range(raw, options_.depth_scale, options_.min_depth,
                        options_.max_depth);
}

image_proc::DepthImage DepthProcessor<MemorySpace::kHost>::bilateral_filter(
    const image_proc::DepthImage& depth_image) const {
  if (!options_.bilateral_filter || options_.bilateral_radius == 0) {
    return depth_image;
  }

  image_proc::DepthImage filtered{depth_image.width(), depth_image.height()};
  for (std::size_t row = 0; row < depth_image.height(); ++row) {
    for (std::size_t col = 0; col < depth_image.width(); ++col) {
      const auto center_meters = usable_depth(depth_image.at(col, row));
      if (!center_meters) {
        continue;
      }
      if (const auto value =
              bilateral_filtered_pixel(depth_image, static_cast<int>(col),
                                       static_cast<int>(row), *center_meters)) {
        filtered.at(col, row) = *value;
      }
    }
  }
  return filtered;
}

std::optional<std::uint16_t>
DepthProcessor<MemorySpace::kHost>::bilateral_filtered_pixel(
    const image_proc::DepthImage& depth_image, int col, int row,
    float center_meters) const {
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());
  const int radius = options_.bilateral_radius;
  const int x_begin = std::max(col - radius, 0);
  const int x_end = std::min(col + radius, width - 1);
  const int y_end = std::min(row + radius, height - 1);

  float weighted_sum = 0.0F;
  float weight_sum = 0.0F;
  for (int y = std::max(row - radius, 0); y <= y_end; ++y) {
    for (int x = x_begin; x <= x_end; ++x) {
      const auto raw = depth_image.at(static_cast<std::size_t>(x),
                                      static_cast<std::size_t>(y));
      const auto sample_meters = usable_depth(raw);
      if (!sample_meters) {
        continue;
      }
      const auto pixel_distance2 =
          static_cast<float>(((x - col) * (x - col)) + ((y - row) * (y - row)));
      const float depth_difference = *sample_meters - center_meters;
      const float weight =
          std::exp((pixel_distance2 * spatial_scale_) +
                   (depth_difference * depth_difference * range_scale_));
      weighted_sum += weight * static_cast<float>(raw);
      weight_sum += weight;
    }
  }
  if (weight_sum <= 0.0F) {
    return std::nullopt;
  }
  return static_cast<std::uint16_t>(std::lround(weighted_sum / weight_sum));
}

image_proc::DepthImage DepthProcessor<MemorySpace::kHost>::downsample(
    const image_proc::DepthImage& depth_image) const {
  const std::size_t width = depth_image.width() / kDownsampleFactor;
  const std::size_t height = depth_image.height() / kDownsampleFactor;
  image_proc::DepthImage downsampled{width, height};

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      if (const auto value = downsampled_block(depth_image, col, row)) {
        downsampled.at(col, row) = *value;
      }
    }
  }
  return downsampled;
}

std::optional<std::uint16_t>
DepthProcessor<MemorySpace::kHost>::downsampled_block(
    const image_proc::DepthImage& depth_image, std::size_t col,
    std::size_t row) const {
  std::uint32_t sum = 0;
  unsigned int count = 0;
  std::uint16_t minimum_observed_depth =
      std::numeric_limits<std::uint16_t>::max();
  std::uint16_t maximum_observed_depth = 0;

  for (std::size_t dy = 0; dy < kDownsampleFactor; ++dy) {
    for (std::size_t dx = 0; dx < kDownsampleFactor; ++dx) {
      const std::uint16_t sample = depth_image.at(
          (col * kDownsampleFactor) + dx, (row * kDownsampleFactor) + dy);
      if (sample == 0) {
        continue;
      }
      sum += sample;
      ++count;
      minimum_observed_depth = std::min(minimum_observed_depth, sample);
      maximum_observed_depth = std::max(maximum_observed_depth, sample);
    }
  }

  if (count == 0) {
    return std::nullopt;
  }
  if (depth_to_meters(maximum_observed_depth, options_.depth_scale) -
          depth_to_meters(minimum_observed_depth, options_.depth_scale) >
      options_.max_downsample_depth_jump) {
    return std::nullopt;
  }

  return static_cast<std::uint16_t>((sum + (count / 2U)) / count);
}

image_proc::Vector3fImage
DepthProcessor<MemorySpace::kHost>::project_to_vertices(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose) const {
  image_proc::Vector3fImage vertices{depth_image.width(), depth_image.height(),
                                     invalid_vec3f()};
  for (std::size_t row = 0; row < depth_image.height(); ++row) {
    for (std::size_t col = 0; col < depth_image.width(); ++col) {
      const auto depth = usable_depth(depth_image.at(col, row));
      if (!depth) {
        continue;
      }
      const Eigen::Vector3f camera_point = intrinsics.back_project(
          {static_cast<float>(col), static_cast<float>(row)}, *depth);
      // .homogeneous() makes the camera_point (x,y,z,1) to multiply with the
      // camera pose
      const Eigen::Vector3f world_point =
          (camera_pose * camera_point.homogeneous()).head<3>();
      vertices.at(col, row) = from_eigen(world_point);
    }
  }
  return vertices;
}

image_proc::Vector3fImage DepthProcessor<MemorySpace::kHost>::compute_normals(
    const image_proc::Vector3fImage& vertices) const {
  image_proc::Vector3fImage normals{vertices.width(), vertices.height(),
                                    invalid_vec3f()};
  if (vertices.width() < 3U || vertices.height() < 3U) {
    return normals;
  }
  // Interior pixels only: the stencil needs all four neighbours.
  for (std::size_t row = 1; row + 1 < vertices.height(); ++row) {
    for (std::size_t col = 1; col + 1 < vertices.width(); ++col) {
      if (const auto normal = stencil_normal(vertices, col, row)) {
        normals.at(col, row) = *normal;
      }
    }
  }
  return normals;
}

std::optional<Vec3f> DepthProcessor<MemorySpace::kHost>::stencil_normal(
    const image_proc::Vector3fImage& vertices, std::size_t col,
    std::size_t row) const {
  const Vec3f& left = vertices.at(col - 1, row);
  const Vec3f& right = vertices.at(col + 1, row);
  const Vec3f& top = vertices.at(col, row - 1);
  const Vec3f& bottom = vertices.at(col, row + 1);
  // NaN propagates through the sum
  if (!all_finite(vertices.at(col, row) + left + right + top + bottom)) {
    return std::nullopt;
  }
  if (std::abs(right.z - left.z) > options_.max_normal_depth_jump ||
      std::abs(bottom.z - top.z) > options_.max_normal_depth_jump) {
    return std::nullopt;
  }
  const Vec3f normal = cross(bottom - top, right - left);
  const float length = norm(normal);
  if (length <= 0.0F) {
    return std::nullopt;
  }
  return normal / length;
}

SurfacePyramid DepthProcessor<MemorySpace::kHost>::build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose) const {
  SurfacePyramid pyramid;
  pyramid.reserve(options_.levels);
  image_proc::DepthImage current = bilateral_filter(depth_image);
  for (unsigned int level = 0; level < options_.levels; ++level) {
    if (current.width() == 0U || current.height() == 0U) {
      break;
    }
    const CameraIntrinsics level_intrinsics = intrinsics.scaled(level);
    image_proc::Vector3fImage vertices =
        project_to_vertices(current, level_intrinsics, camera_pose);
    image_proc::Vector3fImage normals = compute_normals(vertices);
    // Downsample before `current` is moved into the pyramid; the last level
    // leaves `current` empty, ending the loop.
    image_proc::DepthImage next;
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
