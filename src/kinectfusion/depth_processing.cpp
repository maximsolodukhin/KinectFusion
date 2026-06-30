#include <kinectfusion/depth_processing.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace kinectfusion {
namespace {

// Linear resolution reduction per pyramid level: each level halves width and
// height, downsampling over a downsample_factor x downsample_factor block.
constexpr std::size_t downsample_factor = 2U;

constexpr std::uint16_t maximum_depth_value = 0xFFFFU;
constexpr std::uint16_t minimum_depth_value = 0U;

// A raw depth sample is usable for filtering/back-projection when it is
// non-zero and falls inside the configured metric depth range.
void validate_options(const DepthProcessingOptions& options) {
  if (options.levels == 0U) {
    throw std::invalid_argument("Depth pyramid must have at least one level");
  }
  if (options.depth_scale <= 0.0F) {
    throw std::invalid_argument("Depth scale must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Depth range is invalid");
  }
  if (options.max_normal_depth_jump < 0.0F) {
    throw std::invalid_argument(
        "Normal depth jump threshold must be non-negative");
  }
  if (options.max_downsample_depth_jump < 0.0F) {
    throw std::invalid_argument(
        "Downsample depth jump threshold must be non-negative");
  }
  if (options.bilateral_radius < 0) {
    throw std::invalid_argument("Bilateral filter radius must be non-negative");
  }
  if (options.bilateral_spatial_sigma <= 0.0F) {
    throw std::invalid_argument("Bilateral spatial sigma must be positive");
  }
  if (options.bilateral_depth_sigma <= 0.0F) {
    throw std::invalid_argument("Bilateral depth sigma must be positive");
  }
}

[[nodiscard]] inline bool usable_depth(std::uint16_t raw,
                                       const DepthProcessingOptions& options,
                                       float& depth_meters) {
  if (raw == 0) {
    return false;
  }
  depth_meters = depth_to_meters(raw, options.depth_scale);
  return depth_meters >= options.min_depth && depth_meters <= options.max_depth;
}

std::optional<std::uint16_t> downsample_depth_block(
    const image_proc::DepthImage& depth_image, std::size_t col, std::size_t row,
    const DepthProcessingOptions& options) {
  std::uint32_t sum = 0;
  unsigned int count = 0;
  std::uint16_t minimum_observed_depth = maximum_depth_value;
  std::uint16_t maximum_observed_depth = minimum_depth_value;

  for (std::size_t dy = 0; dy < downsample_factor; ++dy) {
    for (std::size_t dx = 0; dx < downsample_factor; ++dx) {
      const std::uint16_t sample = depth_image.at(col * downsample_factor + dx,
                                                  row * downsample_factor + dy);
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
  if (depth_to_meters(maximum_observed_depth, options.depth_scale) -
          depth_to_meters(minimum_observed_depth, options.depth_scale) >
      options.max_downsample_depth_jump) {
    return std::nullopt;
  }

  return static_cast<std::uint16_t>((sum + count / 2U) / count);
}
 
}  // namespace

image_proc::Vector3fImage compute_normals_central_differences(
    const image_proc::Vector3fImage& vertices,
    const DepthProcessingOptions& options) {
  validate_options(options);
  image_proc::Vector3fImage normals{
      vertices.width(), vertices.height(), invalid_vector()};
  if (vertices.width() < 3U || vertices.height() < 3U) {
    return normals;
  }
  for (unsigned int row = 1U; row + 1U < vertices.height(); ++row) {
    for (unsigned int col = 1U; col + 1U < vertices.width(); ++col) {
      const Vec3f& center = vertices.at(col, row);
      const Vec3f& left = vertices.at(col - 1U, row);
      const Vec3f& right = vertices.at(col + 1U, row);
      const Vec3f& top = vertices.at(col, row - 1U);
      const Vec3f& bottom = vertices.at(col, row + 1U);
      if (!all_finite(center) || !all_finite(left) || !all_finite(right) ||
          !all_finite(top) || !all_finite(bottom)) {
        continue;
      }

      if (std::abs(right.z - left.z) > options.max_normal_depth_jump ||
          std::abs(bottom.z - top.z) > options.max_normal_depth_jump) {
        continue;
      }
      const Vec3f normal = cross(bottom - top, right - left);
      const float normal_length = norm(normal);
      if (normal_length > 0.0F) {
        normals.at(col, row) = normal / normal_length;
      }
    }
  }
  return normals;
}

image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);
  if (!options.bilateral_filter || options.bilateral_radius == 0) {
    return depth_image;
  }

  image_proc::DepthImage filtered{depth_image.width(), depth_image.height()};
  const float spatial_sigma2 =
      options.bilateral_spatial_sigma * options.bilateral_spatial_sigma;
  const float depth_sigma2 =
      options.bilateral_depth_sigma * options.bilateral_depth_sigma;
  const float spatial_scale = -0.5F / spatial_sigma2;
  const float depth_weight_scale = -0.5F / depth_sigma2;
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const auto center = depth_image.at(static_cast<unsigned int>(col),
                                         static_cast<unsigned int>(row));
      float center_meters = 0.0F;
      if (!usable_depth(center, options, center_meters)) {
        continue;
      }
      float weighted_sum = 0.0F;
      float weight_sum = 0.0F;
      for (int dy = -options.bilateral_radius; dy <= options.bilateral_radius;
           ++dy) {
        const int y = row + dy;
        if (y < 0 || y >= height) {
          continue;
        }
        const auto y_index = static_cast<unsigned int>(y);

        for (int dx = -options.bilateral_radius;
             dx <= options.bilateral_radius; ++dx) {
          const int x = col + dx;
          if (x < 0 || x >= width) {
            continue;
          }
          const auto x_index = static_cast<unsigned int>(x);
          const auto sample = depth_image.at(x_index, y_index);
          float sample_meters = 0.0F;
          if (!usable_depth(sample, options, sample_meters)) {
            continue;
          }
          const float depth_difference = sample_meters - center_meters;
          const auto spatial = static_cast<float>(dx * dx + dy * dy);
          const float weight =
              std::exp(spatial * spatial_scale +
                       depth_difference * depth_difference * depth_weight_scale);
          weighted_sum += weight * static_cast<float>(sample);
          weight_sum += weight;
        }
      }
      if (weight_sum > 0.0F) {
        filtered.at(static_cast<unsigned int>(col),
                    static_cast<unsigned int>(row)) =
            static_cast<std::uint16_t>(std::lround(weighted_sum / weight_sum));
      }
    }
  }
  return filtered;
}

image_proc::DepthImage build_depth_pyramid_level(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);

  const std::size_t width = depth_image.width() / downsample_factor;
  const std::size_t height = depth_image.height() / downsample_factor;
  image_proc::DepthImage downsampled{width, height};

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      if (const auto value =
              downsample_depth_block(depth_image, col, row, options)) {
        downsampled.at(col, row) = *value;
      }
    }
  }
  return downsampled;
}

image_proc::Vector3fImage project_depth_to_vertices(
    const image_proc::DepthImage& depth_image, const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options) {
  validate_options(options);
  image_proc::Vector3fImage vertices{
      depth_image.width(), depth_image.height(), invalid_vector()};
  for (std::size_t row = 0; row < depth_image.height(); ++row) {
    for (std::size_t col = 0; col < depth_image.width(); ++col) {
      const auto raw = depth_image.at(col, row);
      const float depth = depth_to_meters(raw, options.depth_scale);
      if (raw == 0 || depth < options.min_depth || depth > options.max_depth) {
        continue;
      }
      const Eigen::Vector3f camera_point = intrinsics.back_project(
          {static_cast<float>(col), static_cast<float>(row)}, depth);
      // .homogeneous() makes the camera_point (x,y,z,1) to multiply with the camera pose
      const Eigen::Vector3f world_point =
          (camera_pose * camera_point.homogeneous()).head<3>();
      vertices.at(col, row) = from_eigen(world_point);
    }
  }
  return vertices;
}

std::vector<DepthProcessingLevel> build_surface_pyramid(
    const image_proc::DepthImage& depth_image, const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options) {
  // Each helper below validates `options` itself; validating here too would
  // just repeat the same checks once per pyramid level.
  std::vector<DepthProcessingLevel> pyramid;
  image_proc::DepthImage current = bilateral_filter_depth(depth_image, options);
  for (unsigned int level = 0; level < options.levels; ++level) {
    if (current.width() == 0U || current.height() == 0U) {
      break;
    }
    const CameraIntrinsics level_intrinsics = intrinsics.scaled(level);
    image_proc::Vector3fImage vertices =
        project_depth_to_vertices(current, level_intrinsics, camera_pose,
                                  options);
    image_proc::Vector3fImage normals =
        compute_normals_central_differences(vertices, options);
    if (level + 1U < options.levels) {
      pyramid.emplace_back(current, level_intrinsics, std::move(vertices),
                         std::move(normals));
      current = build_depth_pyramid_level(current, options);
    } else {
      pyramid.emplace_back(std::move(current), level_intrinsics,
                             std::move(vertices), std::move(normals));
    }
  }
  return pyramid;
}

}  // namespace kinectfusion
