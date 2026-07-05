#include <kinectfusion/depth_processing.hpp>

#include "depth_processing_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace kinectfusion {
namespace {

using depth_processing_detail::downsample_factor;
using depth_processing_detail::usable_depth_meters;
using depth_processing_detail::validate_options;

std::optional<std::uint16_t> downsample_depth_block(
    const image_proc::DepthImage& depth_image, std::size_t col, std::size_t row,
    const DepthProcessingOptions& options) {
  std::uint32_t sum = 0;
  unsigned int count = 0;
  std::uint16_t minimum = 0xFFFFU;
  std::uint16_t maximum = 0;

  for (std::size_t dy = 0; dy < downsample_factor; ++dy) {
    for (std::size_t dx = 0; dx < downsample_factor; ++dx) {
      const std::uint16_t sample = depth_image.at(col * downsample_factor + dx,
                                                  row * downsample_factor + dy);
      if (sample == 0) {
        continue;
      }
      sum += sample;
      ++count;
      minimum = std::min(minimum, sample);
      maximum = std::max(maximum, sample);
    }
  }

  if (count == 0) {
    return std::nullopt;
  }
  if (depth_to_meters(maximum, options.depth_scale) -
          depth_to_meters(minimum, options.depth_scale) >
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
  const float nan = std::numeric_limits<float>::quiet_NaN();
  image_proc::Vector3fImage normals{
      vertices.width(), vertices.height(), Eigen::Vector3f::Constant(nan)};
  if (vertices.width() < 3U || vertices.height() < 3U) {
    return normals;
  }
  for (unsigned int row = 1U; row + 1U < vertices.height(); ++row) {
    for (unsigned int col = 1U; col + 1U < vertices.width(); ++col) {
      const Eigen::Vector3f& center = vertices.at(col, row);
      const Eigen::Vector3f& left = vertices.at(col - 1U, row);
      const Eigen::Vector3f& right = vertices.at(col + 1U, row);
      const Eigen::Vector3f& top = vertices.at(col, row - 1U);
      const Eigen::Vector3f& bottom = vertices.at(col, row + 1U);
      if (!center.allFinite() || !left.allFinite() || !right.allFinite() ||
          !top.allFinite() || !bottom.allFinite()) {
        continue;
      }

      if (std::abs(right.z() - left.z()) > options.max_normal_depth_jump ||
          std::abs(bottom.z() - top.z()) > options.max_normal_depth_jump) {
        continue;
      }
      const Eigen::Vector3f normal = (bottom - top).cross(right - left);
      const float norm = normal.norm();
      if (norm > 0.0F) {
        normals.at(col, row) = normal / norm;
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
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());
  const float spatial_scale =
      depth_processing_detail::bilateral_weight_scale(
          options.bilateral_spatial_sigma);
  const float depth_weight_scale =
      depth_processing_detail::bilateral_weight_scale(
          options.bilateral_depth_sigma);
  // Convert every sample to meters once instead of once per window visit;
  // NaN marks unusable samples.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> meters(depth_image.data().size());
  for (std::size_t i = 0; i < meters.size(); ++i) {
    const float depth = usable_depth_meters(depth_image.data()[i], options);
    meters[i] = depth > 0.0F ? depth : nan;
  }
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const float center_meters =
          meters[static_cast<std::size_t>(row) * depth_image.width() +
                 static_cast<std::size_t>(col)];
      if (std::isnan(center_meters)) {
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
          const float sample_meters =
              meters[static_cast<std::size_t>(y_index) * depth_image.width() +
                     x_index];
          if (std::isnan(sample_meters)) {
            continue;
          }
          const auto sample = depth_image.at(x_index, y_index);
          const float depth_difference = sample_meters - center_meters;
          const float spatial = static_cast<float>(dx * dx + dy * dy);
          const float weight =
              std::exp(spatial * spatial_scale +
                       depth_difference * depth_difference *
                           depth_weight_scale);
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
  const float nan = std::numeric_limits<float>::quiet_NaN();
  image_proc::Vector3fImage vertices{
      depth_image.width(), depth_image.height(),
      Eigen::Vector3f::Constant(nan)};
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
      vertices.at(col, row) =
          (camera_pose * camera_point.homogeneous()).head<3>();
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
