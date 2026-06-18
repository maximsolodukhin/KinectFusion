#include <kinectfusion/depth_processing.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace kinectfusion {
namespace {

// A raw depth sample is usable for filtering/back-projection when it is
// non-zero and falls inside the configured metric depth range.
[[nodiscard]] bool usable_depth(std::uint16_t raw, float& depth_meters) {
  if (raw == 0) {
    return false;
  }
  depth_meters = depth_to_meters(raw);
  return depth_meters >= kMinDepth && depth_meters <= kMaxDepth;
}

}  // namespace

image_proc::Image<Eigen::Vector3f> compute_normals_central_differences(
    const image_proc::Image<Eigen::Vector3f>& vertices) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  image_proc::Image<Eigen::Vector3f> normals{vertices.width(),
                                             vertices.height()};
  for (auto& normal : normals.data()) {
    normal = Eigen::Vector3f::Constant(nan);
  }
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

      if (std::abs(right.z() - left.z()) > 0.1F ||
          std::abs(bottom.z() - top.z()) > 0.1F) {
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
    const image_proc::DepthImage& depth_image) {
  image_proc::DepthImage filtered{depth_image.width(), depth_image.height()};
  const float spatial_scale =
      -0.5F / (kBilateralSpatialSigma * kBilateralSpatialSigma);
  const float depth_weight_scale =
      -0.5F / (kBilateralDepthSigma * kBilateralDepthSigma);
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const auto center = depth_image.at(static_cast<unsigned int>(col),
                                         static_cast<unsigned int>(row));
      float center_meters = 0.0F;
      if (!usable_depth(center, center_meters)) {
        continue;
      }
      float weighted_sum = 0.0F;
      float weight_sum = 0.0F;
      for (int dy = -kBilateralRadius; dy <= kBilateralRadius; ++dy) {
        const int y = row + dy;
        if (y < 0 || y >= height) {
          continue;
        }
        for (int dx = -kBilateralRadius; dx <= kBilateralRadius; ++dx) {
          const int x = col + dx;
          if (x < 0 || x >= width) {
            continue;
          }
          const auto sample = depth_image.at(static_cast<unsigned int>(x),
                                             static_cast<unsigned int>(y));
          float sample_meters = 0.0F;
          if (!usable_depth(sample, sample_meters)) {
            continue;
          }
          const float depth_difference = sample_meters - center_meters;
          const float spatial = static_cast<float>(dx * dx + dy * dy);
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
    const image_proc::DepthImage& depth_image) {
  const unsigned int width = depth_image.width() / 2U;
  const unsigned int height = depth_image.height() / 2U;
  image_proc::DepthImage downsampled{width, height};
  for (unsigned int row = 0; row < height; ++row) {
    for (unsigned int col = 0; col < width; ++col) {
      std::uint32_t sum = 0;
      unsigned int count = 0;
      std::uint16_t minimum = 0xFFFFU;
      std::uint16_t maximum = 0;
      for (unsigned int dy = 0; dy < 2U; ++dy) {
        for (unsigned int dx = 0; dx < 2U; ++dx) {
          const auto sample = depth_image.at(col * 2U + dx, row * 2U + dy);
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
        continue;
      }
      if (depth_to_meters(maximum) - depth_to_meters(minimum) >
          kMaxDownsampleDepthJump) {
        continue;
      }
      downsampled.at(col, row) = static_cast<std::uint16_t>(
          std::lround(static_cast<double>(sum) / static_cast<double>(count)));
    }
  }
  return downsampled;
}

image_proc::Image<Eigen::Vector3f> project_depth_to_vertices(
    const image_proc::DepthImage& depth_image, const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  image_proc::Image<Eigen::Vector3f> vertices{depth_image.width(),
                                              depth_image.height()};
  for (unsigned int row = 0; row < depth_image.height(); ++row) {
    for (unsigned int col = 0; col < depth_image.width(); ++col) {
      const auto raw = depth_image.at(col, row);
      const float depth = depth_to_meters(raw);
      if (raw == 0 || depth < kMinDepth || depth > kMaxDepth) {
        vertices.at(col, row) = Eigen::Vector3f::Constant(nan);
        continue;
      }
      const Eigen::Vector4f camera_point{
          (static_cast<float>(col) - intrinsics.cx) * depth / intrinsics.fx,
          (static_cast<float>(row) - intrinsics.cy) * depth / intrinsics.fy,
          depth, 1.0F};
      vertices.at(col, row) = (camera_pose * camera_point).head<3>();
    }
  }
  return vertices;
}

std::vector<DepthProcessingLevel> build_surface_pyramid(
    const image_proc::DepthImage& depth_image, const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose) {
  std::vector<DepthProcessingLevel> pyramid;
  image_proc::DepthImage current = bilateral_filter_depth(depth_image);
  for (unsigned int level = 0; level < kPyramidLevels; ++level) {
    const CameraIntrinsics level_intrinsics =
        scale_intrinsics(intrinsics, level);
    image_proc::Image<Eigen::Vector3f> vertices =
        project_depth_to_vertices(current, level_intrinsics, camera_pose);
    image_proc::Image<Eigen::Vector3f> normals =
        compute_normals_central_differences(vertices);
    pyramid.push_back(DepthProcessingLevel{
        .depth_image = current,
        .intrinsics = level_intrinsics,
        .maps = VertexNormalMaps{.vertices = std::move(vertices),
                                 .normals = std::move(normals)}});
    if (level + 1U < kPyramidLevels) {
      current = build_depth_pyramid_level(current);
    }
  }
  return pyramid;
}

}  // namespace kinectfusion
