#include <kinectfusion/depth_processing.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace kinectfusion {
namespace {

struct ImagePixel {
  unsigned int x{};
  unsigned int y{};
};

struct BilateralAccumulation {
  float weighted_depth_sum{};
  float weight_sum{};
};

void validate_options(const DepthProcessingOptions& options) {
  if (options.levels == 0) {
    throw std::invalid_argument("Depth pyramid must have at least one level");
  }
  if (options.depth_scale <= 0.0F) {
    throw std::invalid_argument("Depth scale must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Depth range is invalid");
  }
  if (options.max_normal_depth_jump < 0.0F) {
    throw std::invalid_argument("Normal depth jump threshold must be non-negative");
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

[[nodiscard]] bool usable_depth(std::uint16_t raw,
                                const DepthProcessingOptions& options,
                                float& depth_m) {
  if (raw == 0) {
    return false;
  }

  depth_m = depth_to_meters(raw, options.depth_scale);
  return depth_m >= options.min_depth && depth_m <= options.max_depth;
}

[[nodiscard]] BilateralAccumulation accumulate_bilateral_neighbors(
    const image_proc::DepthImage& depth_image,
    ImagePixel pixel,
    float center_depth,
    const DepthProcessingOptions& options) {
  const float spatial_scale =
      -0.5F / (options.bilateral_spatial_sigma *
               options.bilateral_spatial_sigma);
  const float depth_scale =
      -0.5F / (options.bilateral_depth_sigma *
               options.bilateral_depth_sigma);
  const auto radius = options.bilateral_radius;
  BilateralAccumulation accumulation;

  for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
    const auto sample_y = static_cast<int>(pixel.y) + offset_y;
    if (sample_y < 0 ||
        std::cmp_greater_equal(sample_y, depth_image.height())) {
      continue;
    }

    for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
      const auto sample_x = static_cast<int>(pixel.x) + offset_x;
      if (sample_x < 0 ||
          std::cmp_greater_equal(sample_x, depth_image.width())) {
        continue;
      }

      const auto sample_raw =
          depth_image.at(static_cast<unsigned int>(sample_x),
                         static_cast<unsigned int>(sample_y));
      float sample_depth = 0.0F;
      if (!usable_depth(sample_raw, options, sample_depth)) {
        continue;
      }

      const auto spatial_distance =
          static_cast<float>((offset_x * offset_x) + (offset_y * offset_y));
      const float depth_delta = sample_depth - center_depth;
      const float weight =
          std::exp((spatial_distance * spatial_scale) +
                   ((depth_delta * depth_delta) * depth_scale));
      accumulation.weighted_depth_sum += weight * sample_depth;
      accumulation.weight_sum += weight;
    }
  }

  return accumulation;
}

[[nodiscard]] std::uint16_t filtered_depth_at_pixel(
    const image_proc::DepthImage& depth_image,
    ImagePixel pixel,
    const DepthProcessingOptions& options) {
  float center_depth = 0.0F;
  if (!usable_depth(depth_image.at(pixel.x, pixel.y), options, center_depth)) {
    return 0;
  }

  const auto accumulation =
      accumulate_bilateral_neighbors(depth_image, pixel, center_depth, options);
  if (accumulation.weight_sum <= 0.0F) {
    return 0;
  }

  return static_cast<std::uint16_t>(
      std::lround((accumulation.weighted_depth_sum /
                   accumulation.weight_sum) *
                  options.depth_scale));
}

[[nodiscard]] std::vector<std::uint16_t> valid_downsample_candidates(
    const image_proc::DepthImage& previous,
    ImagePixel pixel) {
  std::vector<std::uint16_t> candidates;
  candidates.reserve(4);

  for (unsigned int offset_y = 0; offset_y < 2; ++offset_y) {
    for (unsigned int offset_x = 0; offset_x < 2; ++offset_x) {
      const auto raw =
          previous.at((pixel.x * 2U) + offset_x, (pixel.y * 2U) + offset_y);
      if (raw != 0) {
        candidates.push_back(raw);
      }
    }
  }

  return candidates;
}

[[nodiscard]] std::uint16_t downsample_depth_pixel(
    const image_proc::DepthImage& previous,
    ImagePixel pixel,
    const DepthProcessingOptions& options) {
  auto candidates = valid_downsample_candidates(previous, pixel);
  if (candidates.empty()) {
    return 0;
  }

  std::ranges::sort(candidates);
  const auto min_depth_m =
      depth_to_meters(candidates.front(), options.depth_scale);
  const auto max_depth_m =
      depth_to_meters(candidates.back(), options.depth_scale);
  if (max_depth_m - min_depth_m > options.max_normal_depth_jump) {
    return 0;
  }

  std::uint64_t sum = 0;
  for (const auto candidate : candidates) {
    sum += candidate;
  }
  return static_cast<std::uint16_t>(
      std::lround(static_cast<double>(sum) /
                  static_cast<double>(candidates.size())));
}

void compute_paper_forward_normals(
    const image_proc::Image<Eigen::Vector3f>& vertices,
    image_proc::Image<Eigen::Vector3f>& normals,
    const DepthProcessingOptions& options) {
  for (unsigned int y = 0; y + 1 < vertices.height(); ++y) {
    for (unsigned int x = 0; x + 1 < vertices.width(); ++x) {
      const auto& center = vertices.at(x, y);
      const auto& right = vertices.at(x + 1, y);
      const auto& down = vertices.at(x, y + 1);

      if (!center.allFinite() || !right.allFinite() || !down.allFinite()) {
        continue;
      }

      if (std::abs(right.z() - center.z()) > options.max_normal_depth_jump ||
          std::abs(down.z() - center.z()) > options.max_normal_depth_jump) {
        continue;
      }

      const auto normal = (down - center).cross(right - center);
      if (!normal.allFinite() || normal.norm() == 0.0F) {
        continue;
      }
      normals.at(x, y) = normal.normalized();
    }
  }
}

void compute_central_difference_normals(
    const image_proc::Image<Eigen::Vector3f>& vertices,
    image_proc::Image<Eigen::Vector3f>& normals,
    const DepthProcessingOptions& options) {
  for (unsigned int y = 1; y + 1 < vertices.height(); ++y) {
    for (unsigned int x = 1; x + 1 < vertices.width(); ++x) {
      const auto& center = vertices.at(x, y);
      const auto& left = vertices.at(x - 1, y);
      const auto& right = vertices.at(x + 1, y);
      const auto& up = vertices.at(x, y - 1);
      const auto& down = vertices.at(x, y + 1);

      if (!center.allFinite() || !left.allFinite() || !right.allFinite() ||
          !up.allFinite() || !down.allFinite()) {
        continue;
      }

      if (std::abs(right.z() - left.z()) > options.max_normal_depth_jump ||
          std::abs(down.z() - up.z()) > options.max_normal_depth_jump) {
        continue;
      }

      const auto normal = (down - up).cross(right - left);
      if (!normal.allFinite() || normal.norm() == 0.0F) {
        continue;
      }
      normals.at(x, y) = normal.normalized();
    }
  }
}

}  // namespace

image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);

  if (!options.bilateral_filter || options.bilateral_radius == 0) {
    return depth_image;
  }

  image_proc::DepthImage filtered{depth_image.width(), depth_image.height()};
  std::ranges::fill(filtered.data(), std::uint16_t{});

  for (unsigned int y = 0; y < depth_image.height(); ++y) {
    for (unsigned int x = 0; x < depth_image.width(); ++x) {
      filtered.at(x, y) =
          filtered_depth_at_pixel(depth_image, ImagePixel{.x = x, .y = y},
                                  options);
    }
  }

  return filtered;
}

DepthPyramid build_depth_pyramid(const image_proc::DepthImage& depth_image,
                                 const DepthProcessingOptions& options) {
  validate_options(options);

  DepthPyramid pyramid;
  pyramid.reserve(options.levels);
  pyramid.push_back(depth_image);

  for (unsigned int level = 1; level < options.levels; ++level) {
    const auto& previous = pyramid.back();
    const unsigned int width = previous.width() / 2;
    const unsigned int height = previous.height() / 2;
    if (width == 0 || height == 0) {
      break;
    }

    image_proc::DepthImage current{width, height};
    for (unsigned int y = 0; y < height; ++y) {
      for (unsigned int x = 0; x < width; ++x) {
        current.at(x, y) =
            downsample_depth_pixel(previous, ImagePixel{.x = x, .y = y},
                                   options);
      }
    }

    pyramid.push_back(std::move(current));
  }

  return pyramid;
}

CameraIntrinsics scale_intrinsics(const CameraIntrinsics& base,
                                  unsigned int level) {
  const float scale = 1.0F / static_cast<float>(1U << level);
  return CameraIntrinsics{
      .fx = base.fx * scale,
      .fy = base.fy * scale,
      .cx = base.cx * scale,
      .cy = base.cy * scale};
}

image_proc::Image<Eigen::Vector3f> back_project_depth(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_to_world,
    const DepthProcessingOptions& options) {
  validate_options(options);

  image_proc::Image<Eigen::Vector3f> vertices{depth_image.width(),
                                       depth_image.height()};
  std::ranges::fill(vertices.data(), invalid_vector());

  for (unsigned int y = 0; y < depth_image.height(); ++y) {
    for (unsigned int x = 0; x < depth_image.width(); ++x) {
      const auto raw = depth_image.at(x, y);
      if (raw == 0) {
        continue;
      }

      const auto depth_m = depth_to_meters(raw, options.depth_scale);
      if (depth_m < options.min_depth || depth_m > options.max_depth) {
        continue;
      }

      const auto camera_point = Eigen::Vector4f{
          (static_cast<float>(x) - intrinsics.cx) * depth_m / intrinsics.fx,
          (static_cast<float>(y) - intrinsics.cy) * depth_m / intrinsics.fy,
          depth_m,
          1
      };
      vertices.at(x, y) = (camera_to_world * camera_point).head<3>();
    }
  }

  return vertices;
}

image_proc::Image<Eigen::Vector3f> compute_normals(
    const image_proc::Image<Eigen::Vector3f>& vertices,
    const DepthProcessingOptions& options) {
  validate_options(options);

  image_proc::Image<Eigen::Vector3f> normals{vertices.width(), vertices.height()};
  std::ranges::fill(normals.data(), invalid_vector());

  if (vertices.width() < 3 || vertices.height() < 3) {
    return normals;
  }

  if (options.normal_computation == NormalComputation::paper_forward) {
    compute_paper_forward_normals(vertices, normals, options);
    return normals;
  }

  compute_central_difference_normals(vertices, normals, options);

  return normals;
}

VertexNormalMaps build_vertex_normal_maps(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_to_world,
    const DepthProcessingOptions& options) {
  auto vertices =
      back_project_depth(depth_image, intrinsics, camera_to_world, options);
  auto normals = compute_normals(vertices, options);
  return VertexNormalMaps{.vertices = std::move(vertices),
                          .normals = std::move(normals)};
}

SurfacePyramid build_surface_pyramid(const image_proc::DepthImage& depth_image,
                                     const CameraIntrinsics& intrinsics,
                                     const Eigen::Matrix4f& camera_to_world,
                                     const DepthProcessingOptions& options) {
  validate_options(options);

  SurfacePyramid surface_pyramid;
  const auto tracking_depth = bilateral_filter_depth(depth_image, options);
  const auto depth_pyramid = build_depth_pyramid(tracking_depth, options);
  surface_pyramid.reserve(depth_pyramid.size());

  for (std::size_t level = 0; level < depth_pyramid.size(); ++level) {
    const auto level_intrinsics =
        scale_intrinsics(intrinsics, static_cast<unsigned int>(level));
    auto maps = build_vertex_normal_maps(depth_pyramid.at(level), level_intrinsics,
                                         camera_to_world, options);
    surface_pyramid.push_back(
        DepthProcessingLevel{.depth = depth_pyramid.at(level),
                             .intrinsics = level_intrinsics,
                             .maps = std::move(maps)});
  }

  return surface_pyramid;
}

}  // namespace kinectfusion
