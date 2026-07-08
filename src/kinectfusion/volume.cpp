#include <algorithm>
#include <array>
#include <cmath>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/volume.hpp>
#include <optional>

namespace kinectfusion {
namespace {
constexpr float kColorIntegrationTruncationFraction = 0.5F;
// TSDF weight sums below this are treated as no coverage and cause the
// interpolation helpers to bail out.
constexpr float kMinimumTrilinearWeightSum = 1.0e-6F;

// Voxels are sampled at their center, half a cell from the lower corner.
constexpr float kVoxelCenterOffset = 0.5F;

// One axis of a trilinear weight: the linear blend between the near voxel
// (offset 0, weight 1 - fraction) and the far voxel (offset 1, weight
// fraction).
[[nodiscard]] float axis_weight(float fraction, int offset) {
  return offset == 0 ? 1.0F - fraction : fraction;
}

void validate_options(const TsdfIntegrationOptions& options) {
  require(options.depth_scale > 0.0F, "Depth scale must be positive");
  require(options.observation_weight > 0.0F,
          "Observation weight must be positive");
  require(options.max_weight > 0.0F, "Max weight must be positive");
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Depth range is invalid");
  require(options.truncation_distance_scale >= 0.0F,
          "Truncation distance scale must be non-negative");
}

void validate_options(const RaycastOptions& options) {
  require(options.intrinsics.fx > 0.0F && options.intrinsics.fy > 0.0F,
          "Raycast intrinsics must have positive focal lengths");
  require(options.width > 0U && options.height > 0U,
          "Raycast dimensions must be positive");
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Raycast depth range is invalid");
  require(options.step_scale > 0.0F, "Raycast step scale must be positive");
}

// Bounds-checked integer pixel coordinate for image lookups.
struct Pixel {
  std::size_t x;
  std::size_t y;
};

[[nodiscard]] Eigen::Vector2f pixel_as_vector(const Pixel& pixel) {
  return Eigen::Vector2f{static_cast<float>(pixel.x),
                         static_cast<float>(pixel.y)};
}

// Project a point in a camera-space into pixel coordinates, returning nullopt
// if the point is behind the camera or outside the image bounds.
[[nodiscard]] std::optional<Pixel> project_to_pixel(
    const CameraIntrinsics& intrinsics, const Eigen::Vector3f& camera_point,
    std::size_t width, std::size_t height) {
  if (camera_point.z() <= 0.0F) {
    return std::nullopt;
  }
  const Eigen::Vector2f pixel = intrinsics.project(camera_point);
  const auto rounded_x = std::lround(pixel.x());
  const auto rounded_y = std::lround(pixel.y());
  if (rounded_x < 0 || rounded_y < 0) {
    return std::nullopt;
  }

  const auto col = static_cast<std::size_t>(rounded_x);
  const auto row = static_cast<std::size_t>(rounded_y);
  if (col >= width || row >= height) {
    return std::nullopt;
  }

  return Pixel{.x = col, .y = row};
}

[[nodiscard]] float projective_sdf(const TsdfIntegrationOptions& options,
                                   const Eigen::Vector3f& camera_point,
                                   float lambda, float surface_depth) {
  if (options.projective_distance) {
    return surface_depth - (camera_point.norm() / lambda);
  }
  return surface_depth - camera_point.z();
}

[[nodiscard]] float truncation_for(const TsdfIntegrationOptions& options,
                                   float base, float surface_depth) {
  if (options.distance_scaled_truncation) {
    return base + (options.truncation_distance_scale * surface_depth);
  }
  return base;
}

// cos(theta)/depth (paper 3.3)
[[nodiscard]] float observation_weight(const TsdfIntegrationOptions& options,
                                       const image_proc::Vector3fImage* normals,
                                       const Eigen::Vector3f& ray,
                                       float surface_depth,
                                       const Pixel& pixel) {
  float weight = options.observation_weight;
  if (options.view_angle_weighting && normals != nullptr) {
    const Vec3f& normal_sample = normals->at(pixel.x, pixel.y);
    if (all_finite(normal_sample)) {
      const Eigen::Vector3f normal = to_eigen(normal_sample);
      // View direction is the (quantised) pixel ray, matching the SDF above.
      const Eigen::Vector3f view = -ray.normalized();
      weight *= std::max(0.0F, normal.normalized().dot(view)) / surface_depth;
    }
  }
  return weight;
}

// Product of the per-axis weights for one corner.
[[nodiscard]] float trilinear_weight(const Eigen::Vector3f& fraction,
                                     int offset_x, int offset_y, int offset_z) {
  return axis_weight(fraction.x(), offset_x) *
         axis_weight(fraction.y(), offset_y) *
         axis_weight(fraction.z(), offset_z);
}

}  // namespace

void Volume::integrate_depth_image(const DepthFrame& frame,
                                   const TsdfIntegrationOptions& options) {
  require(frame.depth != nullptr, "Depth frame requires a depth image");
  validate_options(options);
  const IntegrationContext context{
      .frame = &frame,
      .options = &options,
      .rotation = frame.world_to_camera.block<3, 3>(0, 0),
      .translation = frame.world_to_camera.block<3, 1>(0, 3)};
  for (std::size_t z = 0; z < resolution_.z(); ++z) {
    for (std::size_t y = 0; y < resolution_.y(); ++y) {
      for (std::size_t x = 0; x < resolution_.x(); ++x) {
        integrate_voxel(context, x, y, z);
      }
    }
  }
}

// World-space center of voxel (x, y, z).
Eigen::Vector3f Volume::cell_center(std::size_t x, std::size_t y,
                                    std::size_t z) const {
  const Eigen::Vector3f index{static_cast<float>(x), static_cast<float>(y),
                              static_cast<float>(z)};
  return to_eigen(origin_) +
         (index + Eigen::Vector3f::Constant(kVoxelCenterOffset)) * voxel_size_;
}

void Volume::integrate_voxel(const IntegrationContext& context, std::size_t x,
                             std::size_t y, std::size_t z) {
  const DepthFrame& frame = *context.frame;
  const TsdfIntegrationOptions& options = *context.options;

  const Eigen::Vector3f camera_point =
      (context.rotation * cell_center(x, y, z)) + context.translation;
  const auto pixel =
      project_to_pixel(frame.intrinsics, camera_point, frame.depth->width(),
                       frame.depth->height());
  if (!pixel) {
    return;
  }
  const auto measured =
      depth_in_range(frame.depth->at(pixel->x, pixel->y), options.depth_scale,
                     options.min_depth, options.max_depth);
  if (!measured) {
    return;
  }
  const float surface_depth = *measured;
  const Eigen::Vector3f ray =
      frame.intrinsics.back_project(pixel_as_vector(*pixel), 1.0F);
  const float lambda = ray.norm();
  if (!ray.allFinite() || lambda == 0.0F) {
    return;
  }

  const float truncation =
      truncation_for(options, truncation_distance_, surface_depth);
  const float signed_distance =
      projective_sdf(options, camera_point, lambda, surface_depth);
  if (signed_distance < -truncation) {
    return;
  }
  const float tsdf = std::clamp(signed_distance / truncation, -1.0F, 1.0F);

  const float weight =
      observation_weight(options, frame.normals, ray, surface_depth, *pixel);
  if (weight <= 0.0F) {
    return;
  }
  Voxel& voxel = at(x, y, z);
  voxel = voxel.fused(tsdf, weight, options.max_weight);

  if (frame.color != nullptr &&
      signed_distance <= truncation * kColorIntegrationTruncationFraction) {
    const ColorRgba rgba = rgba_from_pixel(frame.color->at(pixel->x, pixel->y));
    ColorVoxel& color_voxel = color_at(x, y, z);
    color_voxel = color_voxel.fused(make_vec3f(rgba.x(), rgba.y(), rgba.z()),
                                    weight, options.max_weight);
  }
}

SurfaceMaps Volume::raycast(const CameraIntrinsics& intrinsics,
                            std::size_t width, std::size_t height,
                            const Eigen::Matrix4f& camera_to_world) const {
  return raycast(RaycastOptions{.intrinsics = intrinsics,
                                .width = width,
                                .height = height,
                                .camera_to_world = camera_to_world});
}

SurfaceMaps Volume::raycast(const RaycastOptions& options) const {
  validate_options(options);
  using Vec3fImg = image_proc::Vector3fImage;
  using ColorImg = image_proc::ColorImage;
  SurfaceMaps maps{
      .points = Vec3fImg{options.width, options.height, invalid_vec3f()},
      .normals = Vec3fImg{options.width, options.height, invalid_vec3f()},
      .colors = ColorImg{options.width, options.height}};

  const Eigen::Matrix3f rotation = options.camera_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3f origin = options.camera_to_world.block<3, 1>(0, 3);

  for (std::size_t row = 0; row < options.height; ++row) {
    for (std::size_t col = 0; col < options.width; ++col) {
      const Eigen::Vector3f direction =
          rotation * options.intrinsics.back_project(
                         pixel_as_vector({.x = col, .y = row}), 1.0F);
      const float direction_norm = direction.norm();
      if (direction_norm <= 0.0F) {
        continue;
      }
      const auto surface =
          find_zero_crossing(origin, direction / direction_norm, options);
      if (surface) {
        write_surface_sample(maps, col, row, *surface,
                             options.tsdf_corner_policy);
      }
    }
  }
  return maps;
}

// Marches from `origin` along unit `direction`, returning the interpolated
// zero crossing where the ray first passes from front to back,
// sign change(+ to -). Gives up at max_depth or when leaving a surface from
// behind.
std::optional<Eigen::Vector3f> Volume::find_zero_crossing(
    const Eigen::Vector3f& origin, const Eigen::Vector3f& direction,
    const RaycastOptions& options) const {
  struct Sample {
    float tsdf;
    Eigen::Vector3f point;
  };

  const float base_step = voxel_size_ * options.step_scale;
  std::optional<Sample> previous;
  float ray_length = options.min_depth;
  while (ray_length <= options.max_depth) {
    const Eigen::Vector3f point = origin + ray_length * direction;
    const auto tsdf = sample_tsdf(point, options.tsdf_corner_policy);
    if (!tsdf) {
      previous.reset();
      ray_length += base_step;
      continue;
    }
    if (previous && previous->tsdf > 0.0F && *tsdf <= 0.0F) {
      // Interpolate the crossing between the last two samples.
      const float gap = previous->tsdf - *tsdf;
      const float alpha = gap != 0.0F ? previous->tsdf / gap : 0.0F;
      return previous->point + alpha * (point - previous->point);
    }
    if (previous && previous->tsdf < 0.0F && *tsdf > 0.0F) {
      return std::nullopt;
    }
    previous = Sample{.tsdf = *tsdf, .point = point};
    auto step =
        std::max(base_step, *tsdf * truncation_distance_ * options.step_scale);
    ray_length += *tsdf > 0.0F ? step : base_step;
  }
  return std::nullopt;
}

void Volume::write_surface_sample(SurfaceMaps& maps, std::size_t col,
                                  std::size_t row,
                                  const Eigen::Vector3f& surface,
                                  CornerPolicy tsdf_corner_policy) const {
  const auto normal = sample_normal(surface, tsdf_corner_policy);
  if (!normal) {
    return;
  }
  maps.points.at(col, row) = from_eigen(surface);
  maps.normals.at(col, row) = *normal;
  if (const auto color = sample_color(surface)) {
    maps.colors.at(col, row) = pixel_from_color(*color);
  }
}

Volume::GridSample Volume::grid_sample(const Eigen::Vector3f& point) const {
  const Eigen::Vector3f volume_origin = to_eigen(origin_);
  const Eigen::Vector3f grid = (point - volume_origin) / voxel_size_ -
                               Eigen::Vector3f::Constant(kVoxelCenterOffset);
  const Eigen::Vector3f floored = grid.array().floor().matrix();
  return GridSample{.base = floored.cast<int>(), .fraction = grid - floored};
}

std::array<Corner, kTrilinearCornerCount> Volume::trilinear_corners(
    const GridSample& sample) {
  std::array<Corner, kTrilinearCornerCount> out{};
  std::size_t corner_idx = 0;
  for (int offset_z = 0; offset_z <= 1; ++offset_z) {
    for (int offset_y = 0; offset_y <= 1; ++offset_y) {
      for (int offset_x = 0; offset_x <= 1; ++offset_x) {
        auto weight =
            trilinear_weight(sample.fraction, offset_x, offset_y, offset_z);
        out.at(corner_idx++) = {.x = sample.base.x() + offset_x,
                                .y = sample.base.y() + offset_y,
                                .z = sample.base.z() + offset_z,
                                .weight = weight};
      }
    }
  }
  return out;
}

std::optional<float> Volume::sample_tsdf(const Eigen::Vector3f& point,
                                         CornerPolicy corner_policy) const {
  if (!contains(point)) {
    return std::nullopt;
  }

  const GridSample sample = grid_sample(point);
  float accumulated = 0.0F;
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    const Voxel* voxel = find_voxel(corner);
    if (voxel != nullptr && voxel->weight > 0.0F &&
        std::isfinite(voxel->distance)) {
      accumulated += corner.weight * voxel->distance;
      weight_sum += corner.weight;
      continue;
    }
    // Corner is out of bounds or unobserved. `kRequireAll` fails the sample;
    // otherwise the corner is skipped and the remaining ones reweight.
    if (corner_policy == CornerPolicy::kRequireAll) {
      return std::nullopt;
    }
  }
  if (weight_sum <= kMinimumTrilinearWeightSum) {
    return std::nullopt;
  }

  const float distance = accumulated / weight_sum;
  return std::isfinite(distance) ? std::optional<float>{distance}
                                 : std::nullopt;
}

std::optional<Vec3f> Volume::sample_color(const Eigen::Vector3f& point) const {
  const GridSample sample = grid_sample(point);
  Vec3f accumulated{};
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    const ColorVoxel* voxel = find_color_voxel(corner);
    if (voxel == nullptr || voxel->weight <= 0.0F) {
      continue;
    }
    accumulated += corner.weight * voxel->color;
    weight_sum += corner.weight;
  }
  if (weight_sum <= kMinimumTrilinearWeightSum) {
    return std::nullopt;
  }
  return accumulated / weight_sum;
}

std::optional<Vec3f> Volume::sample_normal(
    const Eigen::Vector3f& point, CornerPolicy tsdf_corner_policy) const {
  Eigen::Vector3f gradient;
  for (int axis = 0; axis < 3; ++axis) {
    // Central difference: one voxel step either way along this axis.
    const Eigen::Vector3f offset = Eigen::Vector3f::Unit(axis) * voxel_size_;
    const auto plus = sample_tsdf(point + offset, tsdf_corner_policy);
    const auto minus = sample_tsdf(point - offset, tsdf_corner_policy);
    if (!plus || !minus) {
      return std::nullopt;
    }
    gradient[axis] = *plus - *minus;
  }

  const float norm = gradient.norm();
  if (norm <= 0.0F) {
    return std::nullopt;
  }
  return from_eigen(gradient / norm);
}

}  // namespace kinectfusion
