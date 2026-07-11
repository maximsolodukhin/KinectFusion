#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <optional>
#include <type_traits>

namespace kinectfusion {
namespace {
constexpr float kColorIntegrationTruncationFraction = 0.5F;
// TSDF weight sums below this are treated as no coverage and cause the
// interpolation helpers to bail out.
constexpr float kMinimumTrilinearWeightSum = 1.0e-6F;

// Voxels are sampled at their center, half a cell from the lower corner.
constexpr float kVoxelCenterOffset = 0.5F;

// A voxel cube has eight corners; used both as the trilinear stencil size
// and for the `Corner` arrays returned by `trilinear_corners`.
constexpr std::size_t kTrilinearCornerCount = 8;

struct Corner {
  int x, y, z;
  float weight;
};

struct GridSample {
  // may be negative so vec3i
  Eigen::Vector3i base = Eigen::Vector3i::Zero();
  Eigen::Vector3f fraction = Eigen::Vector3f::Zero();
};

// One axis of a trilinear weight: the linear blend between the near voxel
// (offset 0, weight 1 - fraction) and the far voxel (offset 1, weight
// fraction).
[[nodiscard]] float axis_weight(float fraction, int offset) {
  return offset == 0 ? 1.0F - fraction : fraction;
}

// Product of the per-axis weights for one corner.
[[nodiscard]] float trilinear_weight(const Eigen::Vector3f& fraction,
                                     int offset_x, int offset_y, int offset_z) {
  return axis_weight(fraction.x(), offset_x) *
         axis_weight(fraction.y(), offset_y) *
         axis_weight(fraction.z(), offset_z);
}

[[nodiscard]] TsdfIntegrationOptions validated(TsdfIntegrationOptions options) {
  require(options.depth_scale > 0.0F, "Depth scale must be positive");
  require(options.observation_weight > 0.0F,
          "Observation weight must be positive");
  require(options.max_weight > 0.0F, "Max weight must be positive");
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Depth range is invalid");
  require(options.truncation_distance_scale >= 0.0F,
          "Truncation distance scale must be non-negative");
  return options;
}

[[nodiscard]] RaycastOptions validated(RaycastOptions options) {
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Raycast depth range is invalid");
  require(options.step_scale > 0.0F, "Raycast step scale must be positive");
  return options;
}

void validate_frame(const DepthFrame& frame) {
  require(frame.depth != nullptr, "Depth frame requires a depth image");
  require(frame.depth->width() > 0U && frame.depth->height() > 0U,
          "Depth frame depth image must be non-empty");
  require(frame.intrinsics.fx > 0.0F && frame.intrinsics.fy > 0.0F,
          "Depth frame intrinsics must have positive focal lengths");
  require(frame.world_to_camera.allFinite(),
          "Depth frame world_to_camera must be finite");
}

void validate_camera(const RaycastCamera& camera) {
  require(camera.intrinsics.fx > 0.0F && camera.intrinsics.fy > 0.0F,
          "Raycast intrinsics must have positive focal lengths");
  require(camera.width > 0U && camera.height > 0U,
          "Raycast dimensions must be positive");
  require(camera.camera_to_world.allFinite(),
          "Raycast camera_to_world must be finite");
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

template <typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] bool in_bounds(const Size3& resolution, Scalar x, Scalar y,
                             Scalar z) {
  return x >= Scalar{0} && y >= Scalar{0} && z >= Scalar{0} &&
         static_cast<std::size_t>(x) < resolution.x &&
         static_cast<std::size_t>(y) < resolution.y &&
         static_cast<std::size_t>(z) < resolution.z;
}

// Converts to grid coordinates before bounds before checking bounds.
[[nodiscard]] bool contains(const ConstHostVolumeView& volume,
                            const Eigen::Vector3f& point) {
  const Eigen::Vector3f local =
      (point - to_eigen(volume.origin)) / volume.voxel_size;
  // Scalar = float, no truncation
  return in_bounds(volume.resolution, local.x(), local.y(), local.z());
}

// World-space center of voxel (x, y, z).
[[nodiscard]] Eigen::Vector3f cell_center(const HostVolumeView& volume,
                                          std::size_t x, std::size_t y,
                                          std::size_t z) {
  const Eigen::Vector3f index{static_cast<float>(x), static_cast<float>(y),
                              static_cast<float>(z)};
  return to_eigen(volume.origin) +
         (index + Eigen::Vector3f::Constant(kVoxelCenterOffset)) *
             volume.voxel_size;
}

[[nodiscard]] GridSample grid_sample(const ConstHostVolumeView& volume,
                                     const Eigen::Vector3f& point) {
  const Eigen::Vector3f volume_origin = to_eigen(volume.origin);
  const Eigen::Vector3f grid = (point - volume_origin) / volume.voxel_size -
                               Eigen::Vector3f::Constant(kVoxelCenterOffset);
  const Eigen::Vector3f floored = grid.array().floor().matrix();
  return GridSample{.base = floored.cast<int>(), .fraction = grid - floored};
}

[[nodiscard]] std::array<Corner, kTrilinearCornerCount> trilinear_corners(
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

// Bounds-checked corner lookups for trilinear sampling; nullptr when the
// corner lies outside the volume.
[[nodiscard]] const Voxel* find_voxel(const ConstHostVolumeView& volume,
                                      const Corner& corner) {
  if (!in_bounds(volume.resolution, corner.x, corner.y, corner.z)) {
    return nullptr;
  }
  return &volume.voxel_at(static_cast<std::size_t>(corner.x),
                          static_cast<std::size_t>(corner.y),
                          static_cast<std::size_t>(corner.z));
}

[[nodiscard]] const ColorVoxel* find_color_voxel(
    const ConstHostVolumeView& volume, const Corner& corner) {
  if (!in_bounds(volume.resolution, corner.x, corner.y, corner.z)) {
    return nullptr;
  }
  return &volume.color_at(static_cast<std::size_t>(corner.x),
                          static_cast<std::size_t>(corner.y),
                          static_cast<std::size_t>(corner.z));
}

// Trilinear TSDF interpolation at `point`. `kRequireAll` returns nullopt if
// at least one of surrounding voxels is unobserved, while `kSkipMissing`
// drops missing/uninitialised corners and reweights the rest.
[[nodiscard]] std::optional<float> sample_tsdf(
    const ConstHostVolumeView& volume, const Eigen::Vector3f& point,
    CornerPolicy corner_policy) {
  if (!contains(volume, point)) {
    return std::nullopt;
  }

  const GridSample sample = grid_sample(volume, point);
  float accumulated = 0.0F;
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    const Voxel* voxel = find_voxel(volume, corner);
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

[[nodiscard]] std::optional<Vec3f> sample_color(
    const ConstHostVolumeView& volume, const Eigen::Vector3f& point) {
  const GridSample sample = grid_sample(volume, point);
  Vec3f accumulated{};
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    const ColorVoxel* voxel = find_color_voxel(volume, corner);
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

[[nodiscard]] std::optional<Vec3f> sample_normal(
    const ConstHostVolumeView& volume, const Eigen::Vector3f& point,
    CornerPolicy tsdf_corner_policy) {
  Eigen::Vector3f gradient;
  for (int axis = 0; axis < 3; ++axis) {
    // Central difference: one voxel step either way along this axis.
    const Eigen::Vector3f offset =
        Eigen::Vector3f::Unit(axis) * volume.voxel_size;
    const auto plus = sample_tsdf(volume, point + offset, tsdf_corner_policy);
    const auto minus = sample_tsdf(volume, point - offset, tsdf_corner_policy);
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

struct IntegrationContext {
  const DepthFrame* frame;
  const TsdfIntegrationOptions* options;
  Eigen::Matrix3f rotation;
  Eigen::Vector3f translation;
};

void integrate_voxel(const HostVolumeView& volume,
                     const IntegrationContext& context, std::size_t x,
                     std::size_t y, std::size_t z) {
  const DepthFrame& frame = *context.frame;
  const TsdfIntegrationOptions& options = *context.options;

  const Eigen::Vector3f camera_point =
      (context.rotation * cell_center(volume, x, y, z)) + context.translation;

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
      truncation_for(options, volume.truncation_distance, surface_depth);
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

  Voxel& voxel = volume.voxel_at(x, y, z);
  voxel = voxel.fused(tsdf, weight, options.max_weight);

  if (frame.color == nullptr) {
    return;
  }

  if (signed_distance <= truncation * kColorIntegrationTruncationFraction) {
    const ColorRgba rgba = rgba_from_pixel(frame.color->at(pixel->x, pixel->y));
    ColorVoxel& color_voxel = volume.color_at(x, y, z);
    color_voxel = color_voxel.fused(make_vec3f(rgba.x(), rgba.y(), rgba.z()),
                                    weight, options.max_weight);
  }
}

// Marches from `origin` along unit `direction`, returning the interpolated
// zero crossing where the ray first passes from front to back,
// sign change(+ to -). Gives up at max_depth or when leaving a surface from
// behind.
[[nodiscard]] std::optional<Eigen::Vector3f> find_zero_crossing(
    const ConstHostVolumeView& volume, const Eigen::Vector3f& origin,
    const Eigen::Vector3f& direction, const RaycastOptions& options) {
  struct Sample {
    float tsdf;
    Eigen::Vector3f point;
  };

  const float base_step = volume.voxel_size * options.step_scale;
  std::optional<Sample> previous;
  float ray_length = options.min_depth;
  while (ray_length <= options.max_depth) {
    const Eigen::Vector3f point = origin + ray_length * direction;
    const auto tsdf = sample_tsdf(volume, point, options.tsdf_corner_policy);
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
    auto step = std::max(
        base_step, *tsdf * volume.truncation_distance * options.step_scale);
    ray_length += *tsdf > 0.0F ? step : base_step;
  }
  return std::nullopt;
}

// Samples normal/colour at a zero crossing into the output maps.
void write_surface_sample(const ConstHostVolumeView& volume, SurfaceMaps& maps,
                          std::size_t col, std::size_t row,
                          const Eigen::Vector3f& surface,
                          CornerPolicy tsdf_corner_policy) {
  const auto normal = sample_normal(volume, surface, tsdf_corner_policy);
  if (!normal) {
    return;
  }
  maps.points.at(col, row) = from_eigen(surface);
  maps.normals.at(col, row) = *normal;
  if (const auto color = sample_color(volume, surface)) {
    maps.colors.at(col, row) = pixel_from_color(*color);
  }
}

}  // namespace

TsdfIntegrator::TsdfIntegrator(TsdfIntegrationOptions options)
    : options_(validated(options)) {}

void TsdfIntegrator::integrate(Volume& volume, const DepthFrame& frame) const {
  validate_frame(frame);
  const HostVolumeView view = volume.view();
  const IntegrationContext context{
      .frame = &frame,
      .options = &options_,
      .rotation = frame.world_to_camera.block<3, 3>(0, 0),
      .translation = frame.world_to_camera.block<3, 1>(0, 3)};
  for (std::size_t z = 0; z < view.resolution.z; ++z) {
    for (std::size_t y = 0; y < view.resolution.y; ++y) {
      for (std::size_t x = 0; x < view.resolution.x; ++x) {
        integrate_voxel(view, context, x, y, z);
      }
    }
  }
}

// TODO: change to static method/factory
// That's a very dangerous thing, if exception is thrown, the object will not be
// constructed, and the destructor will not be called. I'm unsure whether it
// leaks,
Raycaster::Raycaster(RaycastOptions options) : options_(validated(options)) {}

SurfaceMaps Raycaster::raycast(const Volume& volume,
                               const RaycastCamera& camera) const {
  validate_camera(camera);
  const ConstHostVolumeView view = volume.view();
  using Vec3fImg = image_proc::Vector3fImage;
  using ColorImg = image_proc::ColorImage;
  SurfaceMaps maps{
      .points = Vec3fImg{camera.width, camera.height, invalid_vec3f()},
      .normals = Vec3fImg{camera.width, camera.height, invalid_vec3f()},
      .colors = ColorImg{camera.width, camera.height}};

  const Eigen::Matrix3f rotation = camera.camera_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3f origin = camera.camera_to_world.block<3, 1>(0, 3);

  for (std::size_t row = 0; row < camera.height; ++row) {
    for (std::size_t col = 0; col < camera.width; ++col) {
      const Eigen::Vector3f direction =
          rotation * camera.intrinsics.back_project(
                         pixel_as_vector({.x = col, .y = row}), 1.0F);
      const float direction_norm = direction.norm();
      if (direction_norm <= 0.0F) {
        continue;
      }
      const auto surface = find_zero_crossing(
          view, origin, direction / direction_norm, options_);
      if (surface) {
        write_surface_sample(view, maps, col, row, *surface,
                             options_.tsdf_corner_policy);
      }
    }
  }
  return maps;
}

}  // namespace kinectfusion
