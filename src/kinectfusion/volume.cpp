#include <algorithm>
#include <array>
#include <cmath>
#include <kinectfusion/volume.hpp>
#include <stdexcept>

namespace kinectfusion {
namespace {

// Colour is only fused near the surface, within this fraction of the
// truncation distance in front of the observed depth.
constexpr float color_integration_truncation_fraction = 0.5F;

// TSDF weight sums below this are treated as no coverage and cause the
// interpolation helpers to bail out.
constexpr float minimum_trilinear_weight_sum = 1.0e-6F;

void validate_options(const TsdfIntegrationOptions& options) {
  if (options.depth_scale <= 0.0F) {
    throw std::invalid_argument("Depth scale must be positive");
  }
  if (options.observation_weight <= 0.0F) {
    throw std::invalid_argument("Observation weight must be positive");
  }
  if (options.max_weight <= 0.0F) {
    throw std::invalid_argument("Max weight must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Depth range is invalid");
  }
  if (options.truncation_distance_scale < 0.0F) {
    throw std::invalid_argument(
        "Truncation distance scale must be non-negative");
  }
}

void validate_options(const RaycastOptions& options) {
  if (options.intrinsics.fx <= 0.0F || options.intrinsics.fy <= 0.0F) {
    throw std::invalid_argument(
        "Raycast intrinsics must have positive focal lengths");
  }
  if (options.width == 0U || options.height == 0U) {
    throw std::invalid_argument("Raycast dimensions must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Raycast depth range is invalid");
  }
  if (options.step_scale <= 0.0F) {
    throw std::invalid_argument("Raycast step scale must be positive");
  }
}

}  // namespace

void Volume::integrate_depth_image(const image_proc::DepthImage& depth_image,
                                   const CameraIntrinsics& intrinsics,
                                   const Eigen::Matrix4f& world_to_camera,
                                   const TsdfIntegrationOptions& options,
                                   const image_proc::ColorImage* color_image,
                                   const image_proc::Vector3fImage* normals) {
  validate_options(options);
  const Eigen::Matrix3f rotation = world_to_camera.block<3, 3>(0, 0);
  const Eigen::Vector3f translation = world_to_camera.block<3, 1>(0, 3);
  const Eigen::Vector3f volume_origin = to_eigen(origin_);
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());

  for (size_t z = 0; z < resolution_.z(); ++z) {
    for (size_t y = 0; y < resolution_.y(); ++y) {
      for (size_t x = 0; x < resolution_.x(); ++x) {
        const Eigen::Vector3f center =
            volume_origin + Eigen::Vector3f{static_cast<float>(x) + 0.5F,
                                            static_cast<float>(y) + 0.5F,
                                            static_cast<float>(z) + 0.5F} *
                                voxel_size_;
        const Eigen::Vector3f camera_point = rotation * center + translation;
        if (camera_point.z() <= 0.0F) {
          continue;
        }
        const Eigen::Vector2f pixel = intrinsics.project(camera_point);
        const int pixel_x = static_cast<int>(std::lround(pixel.x()));
        const int pixel_y = static_cast<int>(std::lround(pixel.y()));
        if (pixel_x < 0 || pixel_y < 0 || pixel_x >= width ||
            pixel_y >= height) {
          continue;
        }
        const auto raw = depth_image.at(static_cast<unsigned int>(pixel_x),
                                        static_cast<unsigned int>(pixel_y));
        if (raw == 0) {
          continue;
        }
        const float surface_depth = depth_to_meters(raw, options.depth_scale);
        if (surface_depth < options.min_depth ||
            surface_depth > options.max_depth) {
          continue;
        }
        const Eigen::Vector3f ray = intrinsics.back_project(
            {static_cast<float>(pixel_x), static_cast<float>(pixel_y)}, 1.0F);
        const float lambda = ray.norm();
        if (!ray.allFinite() || lambda == 0.0F) {
          continue;
        }
        float signed_distance = surface_depth - camera_point.z();
        if (options.projective_distance) {
          signed_distance = surface_depth - (camera_point.norm() / lambda);
        }
        float truncation_distance = truncation_distance_;
        if (options.distance_scaled_truncation) {
          truncation_distance +=
              options.truncation_distance_scale * surface_depth;
        }
        if (signed_distance < -truncation_distance) {
          continue;
        }
        const float tsdf =
            std::clamp(signed_distance / truncation_distance, -1.0F, 1.0F);

        float weight = options.observation_weight;
        if (options.view_angle_weighting && normals != nullptr) {
          const Vec3f& normal_sample =
              normals->at(static_cast<unsigned int>(pixel_x),
                          static_cast<unsigned int>(pixel_y));
          if (all_finite(normal_sample)) {
            const Eigen::Vector3f normal = to_eigen(normal_sample);
            // View direction is the (quantised) pixel ray, matching the ray
            // used for the projective TSDF distance above.
            const Eigen::Vector3f view = -ray.normalized();
            weight *=
                std::max(0.0F, normal.normalized().dot(view)) / surface_depth;
          }
        }
        if (weight <= 0.0F) {
          continue;
        }

        Voxel& voxel = at(x, y, z);
        const float combined = voxel.weight + weight;
        voxel.distance =
            ((voxel.distance * voxel.weight) + (tsdf * weight)) / combined;
        voxel.weight = std::min(combined, options.max_weight);

        if (color_image != nullptr &&
            signed_distance <=
                truncation_distance * color_integration_truncation_fraction) {
          const ColorRgba rgba = rgba_from_pixel(
              color_image->at(static_cast<unsigned int>(pixel_x),
                              static_cast<unsigned int>(pixel_y)));
          const Vec3f observed{.x = static_cast<float>(rgba.x()),
                               .y = static_cast<float>(rgba.y()),
                               .z = static_cast<float>(rgba.z())};
          ColorVoxel& color_voxel = color_at(x, y, z);
          const float color_combined = color_voxel.weight + weight;
          color_voxel.color =
              (color_voxel.color * color_voxel.weight + observed * weight) /
              color_combined;
          color_voxel.weight = std::min(color_combined, options.max_weight);
        }
      }
    }
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
  SurfaceMaps maps{
      .points = image_proc::Vector3fImage{options.width, options.height,
                                          invalid_vec3f()},
      .normals = image_proc::Vector3fImage{options.width, options.height,
                                           invalid_vec3f()},
      .colors = image_proc::ColorImage{options.width, options.height}};

  const Eigen::Matrix3f rotation = options.camera_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3f origin = options.camera_to_world.block<3, 1>(0, 3);
  const float base_step = voxel_size_ * options.step_scale;

  for (unsigned int row = 0; row < options.height; ++row) {
    for (unsigned int col = 0; col < options.width; ++col) {
      const Eigen::Vector3f ray_camera = options.intrinsics.back_project(
          {static_cast<float>(col), static_cast<float>(row)}, 1.0F);
      Eigen::Vector3f direction = rotation * ray_camera;
      const float direction_norm = direction.norm();
      if (direction_norm <= 0.0F) {
        continue;
      }
      direction /= direction_norm;

      float ray_length = options.min_depth;
      float previous_tsdf = 0.0F;
      Eigen::Vector3f previous_point = origin;
      bool have_previous = false;
      while (ray_length <= options.max_depth) {
        const Eigen::Vector3f point = origin + ray_length * direction;
        float tsdf = 0.0F;
        const bool sampled =
            sample_tsdf(point, tsdf, options.tsdf_from_valid_corners);
        if (!sampled) {
          ray_length += base_step;
          have_previous = false;
          continue;
        }
        if (have_previous && previous_tsdf > 0.0F && tsdf <= 0.0F) {
          const float gap = previous_tsdf - tsdf;
          const float alpha = gap != 0.0F ? previous_tsdf / gap : 0.0F;
          const Eigen::Vector3f surface =
              previous_point + alpha * (point - previous_point);
          Eigen::Vector3f normal;
          if (sample_normal(surface, normal, options.tsdf_from_valid_corners)) {
            maps.points.at(col, row) = from_eigen(surface);
            maps.normals.at(col, row) = from_eigen(normal);
            Vec3f color;
            if (sample_color(surface, color)) {
              maps.colors.at(col, row) = pixel_from_color(color);
            }
          }
          break;
        }
        if (have_previous && previous_tsdf < 0.0F && tsdf > 0.0F) {
          break;
        }
        previous_tsdf = tsdf;
        previous_point = point;
        have_previous = true;
        float step = base_step;
        if (tsdf > 0.0F) {
          step = std::max(base_step,
                          tsdf * truncation_distance_ * options.step_scale);
        }
        ray_length += step;
      }
    }
  }
  return maps;
}

Volume::GridSample Volume::grid_sample(const Eigen::Vector3f& point) const {
  const Eigen::Vector3f volume_origin = to_eigen(origin_);
  const Eigen::Vector3f grid =
      (point - volume_origin) / voxel_size_ - Eigen::Vector3f::Constant(0.5F);
  const int base_x = static_cast<int>(std::floor(grid.x()));
  const int base_y = static_cast<int>(std::floor(grid.y()));
  const int base_z = static_cast<int>(std::floor(grid.z()));
  return GridSample{.base_x = base_x,
                    .base_y = base_y,
                    .base_z = base_z,
                    .tx = grid.x() - static_cast<float>(base_x),
                    .ty = grid.y() - static_cast<float>(base_y),
                    .tz = grid.z() - static_cast<float>(base_z)};
}

float Volume::trilinear_weight(const GridSample& sample, int offset_x,
                               int offset_y, int offset_z) {
  const std::array<float, 2> weights_x = {1.0F - sample.tx, sample.tx};
  const std::array<float, 2> weights_y = {1.0F - sample.ty, sample.ty};
  const std::array<float, 2> weights_z = {1.0F - sample.tz, sample.tz};

  return weights_x.at(static_cast<std::size_t>(offset_x)) *
         weights_y.at(static_cast<std::size_t>(offset_y)) *
         weights_z.at(static_cast<std::size_t>(offset_z));
}

std::array<Corner, trilinear_corner_count> Volume::trilinear_corners(
    const GridSample& sample) {
  std::array<Corner, trilinear_corner_count> out{};
  std::size_t corner_idx = 0;
  for (int offset_z = 0; offset_z <= 1; ++offset_z) {
    for (int offset_y = 0; offset_y <= 1; ++offset_y) {
      for (int offset_x = 0; offset_x <= 1; ++offset_x) {
        out.at(corner_idx++) = {
            .x = sample.base_x + offset_x,
            .y = sample.base_y + offset_y,
            .z = sample.base_z + offset_z,
            .weight = trilinear_weight(sample, offset_x, offset_y, offset_z)};
      }
    }
  }
  return out;
}

bool Volume::sample_tsdf_available_corners(const Eigen::Vector3f& point,
                                           float& distance) const {
  if (!contains(point)) {
    return false;
  }

  const GridSample sample = grid_sample(point);
  float accumulated = 0.0F;
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    if (!in_bounds(corner.x, corner.y, corner.z)) {
      continue;
    }
    const Voxel& voxel = at(corner);
    if (voxel.weight <= 0.0F || !std::isfinite(voxel.distance)) {
      continue;
    }
    accumulated += corner.weight * voxel.distance;
    weight_sum += corner.weight;
  }
  if (weight_sum <= minimum_trilinear_weight_sum) {
    return false;
  }
  distance = accumulated / weight_sum;
  return std::isfinite(distance);
}

bool Volume::sample_tsdf(const Eigen::Vector3f& point, float& distance,
                         bool from_valid_corners) const {
  return from_valid_corners ? sample_tsdf_valid_corners(point, distance)
                            : sample_tsdf_available_corners(point, distance);
}

bool Volume::sample_tsdf_valid_corners(const Eigen::Vector3f& point,
                                       float& distance) const {
  if (!contains(point)) {
    return false;
  }

  const GridSample sample = grid_sample(point);
  float accumulated = 0.0F;
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    if (!in_bounds(corner.x, corner.y, corner.z)) {
      return false;
    }
    const Voxel& voxel = at(corner);
    if (voxel.weight <= 0.0F || !std::isfinite(voxel.distance)) {
      return false;
    }
    accumulated += corner.weight * voxel.distance;
    weight_sum += corner.weight;
  }
  if (weight_sum <= minimum_trilinear_weight_sum) {
    return false;
  }
  distance = accumulated / weight_sum;
  return std::isfinite(distance);
}

bool Volume::sample_color(const Eigen::Vector3f& point, Vec3f& color) const {
  const GridSample sample = grid_sample(point);
  Vec3f accumulated{};
  float weight_sum = 0.0F;
  for (const Corner& corner : trilinear_corners(sample)) {
    if (!in_bounds(corner.x, corner.y, corner.z)) {
      continue;
    }
    const ColorVoxel& voxel = color_at(corner);
    if (voxel.weight <= 0.0F) {
      continue;
    }
    accumulated += corner.weight * voxel.color;
    weight_sum += corner.weight;
  }
  if (weight_sum <= minimum_trilinear_weight_sum) {
    return false;
  }
  color = accumulated / weight_sum;
  return true;
}

bool Volume::sample_normal(const Eigen::Vector3f& point,
                           Eigen::Vector3f& normal,
                           bool tsdf_from_valid_corners) const {
  const float step = voxel_size_;
  float x_plus = 0.0F;
  float x_minus = 0.0F;
  float y_plus = 0.0F;
  float y_minus = 0.0F;
  float z_plus = 0.0F;
  float z_minus = 0.0F;
  const bool corners = tsdf_from_valid_corners;
  if (!sample_tsdf(point + Eigen::Vector3f{step, 0.0F, 0.0F}, x_plus,
                   corners) ||
      !sample_tsdf(point - Eigen::Vector3f{step, 0.0F, 0.0F}, x_minus,
                   corners) ||
      !sample_tsdf(point + Eigen::Vector3f{0.0F, step, 0.0F}, y_plus,
                   corners) ||
      !sample_tsdf(point - Eigen::Vector3f{0.0F, step, 0.0F}, y_minus,
                   corners) ||
      !sample_tsdf(point + Eigen::Vector3f{0.0F, 0.0F, step}, z_plus,
                   corners) ||
      !sample_tsdf(point - Eigen::Vector3f{0.0F, 0.0F, step}, z_minus,
                   corners)) {
    return false;
  }
  normal =
      Eigen::Vector3f{x_plus - x_minus, y_plus - y_minus, z_plus - z_minus};
  const float norm = normal.norm();
  if (norm <= 0.0F) {
    return false;
  }
  normal /= norm;
  return true;
}

std::uint32_t Volume::pixel_from_color(const Vec3f& color) {
  const auto to_byte = [](float value) {
    return static_cast<std::uint32_t>(
        std::clamp(value, 0.0F, max_color_channel_value_f));
  };
  return to_byte(color.x) | (to_byte(color.y) << color_green_shift) |
         (to_byte(color.z) << color_blue_shift) |
         (std::uint32_t{max_color_channel_value} << color_alpha_shift);
}

}  // namespace kinectfusion
