#include <kinectfusion/volume.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace kinectfusion {
namespace {

[[nodiscard]] std::uint32_t rgba_to_uint32(const ColorRgba& color) {
  return static_cast<std::uint32_t>(color.x()) |
         (static_cast<std::uint32_t>(color.y()) << 8U) |
         (static_cast<std::uint32_t>(color.z()) << 16U) |
         (static_cast<std::uint32_t>(color.w()) << 24U);
}

}  // namespace

void Volume::integrate_depth_image(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& world_to_camera,
    const TsdfIntegrationOptions& options,
    const image_proc::ColorImage* color_image,
    const image_proc::Image<Eigen::Vector3f>* normals) {
  validate_options(options);

  for (int z = 0; z < resolution_.z(); ++z) {
    for (int y = 0; y < resolution_.y(); ++y) {
      for (int x = 0; x < resolution_.x(); ++x) {
        integrate_voxel(x, y, z, depth_image, intrinsics, world_to_camera,
                        options, color_image, normals);
      }
    }
  }
}

void Volume::integrate_voxel(
    int x, int y, int z,
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& world_to_camera,
    const TsdfIntegrationOptions& options,
    const image_proc::ColorImage* color_image,
    const image_proc::Image<Eigen::Vector3f>* normals) {
  const Eigen::Vector3f center = voxel_center(x, y, z);
  const Eigen::Vector4f camera_point =
      world_to_camera * Eigen::Vector4f{center.x(), center.y(), center.z(), 1.0F};
  if (camera_point.z() <= 0.0F) {
    return;
  }

  const auto u = static_cast<int>(
      std::lround((camera_point.x() * intrinsics.fx / camera_point.z()) +
                  intrinsics.cx));
  const auto v = static_cast<int>(
      std::lround((camera_point.y() * intrinsics.fy / camera_point.z()) +
                  intrinsics.cy));

  if (u < 0 || v < 0 ||
      std::cmp_greater_equal(u, depth_image.width()) ||
      std::cmp_greater_equal(v, depth_image.height())) {
    return;
  }

  const auto depth_raw =
      depth_image.at(static_cast<unsigned int>(u), static_cast<unsigned int>(v));
  if (!is_valid_depth(depth_raw)) {
    return;
  }

  const float surface_depth = depth_to_meters(depth_raw, options.depth_scale);
  if (surface_depth < options.min_depth || surface_depth > options.max_depth) {
    return;
  }

  const Eigen::Vector3f ray{
      (static_cast<float>(u) - intrinsics.cx) / intrinsics.fx,
      (static_cast<float>(v) - intrinsics.cy) / intrinsics.fy,
      1.0F};
  const float lambda = ray.norm();
  if (!ray.allFinite() || lambda == 0.0F) {
    return;
  }

  const float voxel_depth =
      options.projective_distance
          ? camera_point.head<3>().norm() / lambda
          : camera_point.z();
  const float truncation_distance =
      options.distance_scaled_truncation
          ? truncation_distance_ +
                (options.truncation_distance_scale * surface_depth)
          : truncation_distance_;
  const float signed_distance = surface_depth - voxel_depth;
  if (signed_distance < -truncation_distance) {
    return;
  }

  float observation_weight = options.observation_weight;
  if (options.view_angle_weighting && normals != nullptr &&
      normals->width() == depth_image.width() &&
      normals->height() == depth_image.height()) {
    const Eigen::Vector3f& normal =
        normals->at(static_cast<unsigned int>(u), static_cast<unsigned int>(v));
    if (normal.allFinite()) {
      const Eigen::Vector3f ray_direction = ray.normalized();
      const float cos_theta = std::max(0.0F, normal.normalized().dot(-ray_direction));
      if (cos_theta <= 0.0F) {
        return;
      }
      observation_weight *= cos_theta / surface_depth;
    }
  }
  if (observation_weight <= 0.0F || !std::isfinite(observation_weight)) {
    return;
  }

  const float tsdf =
      std::clamp(signed_distance / truncation_distance, -1.0F, 1.0F);
  auto& voxel = at(x, y, z);
  const float blended_weight = voxel.weight + observation_weight;
  voxel.distance =
      ((voxel.distance * voxel.weight) + (tsdf * observation_weight)) /
      blended_weight;
  voxel.weight = std::min(blended_weight, options.max_weight);

  if (signed_distance <= truncation_distance * 0.5F && color_image != nullptr &&
      color_image->width() == depth_image.width() &&
      color_image->height() == depth_image.height()) {
    integrate_color(x, y, z,
                    color_image->at(static_cast<unsigned int>(u),
                                    static_cast<unsigned int>(v)),
                    observation_weight, options.max_weight);
  }
}

SurfaceMaps Volume::raycast(const RaycastOptions& options) const {
  if (options.width == 0 || options.height == 0) {
    throw std::invalid_argument("Raycast image dimensions must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Raycast depth range is invalid");
  }
  if (options.step_scale <= 0.0F) {
    throw std::invalid_argument("Raycast step scale must be positive");
  }

  SurfaceMaps maps{
      .points = image_proc::Image<Eigen::Vector3f>{options.width,
                                                   options.height},
      .normals = image_proc::Image<Eigen::Vector3f>{options.width,
                                                    options.height},
      .colors = image_proc::ColorImage{options.width, options.height}};

  std::ranges::fill(maps.points.data(), invalid_vector());
  std::ranges::fill(maps.normals.data(), invalid_vector());
  std::ranges::fill(maps.colors.data(), std::uint32_t{});

  const RaycastContext context{
      .rotation = options.camera_to_world.block<3, 3>(0, 0),
      .origin = options.camera_to_world.block<3, 1>(0, 3),
      .base_step = voxel_size_ * options.step_scale};

  for (unsigned int row = 0; row < options.height; ++row) {
    for (unsigned int col = 0; col < options.width; ++col) {
      raycast_pixel(options, context, ImagePixel{.x = col, .y = row}, maps);
    }
  }

  return maps;
}

void Volume::integrate_color(int x, int y, int z, std::uint32_t color,
                       float observation_weight, float max_weight) {
  auto& voxel = colors_[index(x, y, z)];
  const auto rgba = rgba_from_pixel(color);
  const float blended_weight = voxel.weight + observation_weight;
  voxel.r = ((voxel.r * voxel.weight) +
             (static_cast<float>(rgba.x()) * observation_weight)) /
            blended_weight;
  voxel.g = ((voxel.g * voxel.weight) +
             (static_cast<float>(rgba.y()) * observation_weight)) /
            blended_weight;
  voxel.b = ((voxel.b * voxel.weight) +
             (static_cast<float>(rgba.z()) * observation_weight)) /
            blended_weight;
  voxel.weight = std::min(blended_weight, max_weight);
}

Volume::GridSamplePosition Volume::grid_sample_position(
    const Eigen::Vector3f& point) const {
  const Eigen::Vector3f grid_position =
      ((point - origin_) / voxel_size_) - Eigen::Vector3f{0.5F, 0.5F, 0.5F};

  const int base_x = static_cast<int>(std::floor(grid_position.x()));
  const int base_y = static_cast<int>(std::floor(grid_position.y()));
  const int base_z = static_cast<int>(std::floor(grid_position.z()));
  return GridSamplePosition{
      .base_x = base_x,
      .base_y = base_y,
      .base_z = base_z,
      .tx = grid_position.x() - static_cast<float>(base_x),
      .ty = grid_position.y() - static_cast<float>(base_y),
      .tz = grid_position.z() - static_cast<float>(base_z)};
}

bool Volume::sample_tsdf(const Eigen::Vector3f& point, float& distance) const {
  const auto position = grid_sample_position(point);

  float result = 0.0F;
  float weight_sum = 0.0F;
  for (int dx = 0; dx <= 1; ++dx) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dz = 0; dz <= 1; ++dz) {
        float corner_distance = 0.0F;
        if (!strict_tsdf_corner(position.base_x + dx, position.base_y + dy,
                                position.base_z + dz, corner_distance)) {
          return false;
        }

        const float interpolation_weight =
            trilinear_weight(position, dx, dy, dz);
        result += interpolation_weight * corner_distance;
        weight_sum += interpolation_weight;
      }
    }
  }

  if (weight_sum < 1e-6F) {
    return false;
  }

  distance = result / weight_sum;
  return std::isfinite(distance);
}

bool Volume::sample_tsdf_valid_corners(const Eigen::Vector3f& point,
                                          float& distance) const {
  if (!contains(point)) {
    return false;
  }

  const auto position = grid_sample_position(point);

  float result = 0.0F;
  float weight_sum = 0.0F;
  for (int dx = 0; dx <= 1; ++dx) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dz = 0; dz <= 1; ++dz) {
        const int x = position.base_x + dx;
        const int y = position.base_y + dy;
        const int z = position.base_z + dz;
        if (!in_bounds(x, y, z)) {
            continue;
        }

        const auto& voxel = at(x, y, z);
        if (voxel.weight <= 0.0F || !std::isfinite(voxel.distance)) {
            continue;
        }

        const float weight = trilinear_weight(position, dx, dy, dz);
        result += weight * voxel.distance;
        weight_sum += weight;
      }
    }
  }

  if (weight_sum < 1e-6F) {
    return false;
  }
  distance = result / weight_sum;
  return std::isfinite(distance);
}

float Volume::trilinear_weight(const GridSamplePosition& position,
                                   int offset_x, int offset_y, int offset_z) {
  const float wx = offset_x == 0 ? 1.0F - position.tx : position.tx;
  const float wy = offset_y == 0 ? 1.0F - position.ty : position.ty;
  const float wz = offset_z == 0 ? 1.0F - position.tz : position.tz;
  return wx * wy * wz;
}

bool Volume::strict_tsdf_corner(int x, int y, int z, float& distance) const {
  if (!in_bounds(x, y, z)) {
    return false;
  }

  const auto& voxel = at(x, y, z);
  if (voxel.weight <= 0.0F || !std::isfinite(voxel.distance)) {
    return false;
  }

  distance = voxel.distance;
  return true;
}

bool Volume::sample_normal(const Eigen::Vector3f& point, Eigen::Vector3f& normal,
                           bool tsdf_from_valid_corners) const {
  float x_plus = 0.0F;
  float x_minus = 0.0F;
  float y_plus = 0.0F;
  float y_minus = 0.0F;
  float z_plus = 0.0F;
  float z_minus = 0.0F;
  const Eigen::Vector3f dx{voxel_size_, 0.0F, 0.0F};
  const Eigen::Vector3f dy{0.0F, voxel_size_, 0.0F};
  const Eigen::Vector3f dz{0.0F, 0.0F, voxel_size_};

  const auto sample =
      [this, tsdf_from_valid_corners](const Eigen::Vector3f& sample_point,
                                   float& distance) {
        return tsdf_from_valid_corners
                   ? sample_tsdf_valid_corners(sample_point, distance)
                   : sample_tsdf(sample_point, distance);
      };

  if (!sample(point + dx, x_plus) || !sample(point - dx, x_minus) ||
      !sample(point + dy, y_plus) || !sample(point - dy, y_minus) ||
      !sample(point + dz, z_plus) || !sample(point - dz, z_minus)) {
    return false;
  }

  normal = Eigen::Vector3f{x_plus - x_minus, y_plus - y_minus, z_plus - z_minus};
  const float norm = normal.norm();
  if (!normal.allFinite() || norm < 1e-6F) {
    return false;
  }

  normal /= norm;
  return true;
}

ColorRgba Volume::sample_color(const Eigen::Vector3f& point) const {
  const auto position = grid_sample_position(point);

  Eigen::Vector3f result = Eigen::Vector3f::Zero();
  float weight_sum = 0.0F;
  for (int dx = 0; dx <= 1; ++dx) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dz = 0; dz <= 1; ++dz) {
        const int x = position.base_x + dx;
        const int y = position.base_y + dy;
        const int z = position.base_z + dz;
        if (!in_bounds(x, y, z)) {
          continue;
        }

        const auto& color = color_at(x, y, z);
        if (color.weight <= 0.0F) {
          continue;
        }

        const float weight =
            trilinear_weight(position, dx, dy, dz) * color.weight;
        result += weight * Eigen::Vector3f{color.r, color.g, color.b};
        weight_sum += weight;
      }
    }
  }

  if (weight_sum <= 0.0F) {
    return ColorRgba{0, 0, 0, 0};
  }

  result /= weight_sum;
  return ColorRgba{static_cast<std::uint8_t>(std::clamp(result.x(), 0.0F, 255.0F)),
                   static_cast<std::uint8_t>(std::clamp(result.y(), 0.0F, 255.0F)),
                   static_cast<std::uint8_t>(std::clamp(result.z(), 0.0F, 255.0F)),
                   255};
}

Eigen::Vector3f Volume::ray_direction_for_pixel(
    const RaycastOptions& options, const RaycastContext& context,
    ImagePixel pixel) {
  const Eigen::Vector3f ray_direction_camera{
      (static_cast<float>(pixel.x) - options.intrinsics.cx) /
          options.intrinsics.fx,
      (static_cast<float>(pixel.y) - options.intrinsics.cy) /
          options.intrinsics.fy,
      1.0F};
  const Eigen::Vector3f ray_direction = context.rotation * ray_direction_camera;
  const float direction_norm = ray_direction.norm();
  if (!ray_direction.allFinite() || direction_norm == 0.0F) {
    return invalid_vector();
  }
  return ray_direction / direction_norm;
}

void Volume::raycast_pixel(const RaycastOptions& options,
                           const RaycastContext& context, ImagePixel pixel,
                           SurfaceMaps& maps) const {
  const auto ray_direction = ray_direction_for_pixel(options, context, pixel);
  if (!ray_direction.allFinite()) {
    return;
  }

  float ray_length = options.min_depth;
  bool have_previous_sample = false;
  float previous_tsdf = 0.0F;
  Eigen::Vector3f previous_point = Eigen::Vector3f::Zero();

  while (ray_length <= options.max_depth) {
    const Eigen::Vector3f point =
        context.origin + (ray_length * ray_direction);

    float current_tsdf = 0.0F;
    const bool sampled =
        options.tsdf_from_valid_corners
            ? sample_tsdf_valid_corners(point, current_tsdf)
            : sample_tsdf(point, current_tsdf);
    if (!sampled) {
      ray_length += context.base_step;
      have_previous_sample = false;
      continue;
    }

    if (have_previous_sample && previous_tsdf > 0.0F &&
        current_tsdf <= 0.0F) {
      const float alpha = previous_tsdf / (previous_tsdf - current_tsdf);
      const Eigen::Vector3f surface_point =
          previous_point + (alpha * (point - previous_point));

      Eigen::Vector3f normal;
      if (sample_normal(surface_point, normal, options.tsdf_from_valid_corners)) {
        maps.points.at(pixel.x, pixel.y) = surface_point;
        maps.normals.at(pixel.x, pixel.y) = normal;
        maps.colors.at(pixel.x, pixel.y) = rgba_to_uint32(sample_color(surface_point));
      }
      return;
    }

    if (have_previous_sample && previous_tsdf < 0.0F && current_tsdf > 0.0F) {
      return;
    }

    previous_tsdf = current_tsdf;
    previous_point = point;
    have_previous_sample = true;

    const float step =
        current_tsdf > 0.0F
            ? std::max(context.base_step,
                       current_tsdf * truncation_distance_ * options.step_scale)
            : context.base_step;
    ray_length += step;
  }
}

}  // namespace kinectfusion
