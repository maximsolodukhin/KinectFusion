#include <kinectfusion/volume.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace kinectfusion {

void Volume::integrate_depth_image(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& world_to_camera,
    const image_proc::ColorImage* color_image,
    const image_proc::Image<Eigen::Vector3f>* normals) {
  const Eigen::Matrix3f rotation = world_to_camera.block<3, 3>(0, 0);
  const Eigen::Vector3f translation = world_to_camera.block<3, 1>(0, 3);
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());

  for (int z = 0; z < resolution_.z(); ++z) {
    for (int y = 0; y < resolution_.y(); ++y) {
      for (int x = 0; x < resolution_.x(); ++x) {
        const Eigen::Vector3f center =
            origin_ + Eigen::Vector3f{static_cast<float>(x) + 0.5F,
                                      static_cast<float>(y) + 0.5F,
                                      static_cast<float>(z) + 0.5F} *
                          voxel_size_;
        const Eigen::Vector3f camera_point = rotation * center + translation;
        if (camera_point.z() <= 0.0F) {
          continue;
        }
        const float u = intrinsics.fx * camera_point.x() / camera_point.z() +
                        intrinsics.cx;
        const float v = intrinsics.fy * camera_point.y() / camera_point.z() +
                        intrinsics.cy;
        const int px = static_cast<int>(std::lround(u));
        const int py = static_cast<int>(std::lround(v));
        if (px < 0 || py < 0 || px >= width || py >= height) {
          continue;
        }
        const auto raw = depth_image.at(static_cast<unsigned int>(px),
                                        static_cast<unsigned int>(py));
        if (raw == 0) {
          continue;
        }
        const float surface_depth = depth_to_meters(raw);
        if (surface_depth < kMinDepth || surface_depth > kMaxDepth) {
          continue;
        }
        const Eigen::Vector3f ray{
            (static_cast<float>(px) - intrinsics.cx) / intrinsics.fx,
            (static_cast<float>(py) - intrinsics.cy) / intrinsics.fy,
            1.0F};
        const float lambda = ray.norm();
        if (!ray.allFinite() || lambda == 0.0F) {
          continue;
        }
        const float signed_distance =
            surface_depth - camera_point.norm() / lambda;
        if (signed_distance < -truncation_distance_) {
          continue;
        }
        const float tsdf =
            std::clamp(signed_distance / truncation_distance_, -1.0F, 1.0F);

        float weight = kObservationWeight;
        if (normals != nullptr) {
          const Eigen::Vector3f& normal = normals->at(
              static_cast<unsigned int>(px), static_cast<unsigned int>(py));
          if (normal.allFinite()) {
            // View direction is the (quantised) pixel ray, matching the ray
            // used for the projective TSDF distance above.
            const Eigen::Vector3f view = -ray.normalized();
            weight *= std::max(0.0F, normal.normalized().dot(view)) / surface_depth;
          }
        }
        if (weight <= 0.0F) {
          continue;
        }

        Voxel& voxel = at(x, y, z);
        const float combined = voxel.weight + weight;
        voxel.distance =
            (voxel.distance * voxel.weight + tsdf * weight) / combined;
        voxel.weight = std::min(combined, kMaxWeight);

        if (color_image != nullptr && signed_distance <= truncation_distance_ * 0.5F) {
          const ColorRgba rgba = rgba_from_pixel(color_image->at(
              static_cast<unsigned int>(px), static_cast<unsigned int>(py)));
          const Eigen::Vector3f observed{static_cast<float>(rgba.x()),
                                         static_cast<float>(rgba.y()),
                                         static_cast<float>(rgba.z())};
          ColorVoxel& color_voxel = color_at(x, y, z);
          const float color_combined = color_voxel.weight + weight;
          color_voxel.color = (color_voxel.color * color_voxel.weight + observed * weight) / color_combined;
          color_voxel.weight = std::min(color_combined, kMaxWeight);
        }
      }
    }
  }
}

SurfaceMaps Volume::raycast(const CameraIntrinsics& intrinsics,
                            unsigned int width, unsigned int height,
                            const Eigen::Matrix4f& camera_to_world) const {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  SurfaceMaps maps{
      .points = image_proc::Image<Eigen::Vector3f>{width, height},
      .normals = image_proc::Image<Eigen::Vector3f>{width, height},
      .colors = image_proc::ColorImage{width, height}};
  for (auto& point : maps.points.data()) {
    point = Eigen::Vector3f::Constant(nan);
  }
  for (auto& normal : maps.normals.data()) {
    normal = Eigen::Vector3f::Constant(nan);
  }

  const Eigen::Matrix3f rotation = camera_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3f origin = camera_to_world.block<3, 1>(0, 3);
  const float base_step = voxel_size_ * kRaycastStepScale;

  for (unsigned int row = 0; row < height; ++row) {
    for (unsigned int col = 0; col < width; ++col) {
      const Eigen::Vector3f ray_camera{
          (static_cast<float>(col) - intrinsics.cx) / intrinsics.fx,
          (static_cast<float>(row) - intrinsics.cy) / intrinsics.fy, 1.0F};
      Eigen::Vector3f direction = rotation * ray_camera;
      const float direction_norm = direction.norm();
      if (direction_norm <= 0.0F) {
        continue;
      }
      direction /= direction_norm;

      float ray_length = kMinDepth;
      float previous_tsdf = 0.0F;
      Eigen::Vector3f previous_point = origin;
      bool have_previous = false;
      while (ray_length <= kMaxDepth) {
        const Eigen::Vector3f point = origin + ray_length * direction;
        float tsdf = 0.0F;
        if (!sample_tsdf(point, tsdf)) {
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
          if (sample_normal(surface, normal)) {
            maps.points.at(col, row) = surface;
            maps.normals.at(col, row) = normal;
            Eigen::Vector3f color;
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
                          tsdf * truncation_distance_ * kRaycastStepScale);
        }
        ray_length += step;
      }
    }
  }
  return maps;
}

Volume::GridSample Volume::grid_sample(const Eigen::Vector3f& point) const {
  const Eigen::Vector3f grid =
      (point - origin_) / voxel_size_ - Eigen::Vector3f::Constant(0.5F);
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

float Volume::trilinear_weight(const GridSample& sample, int dx, int dy,
                               int dz) {
  const float wx = dx != 0 ? sample.tx : 1.0F - sample.tx;
  const float wy = dy != 0 ? sample.ty : 1.0F - sample.ty;
  const float wz = dz != 0 ? sample.tz : 1.0F - sample.tz;
  return wx * wy * wz;
}

bool Volume::sample_tsdf(const Eigen::Vector3f& point, float& distance) const {
  if (!contains(point)) {
    return false;
  }

  const GridSample sample = grid_sample(point);
  float accumulated = 0.0F;
  float weight_sum = 0.0F;
  for (int dz = 0; dz <= 1; ++dz) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dx = 0; dx <= 1; ++dx) {
        const int x = sample.base_x + dx;
        const int y = sample.base_y + dy;
        const int z = sample.base_z + dz;
        if (!in_bounds(x, y, z)) {
          continue;
        }
        const Voxel& voxel = at(x, y, z);
        if (voxel.weight <= 0.0F || !std::isfinite(voxel.distance)) {
          continue;
        }
        const float weight = trilinear_weight(sample, dx, dy, dz);
        accumulated += weight * voxel.distance;
        weight_sum += weight;
      }
    }
  }
  if (weight_sum <= 1.0e-6F) {
    return false;
  }
  distance = accumulated / weight_sum;
  return std::isfinite(distance);
}

bool Volume::sample_color(const Eigen::Vector3f& point,
                          Eigen::Vector3f& color) const {
  const GridSample sample = grid_sample(point);
  Eigen::Vector3f accumulated = Eigen::Vector3f::Zero();
  float weight_sum = 0.0F;
  for (int dz = 0; dz <= 1; ++dz) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dx = 0; dx <= 1; ++dx) {
        const int x = sample.base_x + dx;
        const int y = sample.base_y + dy;
        const int z = sample.base_z + dz;
        if (!in_bounds(x, y, z)) {
          continue;
        }
        const ColorVoxel& voxel = color_at(x, y, z);
        if (voxel.weight <= 0.0F) {
          continue;
        }
        const float weight =
            trilinear_weight(sample, dx, dy, dz) * voxel.weight;
        accumulated += weight * voxel.color;
        weight_sum += weight;
      }
    }
  }
  if (weight_sum <= 1.0e-6F) {
    return false;
  }
  color = accumulated / weight_sum;
  return true;
}

bool Volume::sample_normal(const Eigen::Vector3f& point,
                           Eigen::Vector3f& normal) const {
  const float h = voxel_size_;
  float x_plus = 0.0F;
  float x_minus = 0.0F;
  float y_plus = 0.0F;
  float y_minus = 0.0F;
  float z_plus = 0.0F;
  float z_minus = 0.0F;
  if (!sample_tsdf(point + Eigen::Vector3f{h, 0.0F, 0.0F}, x_plus) ||
      !sample_tsdf(point - Eigen::Vector3f{h, 0.0F, 0.0F}, x_minus) ||
      !sample_tsdf(point + Eigen::Vector3f{0.0F, h, 0.0F}, y_plus) ||
      !sample_tsdf(point - Eigen::Vector3f{0.0F, h, 0.0F}, y_minus) ||
      !sample_tsdf(point + Eigen::Vector3f{0.0F, 0.0F, h}, z_plus) ||
      !sample_tsdf(point - Eigen::Vector3f{0.0F, 0.0F, h}, z_minus)) {
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

std::uint32_t Volume::pixel_from_color(const Eigen::Vector3f& color) {
  const auto to_byte = [](float value) {
    return static_cast<std::uint32_t>(std::clamp(value, 0.0F, 255.0F));
  };
  return to_byte(color.x()) | (to_byte(color.y()) << 8U) |
         (to_byte(color.z()) << 16U) | (std::uint32_t{255} << 24U);
}

}  // namespace kinectfusion
