#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

struct Corner { int x, y, z; float weight; };

struct Voxel {
  float distance{1.0F};
  float weight{0.0F};
};

struct ColorVoxel {
  Vec3f color{};
  float weight{0.0F};
};

struct SurfaceMaps {
  image_proc::Vector3fImage points;
  image_proc::Vector3fImage normals;
  image_proc::ColorImage colors;
};

struct SurfaceMapsView {
  image_proc::ImageView<Vec3f> points;
  image_proc::ImageView<Vec3f> normals;
  image_proc::ImageView<std::uint32_t> colors;
};

struct ConstSurfaceMapsView {
  image_proc::ImageView<const Vec3f> points;
  image_proc::ImageView<const Vec3f> normals;
  image_proc::ImageView<const std::uint32_t> colors;
};

[[nodiscard]] inline SurfaceMapsView view(SurfaceMaps& maps) {
  return SurfaceMapsView{.points = maps.points.view(),
                         .normals = maps.normals.view(),
                         .colors = maps.colors.view()};
}

[[nodiscard]] inline ConstSurfaceMapsView view(const SurfaceMaps& maps) {
  return ConstSurfaceMapsView{.points = maps.points.view(),
                              .normals = maps.normals.view(),
                              .colors = maps.colors.view()};
}

struct TsdfIntegrationOptions {
  float depth_scale{5000.0F};
  float observation_weight{1.0F};
  float max_weight{196.0F};
  float min_depth{0.4F};
  float max_depth{8.0F};
  bool projective_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{0.01F};
  bool view_angle_weighting{true};
};

struct RaycastOptions {
  CameraIntrinsics intrinsics{};
  std::size_t width{};
  std::size_t height{};
  Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};
  float min_depth{0.4F};
  float max_depth{8.0F};
  float step_scale{1.0F};
  bool tsdf_from_valid_corners{false};
};

struct VolumeView {
  Voxel* voxels{};
  ColorVoxel* colors{};
  Size3 resolution{};
  float voxel_size{};
  Vec3f origin{};
  float truncation_distance{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return (z * resolution.y + y) * resolution.x + x;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Voxel& voxel_at(
      std::size_t x, std::size_t y, std::size_t z) {
    return voxels[index(x, y, z)];
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE ColorVoxel& color_at(
      std::size_t x, std::size_t y, std::size_t z) {
    return colors[index(x, y, z)];
  }
};

struct ConstVolumeView {
  const Voxel* voxels{};
  const ColorVoxel* colors{};
  Size3 resolution{};
  float voxel_size{};
  Vec3f origin{};
  float truncation_distance{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return (z * resolution.y + y) * resolution.x + x;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Voxel& voxel_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return voxels[index(x, y, z)];
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const ColorVoxel& color_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return colors[index(x, y, z)];
  }
};

class Volume {
 public:
  Volume(const kinectfusion::Vector3s& resolution, float voxel_size,
         const Vec3f& origin, float truncation_distance)
      : resolution_(resolution),
        voxel_size_(voxel_size),
        origin_(origin),
        truncation_distance_(truncation_distance),
        voxels_(voxel_count()),
        colors_(voxel_count()) {}

  Volume(const kinectfusion::Vector3s& resolution, float voxel_size,
         const Eigen::Vector3f& origin, float truncation_distance)
      : Volume(resolution, voxel_size, from_eigen(origin),
               truncation_distance) {}

  [[nodiscard]] kinectfusion::Vector3s resolution() const { return resolution_; }
  [[nodiscard]] float voxel_size() const { return voxel_size_; }

  [[nodiscard]] std::size_t observed_voxel_count() const {
    std::size_t count = 0;
    for (const Voxel& voxel : voxels_) {
      if (voxel.weight > 0.0F) {
        ++count;
      }
    }
    return count;
  }

  [[nodiscard]] VolumeView view() {
    return VolumeView{.voxels = voxels_.data(),
                      .colors = colors_.data(),
                      .resolution = Size3{.x = resolution_.x(),
                                          .y = resolution_.y(),
                                          .z = resolution_.z()},
                      .voxel_size = voxel_size_,
                      .origin = origin_,
                      .truncation_distance = truncation_distance_};
  }

  [[nodiscard]] ConstVolumeView view() const {
    return ConstVolumeView{.voxels = voxels_.data(),
                           .colors = colors_.data(),
                           .resolution = Size3{.x = resolution_.x(),
                                               .y = resolution_.y(),
                                               .z = resolution_.z()},
                           .voxel_size = voxel_size_,
                           .origin = origin_,
                           .truncation_distance = truncation_distance_};
  }

  // Fuses a depth (and optional color) frame into the volume. world_to_camera
  // maps world points into the camera frame; optional live normals enable
  // view-angle observation weighting.
  void integrate_depth_image(
      const image_proc::DepthImage& depth_image,
      const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& world_to_camera,
      const TsdfIntegrationOptions& options = {},
      const image_proc::ColorImage* color_image = nullptr,
      const image_proc::Vector3fImage* normals = nullptr);
  void integrate_depth_image(
      const image_proc::DepthImage& depth_image,
      const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& world_to_camera,
      const image_proc::ColorImage* color_image,
      const image_proc::Vector3fImage* normals = nullptr) {
    integrate_depth_image(depth_image, intrinsics, world_to_camera,
                          TsdfIntegrationOptions{}, color_image, normals);
  }

  // Renders the zero-crossing surface seen from camera_to_world into per-pixel
  // point, normal and color maps (non-finite points where no surface is hit).
  [[nodiscard]] SurfaceMaps raycast(const CameraIntrinsics& intrinsics,
                                    std::size_t width, std::size_t height,
                                    const Eigen::Matrix4f& camera_to_world) const;
  [[nodiscard]] SurfaceMaps raycast(const RaycastOptions& options) const;

 private:
  struct GridSample {
    int base_x{};
    int base_y{};
    int base_z{};
    float tx{};
    float ty{};
    float tz{};
  };

  [[nodiscard]] std::size_t voxel_count() const {
    return resolution_.x() * resolution_.y() * resolution_.z();
  }

  [[nodiscard]] std::size_t index(std::size_t x, std::size_t y,
                                  std::size_t z) const {
    return (z * resolution_.y() + y) * resolution_.x() + x;
  }

  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar>
  [[nodiscard]] bool in_bounds(Scalar x, Scalar y, Scalar z) const {
    return x >= Scalar{0} && y >= Scalar{0} && z >= Scalar{0} &&
           static_cast<std::size_t>(x) < resolution_.x() &&
           static_cast<std::size_t>(y) < resolution_.y() &&
           static_cast<std::size_t>(z) < resolution_.z();
  }

  // Converts to grid coordinates before bounds before checking bounds.
  [[nodiscard]] bool contains(const Eigen::Vector3f& point) const {
    const Eigen::Vector3f local = (point - to_eigen(origin_)) / voxel_size_;
    return in_bounds(local.x(), local.y(), local.z());  // Scalar = float, no truncation
  }

  [[nodiscard]] Voxel& at(std::size_t x, std::size_t y, std::size_t z) {
    return voxels_[index(x, y, z)];
  }
  [[nodiscard]] const Voxel& at(std::size_t x, std::size_t y,
                                std::size_t z) const {
    return voxels_[index(x, y, z)];
  }
  [[nodiscard]] ColorVoxel& color_at(std::size_t x, std::size_t y,
                                     std::size_t z) {
    return colors_[index(x, y, z)];
  }
  [[nodiscard]] const ColorVoxel& color_at(std::size_t x, std::size_t y,
                                           std::size_t z) const {
    return colors_[index(x, y, z)];
  }

  [[nodiscard]] GridSample grid_sample(const Eigen::Vector3f& point) const;

  [[nodiscard]] static float trilinear_weight(const GridSample& sample, int dx,
                                              int dy, int dz);
  
  [[nodiscard]] std::array<Corner, 8> trilinear_corners(const GridSample& s) const;

  // Trilinear TSDF interpolation from whatever corners are valid, reweighting
  // to skip missing/uninitialised ones.
  [[nodiscard]] bool sample_tsdf_available_corners(const Eigen::Vector3f& point,
                                                   float& distance) const;
  // Trilinear TSDF interpolation that requires all eight corners to be valid.
  [[nodiscard]] bool sample_tsdf_valid_corners(const Eigen::Vector3f& point,
                                               float& distance) const;

  // Picks the available corners or valid corners sampler so callers don't
  // repeat the branch on `from_valid_corners`.
  [[nodiscard]] bool sample_tsdf(const Eigen::Vector3f& point, float& distance,
                                 bool from_valid_corners) const;

  [[nodiscard]] bool sample_color(const Eigen::Vector3f& point,
                                  Vec3f& color) const;

  [[nodiscard]] bool sample_normal(const Eigen::Vector3f& point,
                                   Eigen::Vector3f& normal,
                                   bool tsdf_from_valid_corners = false) const;

  [[nodiscard]] static std::uint32_t pixel_from_color(
      const Vec3f& color);

  kinectfusion::Vector3s resolution_;
  float voxel_size_{};
  Vec3f origin_;
  float truncation_distance_{};
  std::vector<Voxel> voxels_;
  std::vector<ColorVoxel> colors_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
