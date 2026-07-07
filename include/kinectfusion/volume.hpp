#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <type_traits>
#include <vector>

namespace kinectfusion {

struct Corner {
  int x, y, z;
  float weight;
};

struct Voxel {
  float distance{1.0F};
  float weight{0.0F};
};

struct ColorVoxel {
  Vec3f color{};
  float weight{0.0F};
};

// A voxel cube has eight corners; used both as the trilinear stencil size
// and for the `Corner` arrays returned by `Volume::trilinear_corners`.
inline constexpr std::size_t trilinear_corner_count = 8;

// Defaults for the TSDF integration and raycast options structs below. Kept
// here so callers can refer to them by name instead of copying the literals.
inline constexpr float default_tum_depth_scale = 5000.0F;
inline constexpr float default_tsdf_max_weight = 196.0F;
inline constexpr float default_min_depth_meters = 0.4F;
inline constexpr float default_max_depth_meters = 8.0F;
inline constexpr float default_truncation_distance_scale = 0.01F;

struct SurfaceMaps {
  image_proc::Vector3fImage points;
  image_proc::Vector3fImage normals;
  image_proc::ColorImage colors;
};

template <MemorySpace Space = MemorySpace::Host>
struct SurfaceMapsView {
  image_proc::ImageView<Vec3f, Space> points;
  image_proc::ImageView<Vec3f, Space> normals;
  image_proc::ImageView<std::uint32_t, Space> colors;

  static constexpr MemorySpace memory_space = Space;
};

template <MemorySpace Space = MemorySpace::Host>
struct ConstSurfaceMapsView {
  image_proc::ImageView<const Vec3f, Space> points;
  image_proc::ImageView<const Vec3f, Space> normals;
  image_proc::ImageView<const std::uint32_t, Space> colors;

  static constexpr MemorySpace memory_space = Space;
};

using HostSurfaceMapsView = SurfaceMapsView<MemorySpace::Host>;
using DeviceSurfaceMapsView = SurfaceMapsView<MemorySpace::Device>;
using ConstHostSurfaceMapsView = ConstSurfaceMapsView<MemorySpace::Host>;
using ConstDeviceSurfaceMapsView = ConstSurfaceMapsView<MemorySpace::Device>;

[[nodiscard]] inline HostSurfaceMapsView view(SurfaceMaps& maps) {
  return HostSurfaceMapsView{.points = maps.points.view(),
                             .normals = maps.normals.view(),
                             .colors = maps.colors.view()};
}

[[nodiscard]] inline ConstHostSurfaceMapsView view(const SurfaceMaps& maps) {
  return ConstHostSurfaceMapsView{.points = maps.points.view(),
                                  .normals = maps.normals.view(),
                                  .colors = maps.colors.view()};
}

struct TsdfIntegrationOptions {
  float depth_scale{default_tum_depth_scale};
  float observation_weight{1.0F};
  float max_weight{default_tsdf_max_weight};
  float min_depth{default_min_depth_meters};
  float max_depth{default_max_depth_meters};
  bool projective_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{default_truncation_distance_scale};
  bool view_angle_weighting{true};
};

struct RaycastOptions {
  CameraIntrinsics intrinsics{};
  std::size_t width{};
  std::size_t height{};
  Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};
  float min_depth{default_min_depth_meters};
  float max_depth{default_max_depth_meters};
  float step_scale{1.0F};
  bool tsdf_from_valid_corners{false};
};

template <MemorySpace Space = MemorySpace::Host>
struct VolumeView {
  Voxel* voxels{};
  ColorVoxel* colors{};
  Size3 resolution{};
  float voxel_size{};
  Vec3f origin{};
  float truncation_distance{};

  static constexpr MemorySpace memory_space = Space;

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return (((z * resolution.y) + y) * resolution.x) + x;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Voxel& voxel_at(std::size_t x,
                                                         std::size_t y,
                                                         std::size_t z) {
    return voxels[index(x, y, z)];
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE ColorVoxel& color_at(std::size_t x,
                                                              std::size_t y,
                                                              std::size_t z) {
    return colors[index(x, y, z)];
  }
};

template <MemorySpace Space = MemorySpace::Host>
struct ConstVolumeView {
  const Voxel* voxels{};
  const ColorVoxel* colors{};
  Size3 resolution{};
  float voxel_size{};
  Vec3f origin{};
  float truncation_distance{};

  static constexpr MemorySpace memory_space = Space;

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return (((z * resolution.y) + y) * resolution.x) + x;
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

using HostVolumeView = VolumeView<MemorySpace::Host>;
using DeviceVolumeView = VolumeView<MemorySpace::Device>;
using ConstHostVolumeView = ConstVolumeView<MemorySpace::Host>;
using ConstDeviceVolumeView = ConstVolumeView<MemorySpace::Device>;

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

  [[nodiscard]] kinectfusion::Vector3s resolution() const {
    return resolution_;
  }
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

  [[nodiscard]] HostVolumeView view() {
    return HostVolumeView{.voxels = voxels_.data(),
                          .colors = colors_.data(),
                          .resolution = Size3{.x = resolution_.x(),
                                              .y = resolution_.y(),
                                              .z = resolution_.z()},
                          .voxel_size = voxel_size_,
                          .origin = origin_,
                          .truncation_distance = truncation_distance_};
  }

  [[nodiscard]] ConstHostVolumeView view() const {
    return ConstHostVolumeView{.voxels = voxels_.data(),
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
  [[nodiscard]] SurfaceMaps raycast(
      const CameraIntrinsics& intrinsics, std::size_t width, std::size_t height,
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
    return (((z * resolution_.y()) + y) * resolution_.x()) + x;
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
    return in_bounds(local.x(), local.y(),
                     local.z());  // Scalar = float, no truncation
  }

  [[nodiscard]] Voxel& at(std::size_t x, std::size_t y, std::size_t z) {
    return voxels_[index(x, y, z)];
  }
  [[nodiscard]] const Voxel& at(std::size_t x, std::size_t y,
                                std::size_t z) const {
    return voxels_[index(x, y, z)];
  }
  // Corner-taking overloads. Corners carry signed grid coordinates because
  // they may be one-past-the-edge during trilinear sampling; the callers of
  // these overloads guard with `in_bounds` first, so the cast to unsigned is
  // well-defined here.
  [[nodiscard]] Voxel& at(const Corner& corner) {
    return at(static_cast<std::size_t>(corner.x),
              static_cast<std::size_t>(corner.y),
              static_cast<std::size_t>(corner.z));
  }
  [[nodiscard]] const Voxel& at(const Corner& corner) const {
    return at(static_cast<std::size_t>(corner.x),
              static_cast<std::size_t>(corner.y),
              static_cast<std::size_t>(corner.z));
  }
  [[nodiscard]] ColorVoxel& color_at(std::size_t x, std::size_t y,
                                     std::size_t z) {
    return colors_[index(x, y, z)];
  }
  [[nodiscard]] const ColorVoxel& color_at(std::size_t x, std::size_t y,
                                           std::size_t z) const {
    return colors_[index(x, y, z)];
  }
  [[nodiscard]] ColorVoxel& color_at(const Corner& corner) {
    return color_at(static_cast<std::size_t>(corner.x),
                    static_cast<std::size_t>(corner.y),
                    static_cast<std::size_t>(corner.z));
  }
  [[nodiscard]] const ColorVoxel& color_at(const Corner& corner) const {
    return color_at(static_cast<std::size_t>(corner.x),
                    static_cast<std::size_t>(corner.y),
                    static_cast<std::size_t>(corner.z));
  }

  [[nodiscard]] GridSample grid_sample(const Eigen::Vector3f& point) const;

  [[nodiscard]] static float trilinear_weight(const GridSample& sample,
                                              int offset_x, int offset_y,
                                              int offset_z);

  [[nodiscard]] static std::array<Corner, trilinear_corner_count>
  trilinear_corners(const GridSample& sample);

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

  [[nodiscard]] static std::uint32_t pixel_from_color(const Vec3f& color);

  kinectfusion::Vector3s resolution_;
  float voxel_size_{};
  Vec3f origin_;
  float truncation_distance_{};
  std::vector<Voxel> voxels_;
  std::vector<ColorVoxel> colors_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
