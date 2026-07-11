#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace kinectfusion {

struct Voxel {
  float distance{1.0F};
  float weight{0.0F};

  // The accumulated weight saturates at max_weight.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Voxel fused(float observed,
                                                     float observation_weight,
                                                     float max_weight) const {
    return {.distance = weighted_average(distance, weight, observed,
                                         observation_weight),
            .weight = std::min(weight + observation_weight, max_weight)};
  }
};

struct ColorVoxel {
  Vec3f color{};
  float weight{0.0F};

  // The accumulated weight saturates at max_weight. Weighted average.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE ColorVoxel fused(
      const Vec3f& observed, float observation_weight, float max_weight) const {
    return {
        .color = weighted_average(color, weight, observed, observation_weight),
        .weight = std::min(weight + observation_weight, max_weight)};
  }
};

// Defaults for the TSDF integration and raycast options structs below; the
// shared sensor defaults (depth scale and range) live in rgbd.hpp.
inline constexpr float kDefaultTsdfMaxWeight = 196.0F;
inline constexpr float kDefaultTruncationDistanceScale = 0.01F;

struct SurfaceMaps {
  image_proc::Vector3fImage points;
  image_proc::Vector3fImage normals;
  image_proc::ColorImage colors;
};

// IsConst toggles the pointee types, so the mutable and read-only views share
// one definition. Views have pointer semantics: constness is shallow, like
// std::span.
template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false>
struct SurfaceMapsView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  image_proc::ImageView<Pointee<Vec3f>, Space> points;
  image_proc::ImageView<Pointee<Vec3f>, Space> normals;
  image_proc::ImageView<Pointee<std::uint32_t>, Space> colors;

  static constexpr MemorySpace kMemorySpace = Space;
};

using HostSurfaceMapsView = SurfaceMapsView<MemorySpace::kHost>;
using DeviceSurfaceMapsView = SurfaceMapsView<MemorySpace::kDevice>;
using ConstHostSurfaceMapsView = SurfaceMapsView<MemorySpace::kHost, true>;
using ConstDeviceSurfaceMapsView = SurfaceMapsView<MemorySpace::kDevice, true>;

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
  float depth_scale{kDefaultTumDepthScale};
  float observation_weight{1.0F};
  float max_weight{kDefaultTsdfMaxWeight};
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  bool projective_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{kDefaultTruncationDistanceScale};
  bool view_angle_weighting{true};
};

// One depth-camera observation to fuse into the volume: the depth image is
// required, colour and live normals are optional (normals enable view-angle
// observation weighting). world_to_camera maps world points into the camera
// frame.
struct DepthFrame {
  const image_proc::DepthImage* depth{};
  const image_proc::ColorImage* color{};
  const image_proc::Vector3fImage* normals{};
  CameraIntrinsics intrinsics{};
  Eigen::Matrix4f world_to_camera{Eigen::Matrix4f::Identity()};
};

// TSDF sampling policy
enum class CornerPolicy : std::uint8_t {
  kSkipMissing,
  kRequireAll,
};

struct RaycastOptions {
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  float step_scale{1.0F};
  CornerPolicy tsdf_corner_policy{CornerPolicy::kSkipMissing};
};

struct RaycastCamera {
  CameraIntrinsics intrinsics{};
  std::size_t width{};
  std::size_t height{};
  Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};
};

// A view into a voxel volume, with pointer semantics: constness is
// shallow(std::span)
template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false>
struct VolumeView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  Pointee<Voxel>* voxels{};
  Pointee<ColorVoxel>* colors{};
  Size3 resolution{};
  float voxel_size{};
  Vec3f origin{};
  float truncation_distance{};

  static constexpr MemorySpace kMemorySpace = Space;

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return (((z * resolution.y) + y) * resolution.x) + x;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<Voxel>& voxel_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return voxels[index(x, y, z)];
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<ColorVoxel>& color_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return colors[index(x, y, z)];
  }
};

using HostVolumeView = VolumeView<MemorySpace::kHost>;
using DeviceVolumeView = VolumeView<MemorySpace::kDevice>;
using ConstHostVolumeView = VolumeView<MemorySpace::kHost, true>;
using ConstDeviceVolumeView = VolumeView<MemorySpace::kDevice, true>;

class Volume {
 public:
  Volume(kinectfusion::Vector3s resolution, float voxel_size,
         const Vec3f& origin, float truncation_distance)
      : resolution_(std::move(resolution)),
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

  [[nodiscard]] HostVolumeView view() { return view_as<HostVolumeView>(*this); }
  [[nodiscard]] ConstHostVolumeView view() const {
    return view_as<ConstHostVolumeView>(*this);
  }

 private:
  // Self's constness select the view type via the caller.
  template <typename ViewT, typename Self>
  [[nodiscard]] static ViewT view_as(Self& self) {
    return ViewT{.voxels = self.voxels_.data(),
                 .colors = self.colors_.data(),
                 .resolution = to_size3(self.resolution_),
                 .voxel_size = self.voxel_size_,
                 .origin = self.origin_,
                 .truncation_distance = self.truncation_distance_};
  }

  [[nodiscard]] std::size_t voxel_count() const {
    return resolution_.x() * resolution_.y() * resolution_.z();
  }

  kinectfusion::Vector3s resolution_;
  float voxel_size_{};
  Vec3f origin_;
  float truncation_distance_{};
  std::vector<Voxel> voxels_;
  std::vector<ColorVoxel> colors_;
};

class TsdfIntegrator {
 public:
  // Throws std::invalid_argument
  explicit TsdfIntegrator(TsdfIntegrationOptions options = {});

  // Fuses one depth-camera observation into the volume.
  void integrate(Volume& volume, const DepthFrame& frame) const;

 private:
  TsdfIntegrationOptions options_;
};

class Raycaster {
 public:
  // Throws std::invalid_argument
  explicit Raycaster(RaycastOptions options = {});

  // Renders the zero-crossing surface seen from `camera` into per-pixel
  // point, normal and color maps (non-finite points where no surface is hit).
  [[nodiscard]] SurfaceMaps raycast(const Volume& volume,
                                    const RaycastCamera& camera) const;

 private:
  RaycastOptions options_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
