#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace kinectfusion {

struct Corner {
  int x, y, z;
  float weight;
};

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

// A voxel cube has eight corners; used both as the trilinear stencil size
// and for the `Corner` arrays returned by `Volume::trilinear_corners`.
inline constexpr std::size_t kTrilinearCornerCount = 8;

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
  CameraIntrinsics intrinsics{};
  std::size_t width{};
  std::size_t height{};
  Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  float step_scale{1.0F};
  CornerPolicy tsdf_corner_policy{CornerPolicy::kSkipMissing};
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

  // Fuses one depth-camera observation into the volume.
  void integrate_depth_image(const DepthFrame& frame,
                             const TsdfIntegrationOptions& options = {});

  // Renders the zero-crossing surface seen from camera_to_world into per-pixel
  // point, normal and color maps (non-finite points where no surface is hit).
  [[nodiscard]] SurfaceMaps raycast(
      const CameraIntrinsics& intrinsics, std::size_t width, std::size_t height,
      const Eigen::Matrix4f& camera_to_world) const;
  [[nodiscard]] SurfaceMaps raycast(const RaycastOptions& options) const;

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

  struct GridSample {
    // may be negative so vec3i
    Eigen::Vector3i base = Eigen::Vector3i::Zero();
    Eigen::Vector3f fraction = Eigen::Vector3f::Zero();
  };

  struct IntegrationContext {
    const DepthFrame* frame;
    const TsdfIntegrationOptions* options;
    Eigen::Matrix3f rotation;
    Eigen::Vector3f translation;
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
    // Scalar = float, no truncation
    return in_bounds(local.x(), local.y(), local.z());
  }

  // Write access for the integration sweep, which iterates the grid and is
  // in bounds by construction.
  [[nodiscard]] Voxel& at(std::size_t x, std::size_t y, std::size_t z) {
    return voxels_[index(x, y, z)];
  }
  [[nodiscard]] ColorVoxel& color_at(std::size_t x, std::size_t y,
                                     std::size_t z) {
    return colors_[index(x, y, z)];
  }

  // Flat index of a trilinear corner, or nullopt if outside.
  [[nodiscard]] std::optional<std::size_t> corner_index(
      const Corner& corner) const {
    if (!in_bounds(corner.x, corner.y, corner.z)) {
      return std::nullopt;
    }
    return index(static_cast<std::size_t>(corner.x),
                 static_cast<std::size_t>(corner.y),
                 static_cast<std::size_t>(corner.z));
  }

  // Bounds-checked corner lookups for trilinear sampling; nullptr when the
  // corner lies outside the volume.
  [[nodiscard]] const Voxel* find_voxel(const Corner& corner) const {
    const auto flat = corner_index(corner);
    return flat ? &voxels_[*flat] : nullptr;
  }
  [[nodiscard]] const ColorVoxel* find_color_voxel(const Corner& corner) const {
    const auto flat = corner_index(corner);
    return flat ? &colors_[*flat] : nullptr;
  }

  [[nodiscard]] GridSample grid_sample(const Eigen::Vector3f& point) const;

  [[nodiscard]] static std::array<Corner, kTrilinearCornerCount>
  trilinear_corners(const GridSample& sample);

  // Trilinear TSDF interpolation at `point`. `kRequireAll` returns nullopt if
  // at least one of surrounding voxels is unobserved, while `kSkipMissing`
  // drops missing/uninitialised corners and reweights the rest.
  [[nodiscard]] std::optional<float> sample_tsdf(
      const Eigen::Vector3f& point, CornerPolicy corner_policy) const;

  [[nodiscard]] std::optional<Vec3f> sample_color(
      const Eigen::Vector3f& point) const;

  [[nodiscard]] std::optional<Vec3f> sample_normal(
      const Eigen::Vector3f& point,
      CornerPolicy tsdf_corner_policy = CornerPolicy::kSkipMissing) const;

  // Per-voxel TSDF integration helpers, defined with integrate_depth_image.
  [[nodiscard]] Eigen::Vector3f cell_center(std::size_t x, std::size_t y,
                                            std::size_t z) const;
  void integrate_voxel(const IntegrationContext& context, std::size_t x,
                       std::size_t y, std::size_t z);

  // Raycast helpers: march one ray to the first front-to-back zero crossing,
  // then sample normal/colour at the hit into the output maps.
  [[nodiscard]] std::optional<Eigen::Vector3f> find_zero_crossing(
      const Eigen::Vector3f& origin, const Eigen::Vector3f& direction,
      const RaycastOptions& options) const;
  void write_surface_sample(SurfaceMaps& maps, std::size_t col, std::size_t row,
                            const Eigen::Vector3f& surface,
                            CornerPolicy tsdf_corner_policy) const;

  kinectfusion::Vector3s resolution_;
  float voxel_size_{};
  Vec3f origin_;
  float truncation_distance_{};
  std::vector<Voxel> voxels_;
  std::vector<ColorVoxel> colors_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
