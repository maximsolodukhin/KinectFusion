#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <span>
#include <type_traits>
#include <vector>

namespace kinectfusion {

struct alignas(2 * sizeof(float)) Voxel {
  float distance{1.0F};
  float weight{0.0F};

  // The accumulated weight saturates at max_weight.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Voxel fused(float observed,
                                                     float observation_weight,
                                                     float max_weight) const {
    float weighted_avg =
        weighted_average(distance, weight, observed, observation_weight);
    float truncated_weight =
        compat::min(weight + observation_weight, max_weight);

    return {.distance = weighted_avg, .weight = truncated_weight};
  }
};

struct alignas(4 * sizeof(float)) ColorVoxel {
  Vec3f color{};
  float weight{0.0F};

  // The accumulated weight saturates at max_weight. Weighted average.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE ColorVoxel fused(
      const Vec3f& observed, float observation_weight, float max_weight) const {
    Vec3f weighted_avg =
        weighted_average(color, weight, observed, observation_weight);
    float truncated_weight =
        compat::min(weight + observation_weight, max_weight);

    return {.color = weighted_avg, .weight = truncated_weight};
  }
};

static_assert(sizeof(Voxel) == 2 * sizeof(float) &&
              sizeof(ColorVoxel) == 4 * sizeof(float));

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

// Voxels are sampled at their center, half a cell from the lower corner.
inline constexpr float kVoxelCenterOffset = 0.5F;

struct VolumeGeometry {
  Size3 resolution{};
  float voxel_size{};
  Vec3f origin{};
  float truncation_distance{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t voxel_count() const {
    return resolution.x * resolution.y * resolution.z;
  }

  // World-space center of voxel (x, y, z).
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec3f
  cell_center(std::size_t x, std::size_t y, std::size_t z) const {
    return origin + ((make_vec3f(x, y, z) + make_vec3f(kVoxelCenterOffset,
                                                       kVoxelCenterOffset,
                                                       kVoxelCenterOffset)) *
                     voxel_size);
  }

  friend bool operator==(const VolumeGeometry&,
                         const VolumeGeometry&) = default;
};

// A view into a voxel volume, with pointer semantics: constness is
// shallow(std::span).
template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false>
struct VolumeView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  Pointee<Voxel>* voxels{};
  Pointee<ColorVoxel>* colors{};
  VolumeGeometry geometry{};

  static constexpr MemorySpace kMemorySpace = Space;

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Size3& resolution() const {
    return geometry.resolution;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float voxel_size() const {
    return geometry.voxel_size;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Vec3f& origin() const {
    return geometry.origin;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float truncation_distance() const {
    return geometry.truncation_distance;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t voxel_count() const {
    return geometry.voxel_count();
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec3f
  cell_center(std::size_t x, std::size_t y, std::size_t z) const {
    return geometry.cell_center(x, y, z);
  }

  // Flat element access for coordinate-free sweeps
  [[nodiscard]] std::span<Pointee<Voxel>> voxel_span() const {
    return std::span<Pointee<Voxel>>{voxels, voxel_count()};
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return (((z * geometry.resolution.y) + y) * geometry.resolution.x) + x;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<Voxel>& voxel_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return voxels[index(x, y, z)];
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<ColorVoxel>& color_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return colors[index(x, y, z)];
  }

  // Mutable views convert to read-only views implicitly, like std::span.
  template <bool TargetConst = true>
    requires(TargetConst && !IsConst)
  [[nodiscard]] KINECTFUSION_HOST_DEVICE
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator VolumeView<Space, TargetConst>() const {
    return VolumeView<Space, TargetConst>{
        .voxels = voxels, .colors = colors, .geometry = geometry};
  }
};

using HostVolumeView = VolumeView<MemorySpace::kHost>;
using DeviceVolumeView = VolumeView<MemorySpace::kDevice>;
using ConstHostVolumeView = VolumeView<MemorySpace::kHost, true>;
using ConstDeviceVolumeView = VolumeView<MemorySpace::kDevice, true>;

//  A Buffer must own its elements and expose `data()`
template <MemorySpace Space>
struct SpaceTraits;

template <>
struct SpaceTraits<MemorySpace::kHost> {
  template <typename T>
  using Buffer = std::vector<T>;

  template <typename T>
  [[nodiscard]] static Buffer<T> allocate(std::size_t count) {
    return Buffer<T>(count);
  }
};

// Each specialization is defined in the translation unit that can compile it
template <MemorySpace To, MemorySpace From>
struct Transfer;

template <>
struct Transfer<MemorySpace::kHost, MemorySpace::kHost> {
  static void copy(HostVolumeView destination, ConstHostVolumeView source);
};

// TSDF voxel grid storage in one memory space: owns the buffers, hands out
// views, and copies across spaces on explicit request only. All sampling and
// integration logic lives with the classes operating on views.
template <MemorySpace Space>
class BasicVolume {
 public:
  // Throws std::invalid_argument
  explicit BasicVolume(const VolumeGeometry& geometry)
      : geometry_(validated(geometry)),
        voxels_(SpaceTraits<Space>::template allocate<Voxel>(
            geometry_.voxel_count())),
        colors_(SpaceTraits<Space>::template allocate<ColorVoxel>(
            geometry_.voxel_count())) {}

  [[nodiscard]] const VolumeGeometry& geometry() const { return geometry_; }

  // Explicit cross-space copy;
  // The only place volume data ever moves between memory spaces.
  // Throws std::invalid_argument on geometry mismatch
  // TSDF data is only meaningful under the geometry it was integrated with.
  template <MemorySpace From>
  void copy_from(const BasicVolume<From>& source) {
    require(geometry_ == source.geometry(),
            "Volume copy requires matching geometry");
    Transfer<Space, From>::copy(view(), source.view());
  }

  [[nodiscard]] std::size_t observed_voxel_count() const {
    std::size_t count = 0;
    for (const Voxel& voxel : voxels_) {
      if (voxel.weight > 0.0F) {
        ++count;
      }
    }
    return count;
  }

  [[nodiscard]] VolumeView<Space> view() {
    return view_as<VolumeView<Space>>(*this);
  }
  [[nodiscard]] VolumeView<Space, true> view() const {
    return view_as<VolumeView<Space, true>>(*this);
  }

 private:
  // Self's constness select the view type via the caller.
  template <typename ViewT, typename Self>
  [[nodiscard]] static ViewT view_as(Self& self) {
    return ViewT{.voxels = self.voxels_.data(),
                 .colors = self.colors_.data(),
                 .geometry = self.geometry_};
  }

  [[nodiscard]] static VolumeGeometry validated(VolumeGeometry geometry) {
    require(geometry.resolution.x > 0 && geometry.resolution.y > 0 &&
                geometry.resolution.z > 0,
            "Volume resolution must be positive");
    require(geometry.voxel_size > 0.0F, "Voxel size must be positive");
    require(geometry.truncation_distance > 0.0F,
            "Truncation distance must be positive");
    require(all_finite(geometry.origin), "Volume origin must be finite");
    return geometry;
  }

  VolumeGeometry geometry_;
  typename SpaceTraits<Space>::template Buffer<Voxel> voxels_;
  typename SpaceTraits<Space>::template Buffer<ColorVoxel> colors_;
};

using HostVolume = BasicVolume<MemorySpace::kHost>;
using DeviceVolume = BasicVolume<MemorySpace::kDevice>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
