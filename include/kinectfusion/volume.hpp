#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
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

// TSDF of unobserved space
// Fresh volumes are filled with it
inline constexpr float kUnobservedTsdf = 1.0F;

// Fixed-point signed distance in [-1, 1], stored as int16 (ISMAR 2011).
// BasicVoxel keeps its weighted-average math in float through the explicit
// conversions.
struct QuantizedTsdf {
  static constexpr float kScale = 32767.0F;
  // Reads are the hot path (8 corners per sample). Multiply by the
  // reciprocal; kScale is not a power of two.
  static constexpr float kInvScale = 1.0F / kScale;

  std::int16_t raw{};

  QuantizedTsdf() = default;
  KINECTFUSION_HOST_DEVICE explicit QuantizedTsdf(float tsdf)
      : raw(static_cast<std::int16_t>(compat::clamp(tsdf, -1.0F, 1.0F) *
                                      kScale)) {}

  [[nodiscard]] KINECTFUSION_HOST_DEVICE explicit operator float() const {
    return static_cast<float>(raw) * kInvScale;
  }

  static constexpr bool kAlwaysFinite = true;  // clamped on store
};

// Truncated fp32 (bf16). Dequantization is one integer shift, so corner
// reads never use the quarter-rate convert pipe. This puts the 4-byte voxel
// at float-fps parity. Resolution is relative (8-bit mantissa): fine near
// the zero crossing, coarse far from the surface.
struct Bf16Tsdf {
  static constexpr unsigned int kMantissaShift = 16U;
  static constexpr std::uint32_t kRoundBias = 0x7FFFU;
  static constexpr std::uint16_t kUnobservedRaw = 0x3F80U;  // bf16(1.0f)

  std::uint16_t raw{kUnobservedRaw};

  Bf16Tsdf() = default;
  KINECTFUSION_HOST_DEVICE explicit Bf16Tsdf(float tsdf)
      : raw(static_cast<std::uint16_t>(round_shift(
            std::bit_cast<std::uint32_t>(compat::clamp(tsdf, -1.0F, 1.0F))))) {}

  [[nodiscard]] KINECTFUSION_HOST_DEVICE explicit operator float() const {
    // Shift plus bit_cast: full-rate ALU, no cvt instruction.
    return std::bit_cast<float>(static_cast<std::uint32_t>(raw)
                                << kMantissaShift);
  }

  static constexpr bool kAlwaysFinite = true;  // clamped on store

 private:
  // Round-to-nearest-even into the top 16 bits of the fp32 pattern.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE static std::uint32_t round_shift(
      std::uint32_t bits) {
    return (bits + kRoundBias + ((bits >> kMantissaShift) & 1U)) >>
           kMantissaShift;
  }
};

// Fixed-point observation weight in uint16. The rules produce fractional
// angle weights; an integer store would truncate a glancing-angle voxel to
// zero. The scale keeps ~1/256 resolution, and max_weight stays well below
// 65535 / kScale.
struct QuantizedWeight {
  static constexpr float kScale = 256.0F;  // power of two: reciprocal is exact
  static constexpr float kInvScale = 1.0F / kScale;
  static constexpr float kMax = 65535.0F / kScale;

  std::uint16_t raw{};

  QuantizedWeight() = default;
  KINECTFUSION_HOST_DEVICE explicit QuantizedWeight(float weight)
      : raw(static_cast<std::uint16_t>(compat::clamp(weight, 0.0F, kMax) *
                                       kScale)) {}

  [[nodiscard]] KINECTFUSION_HOST_DEVICE explicit operator float() const {
    return static_cast<float>(raw) * kInvScale;
  }
};

// The geometric TSDF voxel, parametrized by the storage of the distance and
// the weight. All consumers read through the float interface tsdf() and
// weight_value(), so a narrower storage swaps in behind the Voxel alias.
template <typename DistanceStore, typename WeightStore>
struct alignas(sizeof(DistanceStore) + sizeof(WeightStore)) BasicVoxel {
  DistanceStore distance{kUnobservedTsdf};
  WeightStore weight{};

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float tsdf() const {
    return static_cast<float>(distance);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float weight_value() const {
    return static_cast<float>(weight);
  }

  // The corner gather tests validity 8 times per sample. The raw integer
  // compare keeps cvt.rn.f32.u16 out of the raycast kernels.
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE bool observed() const {
    if constexpr (requires(WeightStore stored) { stored.raw; }) {
      return weight.raw != 0;
    } else {
      return weight > 0.0F;
    }
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE bool finite_distance() const {
    if constexpr (requires { DistanceStore::kAlwaysFinite; }) {
      return true;
    } else {
      return std::isfinite(static_cast<float>(distance));
    }
  }

  // The accumulated weight saturates at max_weight.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE BasicVoxel
  fused(float observed, float observation_weight, float max_weight) const {
    const float weighted_avg =
        weighted_average(tsdf(), weight_value(), observed, observation_weight);
    const float truncated_weight =
        compat::min(weight_value() + observation_weight, max_weight);

    return {.distance = DistanceStore(weighted_avg),
            .weight = WeightStore(truncated_weight)};
  }
};

// The default geometric voxel. The representation registry selects other
// storages per pipeline (PipelineConfig::voxel).
using Voxel = BasicVoxel<float, float>;                             // 8 bytes
using QuantizedVoxel = BasicVoxel<QuantizedTsdf, QuantizedWeight>;  // 4 bytes
using Bf16Voxel = BasicVoxel<Bf16Tsdf, QuantizedWeight>;            // 4 bytes

// The float read interface of each geometric voxel storage. Voxels cross
// into kernels by value, so they must be trivially copyable.
template <typename V>
concept TsdfVoxel = std::is_trivially_copyable_v<V> && requires(const V voxel) {
  { voxel.tsdf() } -> std::same_as<float>;
  { voxel.weight_value() } -> std::same_as<float>;
  { voxel.observed() } -> std::same_as<bool>;
  { voxel.finite_distance() } -> std::same_as<bool>;
};

static_assert(TsdfVoxel<Voxel>);
static_assert(TsdfVoxel<QuantizedVoxel>);
static_assert(TsdfVoxel<Bf16Voxel>);

struct alignas(4 * sizeof(float)) ColorVoxel {
  Vec3f color{};
  float weight{0.0F};

  // The accumulated weight saturates at max_weight. Weighted average.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE ColorVoxel fused(
      const Vec3f& observed, float observation_weight, float max_weight) const {
    const Vec3f weighted_avg =
        weighted_average(color, weight, observed, observation_weight);

    const float truncated_weight =
        compat::min(weight + observation_weight, max_weight);

    return {.color = weighted_avg, .weight = truncated_weight};
  }
};

static_assert(sizeof(BasicVoxel<float, float>) == 2 * sizeof(float));
static_assert(sizeof(BasicVoxel<QuantizedTsdf, QuantizedWeight>) == 4);
static_assert(std::is_trivially_copyable_v<BasicVoxel<float, float>>);
static_assert(
    std::is_trivially_copyable_v<BasicVoxel<QuantizedTsdf, QuantizedWeight>>);
static_assert(sizeof(Bf16Voxel) == 4 &&
              std::is_trivially_copyable_v<Bf16Voxel>);
static_assert(sizeof(ColorVoxel) == 4 * sizeof(float));

// How the volume stores color. The facet names the color voxel type and
// gates its buffer, fusion, and sampling. Disabled color costs zero bytes
// and zero raycast bandwidth.
struct FloatColorFacet {
  using Voxel = ColorVoxel;  // 16 bytes
  static constexpr bool kEnabled = true;
};

struct NoColorFacet {
  struct Voxel {};  // never allocated or read
  static constexpr bool kEnabled = false;
};

struct SurfaceMaps {
  image_proc::Vector3fImage points;
  image_proc::Vector3fImage normals;
  image_proc::ColorImage colors;

  [[nodiscard]] static SurfaceMaps allocate(std::size_t width,
                                            std::size_t height) {
    return {.points = {width, height},
            .normals = {width, height},
            .colors = {width, height}};
  }
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

  // Mutable views convert to read-only views implicitly, like std::span.
  template <bool TargetConst = true>
    requires(TargetConst && !IsConst)
  [[nodiscard]] KINECTFUSION_HOST_DEVICE
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator SurfaceMapsView<Space, TargetConst>() const {
    return {.points = points, .normals = normals, .colors = colors};
  }
};

template <MemorySpace Space>
using ConstSurfaceMapsView = SurfaceMapsView<Space, true>;

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
    constexpr auto kVoxelCenter =
        make_vec3f(kVoxelCenterOffset, kVoxelCenterOffset, kVoxelCenterOffset);
    const auto voxel_start = make_vec3f(x, y, z);

    return origin + (voxel_start + kVoxelCenter) * voxel_size;
  }

  friend bool operator==(const VolumeGeometry&,
                         const VolumeGeometry&) = default;
};

// A view into a voxel volume, with pointer semantics: constness is
// shallow(std::span).
template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false,
          typename GeomVoxel = Voxel, typename Color = FloatColorFacet>
struct VolumeView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  using GeometryVoxel = GeomVoxel;
  using ColorFacet = Color;

  Pointee<GeomVoxel>* voxels{};
  Pointee<typename Color::Voxel>* colors{};
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
  [[nodiscard]] std::span<Pointee<GeomVoxel>> voxel_span() const {
    return std::span<Pointee<GeomVoxel>>{voxels, voxel_count()};
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return geometry.resolution.flatten(x, y, z);
  }

  // Raw pointers so views can cross into CUDA kernels, like ImageView.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<GeomVoxel>& voxel_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return voxels[index(x, y, z)];
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<typename Color::Voxel>&
  color_at(std::size_t x, std::size_t y, std::size_t z) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return colors[index(x, y, z)];
  }

  // Mutable views convert to read-only views implicitly, like std::span.
  template <bool TargetConst = true>
    requires(TargetConst && !IsConst)
  [[nodiscard]] KINECTFUSION_HOST_DEVICE
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator VolumeView<Space, TargetConst, GeomVoxel, Color>() const {
    return VolumeView<Space, TargetConst, GeomVoxel, Color>{
        .voxels = voxels, .colors = colors, .geometry = geometry};
  }
};

template <MemorySpace Space, typename GeomVoxel = Voxel,
          typename Color = FloatColorFacet>
using ConstVolumeView = VolumeView<Space, true, GeomVoxel, Color>;

using HostVolumeView = VolumeView<MemorySpace::kHost>;
using DeviceVolumeView = VolumeView<MemorySpace::kDevice>;
using ConstHostVolumeView = VolumeView<MemorySpace::kHost, true>;
using ConstDeviceVolumeView = VolumeView<MemorySpace::kDevice, true>;

// What each storage view offers a per-voxel visitor: geometry accessors
// plus coordinate voxel access. Views cross into kernels by value, so they
// must be trivially copyable.
template <typename V>
concept VoxelGridView =
    std::is_trivially_copyable_v<V> && TsdfVoxel<typename V::GeometryVoxel> &&
    requires(const V view, std::size_t index) {
      typename V::ColorFacet;
      { view.resolution() } -> std::same_as<const Size3&>;
      { view.voxel_size() } -> std::same_as<float>;
      { view.origin() } -> std::same_as<const Vec3f&>;
      { view.truncation_distance() } -> std::same_as<float>;
      { view.cell_center(index, index, index) } -> std::same_as<Vec3f>;
      {
        view.voxel_at(index, index, index)
      } -> std::convertible_to<const typename V::GeometryVoxel&>;
    };

// Refinement for contiguous storage: flat sweeps, transfers, and occupancy
// marking index the voxel array directly.
template <typename V>
concept DenseVoxelGridView = VoxelGridView<V> && requires(const V view) {
  { view.voxel_count() } -> std::same_as<std::size_t>;
  { view.voxel_span() };
  view.voxels;
};

static_assert(DenseVoxelGridView<HostVolumeView>);
static_assert(DenseVoxelGridView<ConstHostVolumeView>);
static_assert(DenseVoxelGridView<DeviceVolumeView>);
static_assert(DenseVoxelGridView<ConstDeviceVolumeView>);

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

// Reductions over one memory space's volume views
template <MemorySpace Space>
class VolumeReduction;

template <>
class VolumeReduction<MemorySpace::kHost> {
 public:
  template <DenseVoxelGridView VolumeViewT>
  [[nodiscard]] static std::size_t observed_voxel_count(
      const VolumeViewT& volume) {
    std::size_t count = 0;
    for (const auto& voxel : volume.voxel_span()) {
      if (voxel.weight_value() > 0.0F) {
        ++count;
      }
    }
    return count;
  }
};

using HostVolumeReduction = VolumeReduction<MemorySpace::kHost>;

// Each specialization is defined in a header that the space can compile.
// Copies are generic over the matching storage of the two views.
template <MemorySpace To, MemorySpace From>
struct Transfer;

template <>
struct Transfer<MemorySpace::kHost, MemorySpace::kHost> {
  template <DenseVoxelGridView DstView, DenseVoxelGridView SrcView>
    requires std::same_as<typename DstView::GeometryVoxel,
                          typename SrcView::GeometryVoxel>
  static void copy(DstView destination, SrcView source) {
    std::copy_n(source.voxels, source.voxel_count(), destination.voxels);

    if constexpr (DstView::ColorFacet::kEnabled) {
      std::copy_n(source.colors, source.voxel_count(), destination.colors);
    }
  }
};

// TSDF voxel grid storage in one memory space: owns the buffers, hands out
// views, and copies across spaces on explicit request only. All sampling and
// integration logic lives with the classes operating on views.
template <MemorySpace Space, typename GeomVoxel = Voxel,
          typename Color = FloatColorFacet>
class BasicVolume {
 public:
  // Throws std::invalid_argument
  explicit BasicVolume(const VolumeGeometry& geometry)
      : geometry_(validated(geometry)),
        voxels_(SpaceTraits<Space>::template allocate<GeomVoxel>(
            geometry_.voxel_count())),
        colors_(SpaceTraits<Space>::template allocate<typename Color::Voxel>(
            Color::kEnabled ? geometry_.voxel_count() : 0U)) {}

  [[nodiscard]] const VolumeGeometry& geometry() const { return geometry_; }

  // Explicit cross-space copy;
  // The only place volume data ever moves between memory spaces.
  // Throws std::invalid_argument on geometry mismatch
  // TSDF data is only meaningful under the geometry it was integrated with.
  template <MemorySpace From>
  void copy_from(const BasicVolume<From, GeomVoxel, Color>& source) {
    require(geometry_ == source.geometry(),
            "Volume copy requires matching geometry");
    Transfer<Space, From>::copy(view(), source.view());
  }

  [[nodiscard]] VolumeView<Space, false, GeomVoxel, Color> view() {
    return view_as<VolumeView<Space, false, GeomVoxel, Color>>(*this);
  }
  [[nodiscard]] ConstVolumeView<Space, GeomVoxel, Color> view() const {
    return view_as<ConstVolumeView<Space, GeomVoxel, Color>>(*this);
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
  SpaceTraits<Space>::template Buffer<GeomVoxel> voxels_;
  SpaceTraits<Space>::template Buffer<typename Color::Voxel> colors_;
};

using HostVolume = BasicVolume<MemorySpace::kHost>;
using DeviceVolume = BasicVolume<MemorySpace::kDevice>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
