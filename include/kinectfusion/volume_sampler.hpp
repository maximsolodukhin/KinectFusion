#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <type_traits>

namespace kinectfusion {

// TSDF sampling policy
enum class CornerPolicy : std::uint8_t {
  kSkipMissing,
  kRequireAll,
};

//  A thin wrapper around a view with pointer semantics.
template <MemorySpace Space = MemorySpace::kHost>
class VolumeSampler {
 public:
  explicit VolumeSampler(ConstVolumeView<Space> volume) : volume_(volume) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float voxel_size() const {
    return volume_.voxel_size();
  }
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float truncation_distance()
      const {
    return volume_.truncation_distance();
  }

  // Converts to grid coordinates before bounds before checking bounds.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool contains(
      const Vec3f& point) const {
    const Vec3f local = grid_coords(point);
    // Scalar = float, no truncation
    return in_bounds(local.x, local.y, local.z);
  }

  // Trilinear TSDF interpolation at `point`. `kRequireAll` returns nullopt if
  // at least one of surrounding voxels is unobserved, while `kSkipMissing`
  // drops missing/uninitialised corners and reweights the rest.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<float> tsdf(
      const Vec3f& point, CornerPolicy corner_policy) const {
    const Vec3f grid = grid_coords(point);
    if (!in_bounds(grid.x, grid.y, grid.z)) {
      return compat::nullopt;
    }

    const GridSample sample = grid_sample_at(grid);
    if (is_interior(sample.base)) {
      return blend_tsdf(sample.fraction, corner_policy,
                        flat_corners(volume_.voxels, sample.base));
    }
    return blend_tsdf(
        sample.fraction, corner_policy,
        [this, &sample](int offset_x, int offset_y, int offset_z) {
          return find_voxel(sample.base.x + offset_x, sample.base.y + offset_y,
                            sample.base.z + offset_z);
        });
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> color(
      const Vec3f& point) const {
    const GridSample sample = grid_sample_at(grid_coords(point));
    if (is_interior(sample.base)) {
      return blend_color(sample.fraction,
                         flat_corners(volume_.colors, sample.base));
    }
    return blend_color(
        sample.fraction,
        [this, &sample](int offset_x, int offset_y, int offset_z) {
          return find_color_voxel(sample.base.x + offset_x,
                                  sample.base.y + offset_y,
                                  sample.base.z + offset_z);
        });
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> normal(
      const Vec3f& point, CornerPolicy tsdf_corner_policy) const {
    std::array<float, 3> gradient{};
    for (std::size_t axis = 0; axis < gradient.size(); ++axis) {
      // Central difference: one voxel step either way along this axis.
      const Vec3f offset = axis_direction(axis) * volume_.voxel_size();
      const auto plus = tsdf(point + offset, tsdf_corner_policy);
      const auto minus = tsdf(point - offset, tsdf_corner_policy);
      if (!plus || !minus) {
        return compat::nullopt;
      }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      gradient[axis] = *plus - *minus;
    }

    const Vec3f direction = make_vec3f(gradient[0], gradient[1], gradient[2]);
    const float length = norm(direction);
    if (length <= 0.0F) {
      return compat::nullopt;
    }
    return direction / length;
  }

 private:
  // TSDF weight sums below this are treated as no coverage and cause the
  // interpolation helpers to bail out.
  static constexpr float kMinimumTrilinearWeightSum = 1.0e-6F;

  // kVoxelCenterOffset, as a vector, for grid-space math.
  static constexpr Vec3f kCenterOffset =
      make_vec3f(kVoxelCenterOffset, kVoxelCenterOffset, kVoxelCenterOffset);

  // Unit vector along axis (0 x, 1 y, 2 z).
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static constexpr Vec3f
  axis_direction(std::size_t axis) {
    return make_vec3f(axis == 0 ? 1.0F : 0.0F, axis == 1 ? 1.0F : 0.0F,
                      axis == 2 ? 1.0F : 0.0F);
  }

  struct GridSample {
    // may be negative so vec3i
    Vec3i base{};
    Vec3f fraction{};
  };

  // Per-component truncation toward zero; callers pass already-floored values.
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static Vec3i to_vec3i(
      const Vec3f& value) {
    return Vec3i{.x = static_cast<int>(value.x),
                 .y = static_cast<int>(value.y),
                 .z = static_cast<int>(value.z)};
  }

  // The eight corners of one trilinear stencil.
  static constexpr int kStencilCorners = 8;

  // Visits the stencil corners in (x, y, z) order; the accumulation order is
  // load-bearing for bit-identical float sums across host and device.
  template <typename Visit>
  KINECTFUSION_FORCEINLINE_DEVICE static void for_each_corner(
      const Visit& visit) {
    for (int offset_z = 0; offset_z <= 1; ++offset_z) {
      for (int offset_y = 0; offset_y <= 1; ++offset_y) {
        for (int offset_x = 0; offset_x <= 1; ++offset_x) {
          visit(offset_x, offset_y, offset_z);
        }
      }
    }
  }

  // One axis of a trilinear weight: the linear blend between the near voxel
  // (offset 0, weight 1 - fraction) and the far voxel (offset 1, weight
  // fraction).
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static float axis_weight(
      float fraction, int offset) {
    return offset == 0 ? 1.0F - fraction : fraction;
  }

  // Product of the per-axis weights for one corner.
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static float trilinear_weight(
      const Vec3f& fraction, int offset_x, int offset_y, int offset_z) {
    return axis_weight(fraction.x, offset_x) *
           axis_weight(fraction.y, offset_y) *
           axis_weight(fraction.z, offset_z);
  }

  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool in_bounds(Scalar x, Scalar y,
                                                        Scalar z) const {
    return x >= Scalar{0} && y >= Scalar{0} && z >= Scalar{0} &&
           static_cast<std::size_t>(x) < volume_.resolution().x &&
           static_cast<std::size_t>(y) < volume_.resolution().y &&
           static_cast<std::size_t>(z) < volume_.resolution().z;
  }

  // World point in continuous grid coordinates; shared by the bounds check
  // and the sample so the conversion happens once per lookup.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec3f
  grid_coords(const Vec3f& point) const {
    return (point - volume_.origin()) / volume_.voxel_size();
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE static GridSample grid_sample_at(
      const Vec3f& grid_point) {
    const Vec3f grid = grid_point - kCenterOffset;
    const Vec3f floored =
        make_vec3f(std::floor(grid.x), std::floor(grid.y), std::floor(grid.z));
    return GridSample{.base = to_vec3i(floored), .fraction = grid - floored};
  }

  // All eight stencil corners lie inside the volume: one check replaces
  // eight per-corner bounds tests.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool is_interior(
      const Vec3i& base) const {
    const Size3 resolution = volume_.resolution();
    return base.x >= 0 && base.y >= 0 && base.z >= 0 &&
           static_cast<std::size_t>(base.x) + 1 < resolution.x &&
           static_cast<std::size_t>(base.y) + 1 < resolution.y &&
           static_cast<std::size_t>(base.z) + 1 < resolution.z;
  }

  // Interior corner accessor over the flat voxel array: base pointer plus
  // precomputed strides, no per-corner bounds checks or index math.
  template <typename VoxelT>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE auto flat_corners(
      VoxelT* voxels, const Vec3i& base) const {
    const Size3 resolution = volume_.resolution();
    const std::size_t stride_y = resolution.x;
    const std::size_t stride_z = resolution.x * resolution.y;
    const std::size_t base_index =
        (((static_cast<std::size_t>(base.z) * resolution.y) +
          static_cast<std::size_t>(base.y)) *
         resolution.x) +
        static_cast<std::size_t>(base.x);
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    VoxelT* origin = voxels + base_index;
    return
        [origin, stride_y, stride_z](int offset_x, int offset_y, int offset_z) {
          return origin + (static_cast<std::size_t>(offset_z) * stride_z) +
                 (static_cast<std::size_t>(offset_y) * stride_y) +
                 static_cast<std::size_t>(offset_x);
        };
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  // Weighted TSDF blend over the eight stencil corners; `corner_voxel`
  // resolves a corner offset to a voxel (nullptr when out of bounds).
  template <typename CornerVoxel>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<float> blend_tsdf(
      const Vec3f& fraction, CornerPolicy corner_policy,
      const CornerVoxel& corner_voxel) const {
    float accumulated = 0.0F;
    float weight_sum = 0.0F;
    int valid_corners = 0;
    for_each_corner([&](int offset_x, int offset_y, int offset_z) {
      const Voxel* voxel = corner_voxel(offset_x, offset_y, offset_z);
      if (voxel == nullptr || voxel->weight <= 0.0F ||
          !std::isfinite(voxel->distance)) {
        return;
      }
      const float weight =
          trilinear_weight(fraction, offset_x, offset_y, offset_z);
      accumulated += weight * voxel->distance;
      weight_sum += weight;
      ++valid_corners;
    });

    // kRequireAll rejects the sample unless every corner was observed;
    // kSkipMissing keeps the contributing corners and reweights.
    if (corner_policy == CornerPolicy::kRequireAll &&
        valid_corners < kStencilCorners) {
      return compat::nullopt;
    }
    if (weight_sum <= kMinimumTrilinearWeightSum) {
      return compat::nullopt;
    }

    const float distance = accumulated / weight_sum;
    return std::isfinite(distance) ? compat::optional<float>{distance}
                                   : compat::nullopt;
  }

  template <typename CornerVoxel>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> blend_color(
      const Vec3f& fraction, const CornerVoxel& corner_voxel) const {
    Vec3f accumulated{};
    float weight_sum = 0.0F;
    for_each_corner([&](int offset_x, int offset_y, int offset_z) {
      const ColorVoxel* voxel = corner_voxel(offset_x, offset_y, offset_z);
      if (voxel == nullptr || voxel->weight <= 0.0F) {
        return;
      }
      const float weight =
          trilinear_weight(fraction, offset_x, offset_y, offset_z);
      accumulated += weight * voxel->color;
      weight_sum += weight;
    });
    if (weight_sum <= kMinimumTrilinearWeightSum) {
      return compat::nullopt;
    }
    return accumulated / weight_sum;
  }

  // Bounds-checked corner lookups for the boundary shell; nullptr when the
  // corner lies outside the volume.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Voxel* find_voxel(int x, int y,
                                                                 int z) const {
    if (!in_bounds(x, y, z)) {
      return nullptr;
    }
    return &volume_.voxel_at(static_cast<std::size_t>(x),
                             static_cast<std::size_t>(y),
                             static_cast<std::size_t>(z));
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const ColorVoxel* find_color_voxel(
      int x, int y, int z) const {
    if (!in_bounds(x, y, z)) {
      return nullptr;
    }
    return &volume_.color_at(static_cast<std::size_t>(x),
                             static_cast<std::size_t>(y),
                             static_cast<std::size_t>(z));
  }

  ConstVolumeView<Space> volume_;
};

using HostVolumeSampler = VolumeSampler<MemorySpace::kHost>;

// Trivial copy is required because samplers are passed into kernels by value.
template <typename S>
concept TsdfVolumeSampler =
    std::is_trivially_copyable_v<S> &&
    requires(const S sampler, const Vec3f& point, CornerPolicy policy) {
      { sampler.voxel_size() } -> std::same_as<float>;
      { sampler.truncation_distance() } -> std::same_as<float>;
      { sampler.tsdf(point, policy) } -> std::same_as<compat::optional<float>>;
      {
        sampler.normal(point, policy)
      } -> std::same_as<compat::optional<Vec3f>>;
      { sampler.color(point) } -> std::same_as<compat::optional<Vec3f>>;
    };

static_assert(TsdfVolumeSampler<HostVolumeSampler>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP */
