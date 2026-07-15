#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP

#include <array>
#include <cmath>
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
    const Vec3f local = (point - volume_.origin()) / volume_.voxel_size();
    // Scalar = float, no truncation
    return in_bounds(local.x, local.y, local.z);
  }

  // Trilinear TSDF interpolation at `point`. `kRequireAll` returns nullopt if
  // at least one of surrounding voxels is unobserved, while `kSkipMissing`
  // drops missing/uninitialised corners and reweights the rest.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<float> tsdf(
      const Vec3f& point, CornerPolicy corner_policy) const {
    if (!contains(point)) {
      return compat::nullopt;
    }

    const GridSample sample = grid_sample(point);
    float accumulated = 0.0F;
    float weight_sum = 0.0F;
    for (const Corner& corner : trilinear_corners(sample)) {
      const Voxel* voxel = find_voxel(corner);
      if (voxel != nullptr && voxel->weight > 0.0F &&
          std::isfinite(voxel->distance)) {
        accumulated += corner.weight * voxel->distance;
        weight_sum += corner.weight;
        continue;
      }
      // Corner is out of bounds or unobserved. `kRequireAll` fails the
      // sample; otherwise the corner is skipped and the remaining ones
      // reweight.
      if (corner_policy == CornerPolicy::kRequireAll) {
        return compat::nullopt;
      }
    }
    if (weight_sum <= kMinimumTrilinearWeightSum) {
      return compat::nullopt;
    }

    const float distance = accumulated / weight_sum;
    return std::isfinite(distance) ? compat::optional<float>{distance}
                                   : compat::nullopt;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> color(
      const Vec3f& point) const {
    const GridSample sample = grid_sample(point);
    Vec3f accumulated{};
    float weight_sum = 0.0F;
    for (const Corner& corner : trilinear_corners(sample)) {
      const ColorVoxel* voxel = find_color_voxel(corner);
      if (voxel == nullptr || voxel->weight <= 0.0F) {
        continue;
      }
      accumulated += corner.weight * voxel->color;
      weight_sum += corner.weight;
    }
    if (weight_sum <= kMinimumTrilinearWeightSum) {
      return compat::nullopt;
    }
    return accumulated / weight_sum;
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

  // A voxel cube has eight corners; used both as the trilinear stencil size
  // and for the `Corner` arrays returned by `trilinear_corners`.
  static constexpr std::size_t kTrilinearCornerCount = 8;

  struct Corner {
    int x, y, z;
    float weight;
  };

  struct GridSample {
    // may be negative so vec3i
    Vec3i base{};
    Vec3f fraction{};
  };

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

  [[nodiscard]] KINECTFUSION_HOST_DEVICE GridSample
  grid_sample(const Vec3f& point) const {
    const Vec3f grid =
        ((point - volume_.origin()) / volume_.voxel_size()) - kCenterOffset;
    const Vec3f floored =
        make_vec3f(std::floor(grid.x), std::floor(grid.y), std::floor(grid.z));
    return GridSample{.base = Vec3i{.x = static_cast<int>(floored.x),
                                    .y = static_cast<int>(floored.y),
                                    .z = static_cast<int>(floored.z)},
                      .fraction = grid - floored};
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE static std::array<
      Corner, kTrilinearCornerCount>
  trilinear_corners(const GridSample& sample) {
    std::array<Corner, kTrilinearCornerCount> out{};
    std::size_t corner_idx = 0;
    for (int offset_z = 0; offset_z <= 1; ++offset_z) {
      for (int offset_y = 0; offset_y <= 1; ++offset_y) {
        for (int offset_x = 0; offset_x <= 1; ++offset_x) {
          auto weight =
              trilinear_weight(sample.fraction, offset_x, offset_y, offset_z);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
          out[corner_idx++] = {.x = sample.base.x + offset_x,
                               .y = sample.base.y + offset_y,
                               .z = sample.base.z + offset_z,
                               .weight = weight};
        }
      }
    }
    return out;
  }

  // Bounds-checked corner lookups for trilinear sampling; nullptr when the
  // corner lies outside the volume.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Voxel* find_voxel(
      const Corner& corner) const {
    if (!in_bounds(corner.x, corner.y, corner.z)) {
      return nullptr;
    }
    return &volume_.voxel_at(static_cast<std::size_t>(corner.x),
                             static_cast<std::size_t>(corner.y),
                             static_cast<std::size_t>(corner.z));
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const ColorVoxel* find_color_voxel(
      const Corner& corner) const {
    if (!in_bounds(corner.x, corner.y, corner.z)) {
      return nullptr;
    }
    return &volume_.color_at(static_cast<std::size_t>(corner.x),
                             static_cast<std::size_t>(corner.y),
                             static_cast<std::size_t>(corner.z));
  }

  ConstVolumeView<Space> volume_;
};

using HostVolumeSampler = VolumeSampler<MemorySpace::kHost>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP */
