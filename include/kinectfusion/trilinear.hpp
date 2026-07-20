#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_TRILINEAR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_TRILINEAR_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
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

// The sampling contract for surface rendering. A volume representation
// plugs into raycasting through it. Samplers cross into kernels by value,
// so they must be trivially copyable.
template <typename S>
concept TsdfVolumeSampler = std::is_trivially_copyable_v<S> && requires {
  { S::kHasColor } -> std::convertible_to<bool>;
} && requires(const S sampler, const Vec3f& point, CornerPolicy policy) {
  { sampler.voxel_size() } -> std::same_as<float>;
  { sampler.truncation_distance() } -> std::same_as<float>;
  { sampler.tsdf(point, policy) } -> std::same_as<compat::optional<float>>;
  { sampler.normal(point, policy) } -> std::same_as<compat::optional<Vec3f>>;
  { sampler.color(point) } -> std::same_as<compat::optional<Vec3f>>;
};

// Maps a 2^3 stencil offset to a voxel pointer. Returns nullptr when the
// corner is out of bounds or unallocated.
template <typename F>
concept TsdfCornerLookup =
    std::invocable<F, int, int, int> &&
    std::is_pointer_v<std::invoke_result_t<F, int, int, int>> &&
    TsdfVoxel<std::remove_const_t<
        std::remove_pointer_t<std::invoke_result_t<F, int, int, int>>>>;

template <typename F>
concept ColorCornerLookup =
    std::invocable<F, int, int, int> && requires(const F& lookup) {
      { lookup(0, 0, 0)->color } -> std::convertible_to<Vec3f>;
      { lookup(0, 0, 0)->weight } -> std::convertible_to<float>;
    };

// The trilinear stencil shared by all samplers. Storage only supplies a
// corner_voxel(dx, dy, dz) -> const Voxel* callable.
namespace trilinear {

// Weight sums below this count as no coverage: the helpers return nullopt.
inline constexpr float kMinimumWeightSum = 1.0e-6F;

inline constexpr int kStencilCorners = 8;

// kVoxelCenterOffset, as a vector, for grid-space math.
inline constexpr Vec3f kCenterOffset =
    make_vec3f(kVoxelCenterOffset, kVoxelCenterOffset, kVoxelCenterOffset);

// Unit vector along axis (0 x, 1 y, 2 z).
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f axis_direction(
    std::size_t axis) {
  return make_vec3f(axis == 0 ? 1.0F : 0.0F, axis == 1 ? 1.0F : 0.0F,
                    axis == 2 ? 1.0F : 0.0F);
}

struct GridSample {
  // may be negative so vec3i
  Vec3i base{};
  Vec3f fraction{};
};

// Per-component truncation toward zero; callers pass already-floored values.
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE Vec3i
to_vec3i(const Vec3f& value) {
  return Vec3i{.x = static_cast<int>(value.x),
               .y = static_cast<int>(value.y),
               .z = static_cast<int>(value.z)};
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE inline GridSample grid_sample_at(
    const Vec3f& grid_point) {
  const Vec3f grid = grid_point - kCenterOffset;
  const Vec3f floored =
      make_vec3f(std::floor(grid.x), std::floor(grid.y), std::floor(grid.z));
  return GridSample{.base = to_vec3i(floored), .fraction = grid - floored};
}

// Clips stencil corners to `resolution` and forwards in-bounds corners to
// `find`. Out-of-bounds corners resolve to nullptr.
template <typename Find>
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE auto clipped_corners(
    const Size3& resolution, const Vec3i& base, const Find& find) {
  return [resolution, base, find](int offset_x, int offset_y, int offset_z) {
    using Pointer = decltype(find(std::size_t{}, std::size_t{}, std::size_t{}));

    const auto corner_x = static_cast<long long>(base.x) + offset_x;
    const auto corner_y = static_cast<long long>(base.y) + offset_y;
    const auto corner_z = static_cast<long long>(base.z) + offset_z;

    if (!resolution.contains(corner_x, corner_y, corner_z)) {
      return static_cast<Pointer>(nullptr);
    }

    return find(static_cast<std::size_t>(corner_x),
                static_cast<std::size_t>(corner_y),
                static_cast<std::size_t>(corner_z));
  };
}

// Visits the stencil corners in (x, y, z) order. Do not change the order:
// host and device float sums must stay bit-identical.
template <std::invocable<int, int, int> Visit>
KINECTFUSION_FORCEINLINE_DEVICE void for_each_corner(const Visit& visit) {
  for (int offset_z = 0; offset_z <= 1; ++offset_z) {
    for (int offset_y = 0; offset_y <= 1; ++offset_y) {
      for (int offset_x = 0; offset_x <= 1; ++offset_x) {
        visit(offset_x, offset_y, offset_z);
      }
    }
  }
}

// One axis of a trilinear weight: 1 - fraction at offset 0, fraction at
// offset 1.
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float axis_weight(float fraction,
                                                                int offset) {
  return offset == 0 ? 1.0F - fraction : fraction;
}

// Product of the per-axis weights for one corner.
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float weight(
    const Vec3f& fraction, int offset_x, int offset_y, int offset_z) {
  return axis_weight(fraction.x, offset_x) * axis_weight(fraction.y, offset_y) *
         axis_weight(fraction.z, offset_z);
}

// Weighted TSDF blend over the eight stencil corners.
template <TsdfCornerLookup CornerVoxel>
[[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<float> blend_tsdf(
    const Vec3f& fraction, CornerPolicy corner_policy,
    const CornerVoxel& corner_voxel) {
  float accumulated = 0.0F;
  float weight_sum = 0.0F;
  int valid_corners = 0;
  for_each_corner([&](int offset_x, int offset_y, int offset_z) {
    const auto* voxel = corner_voxel(offset_x, offset_y, offset_z);
    if (voxel == nullptr) {
      return;
    }

    // By value: one wide load instead of per-field narrow loads.
    const auto corner = *voxel;  // Do not remove dereference.
    if (!corner.observed() || !corner.finite_distance()) {
      return;
    }

    const float corner_weight = weight(fraction, offset_x, offset_y, offset_z);
    accumulated += corner_weight * corner.tsdf();
    weight_sum += corner_weight;
    ++valid_corners;
  });

  // kRequireAll rejects the sample unless all corners are observed.
  // kSkipMissing keeps the valid corners and reweights.
  if (corner_policy == CornerPolicy::kRequireAll &&
      valid_corners < kStencilCorners) {
    return compat::nullopt;
  }
  if (weight_sum <= kMinimumWeightSum) {
    return compat::nullopt;
  }

  const float distance = accumulated / weight_sum;
  return std::isfinite(distance) ? compat::optional<float>{distance}
                                 : compat::nullopt;
}

template <ColorCornerLookup CornerVoxel>
[[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> blend_color(
    const Vec3f& fraction, const CornerVoxel& corner_voxel) {
  Vec3f accumulated{};
  float weight_sum = 0.0F;

  for_each_corner([&](int offset_x, int offset_y, int offset_z) {
    const auto* voxel = corner_voxel(offset_x, offset_y, offset_z);
    if (voxel == nullptr) {
      return;
    }

    const auto corner = *voxel;
    if (corner.weight <= 0.0F) {
      return;
    }

    const float corner_weight = weight(fraction, offset_x, offset_y, offset_z);
    accumulated += corner_weight * corner.color;
    weight_sum += corner_weight;
  });

  if (weight_sum <= kMinimumWeightSum) {
    return compat::nullopt;
  }

  return accumulated / weight_sum;
}

// Analytic gradient of the trilinear blend from the cell corner values.
// Direction only: uniform scaling cancels under normalization.
[[nodiscard]] KINECTFUSION_HOST_DEVICE inline Vec3f cell_gradient(
    const std::array<std::array<std::array<float, 2>, 2>, 2>& corners,
    const Vec3f& fraction) {
  const auto lerp2 = [](float low, float high, float mix) {
    return low + (mix * (high - low));
  };
  // this ugly crap is faster for some reason than a loop
  const float grad_x =
      lerp2(lerp2(corners[0][0][1] - corners[0][0][0],
                  corners[0][1][1] - corners[0][1][0], fraction.y),
            lerp2(corners[1][0][1] - corners[1][0][0],
                  corners[1][1][1] - corners[1][1][0], fraction.y),
            fraction.z);

  const float grad_y =
      lerp2(lerp2(corners[0][1][0] - corners[0][0][0],
                  corners[0][1][1] - corners[0][0][1], fraction.x),
            lerp2(corners[1][1][0] - corners[1][0][0],
                  corners[1][1][1] - corners[1][0][1], fraction.x),
            fraction.z);

  const float grad_z =
      lerp2(lerp2(corners[1][0][0] - corners[0][0][0],
                  corners[1][0][1] - corners[0][0][1], fraction.x),
            lerp2(corners[1][1][0] - corners[0][1][0],
                  corners[1][1][1] - corners[0][1][1], fraction.x),
            fraction.y);

  return make_vec3f(grad_x, grad_y, grad_z);
}

// Central-difference surface normal through any sampler's tsdf().
template <TsdfVolumeSampler Sampler>
[[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> central_normal(
    const Sampler& sampler, const Vec3f& point, CornerPolicy policy) {
  std::array<float, 3> gradient{};
  for (std::size_t axis = 0; axis < gradient.size(); ++axis) {
    // Central difference: one voxel step either way along this axis.
    const Vec3f offset = axis_direction(axis) * sampler.voxel_size();
    const auto plus = sampler.tsdf(point + offset, policy);
    const auto minus = sampler.tsdf(point - offset, policy);

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

}  // namespace trilinear
}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_TRILINEAR_HPP */
