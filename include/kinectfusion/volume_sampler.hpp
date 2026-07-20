#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP

#include <array>
#include <cstddef>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/trilinear.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <type_traits>

namespace kinectfusion {

namespace detail {
struct CornerAccessProbe {
  template <typename Corners>
  KINECTFUSION_HOST_DEVICE void operator()(const Corners& /*corners*/) const {}
};
}  // namespace detail

// How one storage reaches the 2^3 stencil corners around a base voxel.
// `with_corners` passes a TsdfCornerLookup to the continuation. Thus the
// policy selects the corner path once per sample, and each lookup stays
// branch-free. Policies are the extension point for new storages. A policy
// may also override a whole sampler function (see TrilinearSampler).
template <typename A>
concept CornerAccessPolicy =
    std::is_trivially_copyable_v<A> && VoxelGridView<typename A::View> &&
    requires(const A access, const Vec3i& base) {
      { access.view() } -> std::convertible_to<const typename A::View&>;
      access.with_corners(base, detail::CornerAccessProbe{});
    };

// Corner access for dense voxel grids: an interior fast path over the flat
// voxel array, and clipped per-corner lookups on the boundary shell.
template <typename ViewT>
class DenseCornerAccess {
 public:
  using View = ViewT;

  explicit KINECTFUSION_HOST_DEVICE DenseCornerAccess(const ViewT& view)
      : view_(view) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE const ViewT& view() const {
    return view_;
  }

  template <typename Use>
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE decltype(auto) with_corners(
      const Vec3i& base, const Use& use) const {
    if (is_interior(base)) {
      return use(flat_corners(view_.voxels, base));
    }
    return use(
        trilinear::clipped_corners(view_.resolution(), base, VoxelAt{&view_}));
  }

  template <typename Use>
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE decltype(auto)
  with_color_corners(const Vec3i& base, const Use& use) const {
    if (is_interior(base)) {
      return use(flat_corners(view_.colors, base));
    }
    return use(trilinear::clipped_corners(view_.resolution(), base,
                                          ColorVoxelAt{&view_}));
  }

 private:
  struct VoxelAt {
    const ViewT* view;
    [[nodiscard]] KINECTFUSION_HOST_DEVICE const ViewT::GeometryVoxel*
    operator()(std::size_t x, std::size_t y, std::size_t z) const {
      return &view->voxel_at(x, y, z);
    }
  };

  struct ColorVoxelAt {
    const ViewT* view;
    [[nodiscard]] KINECTFUSION_HOST_DEVICE const ViewT::ColorFacet::Voxel*
    operator()(std::size_t x, std::size_t y, std::size_t z) const {
      return &view->color_at(x, y, z);
    }
  };

  // All eight stencil corners lie inside the volume: one check replaces
  // eight per-corner bounds tests.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool is_interior(
      const Vec3i& base) const {
    const Size3 resolution = view_.resolution();
    return resolution.contains(base.x, base.y, base.z) &&
           resolution.contains(base.x + 1, base.y + 1, base.z + 1);
  }

  // Interior corner accessor over the flat voxel array: no per-corner bounds
  // checks.
  template <typename VoxelT>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE auto flat_corners(
      VoxelT* voxels, const Vec3i& base) const {
    const Size3 resolution = view_.resolution();
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    VoxelT* origin = voxels + resolution.flatten(base.x, base.y, base.z);
    return [origin, resolution](int offset_x, int offset_y, int offset_z) {
      return origin + resolution.flatten(offset_x, offset_y, offset_z);
    };
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  ViewT view_;
};

// The one trilinear sampler: the invariant shell (grid transform, bounds,
// blends, normals) over a corner-access policy. When the policy provides a
// function override, the host uses it instead of the default. Thus an
// exotic storage can replace a whole function, for example with hardware
// texture fetches for tsdf().
template <CornerAccessPolicy Access>
class TrilinearSampler : private Access {
  using ViewT = Access::View;
  using GeomVoxel = ViewT::GeometryVoxel;
  using ColorFacet = ViewT::ColorFacet;

 public:
  // True if color() can produce a value. Raycast shading reads this at
  // compile time.
  static constexpr bool kHasColor = ColorFacet::kEnabled;

  explicit KINECTFUSION_HOST_DEVICE TrilinearSampler(const ViewT& view)
      : Access(view), inv_voxel_size_(1.0F / view.voxel_size()) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float voxel_size() const {
    return this->view().voxel_size();
  }
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float truncation_distance()
      const {
    return this->view().truncation_distance();
  }

  // Trilinear TSDF interpolation at `point`. `kRequireAll` returns nullopt
  // when a corner voxel is unobserved. `kSkipMissing` drops missing corners
  // and reweights the rest.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<float> tsdf(
      const Vec3f& point, CornerPolicy corner_policy) const {
    if constexpr (requires(const Access& access) {
                    access.tsdf(point, corner_policy);
                  }) {
      return Access::tsdf(point, corner_policy);
    } else {
      const Vec3f grid = grid_coords(point);
      if (!this->view().resolution().contains(grid)) {
        return compat::nullopt;
      }
      const trilinear::GridSample sample = trilinear::grid_sample_at(grid);
      return this->with_corners(sample.base, [&](const auto& corner_voxel) {
        return trilinear::blend_tsdf(sample.fraction, corner_policy,
                                     corner_voxel);
      });
    }
  }

  // Without color the result is always nullopt, and the else branch never
  // instantiates against the unallocated buffer.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> color(
      const Vec3f& point) const {
    if constexpr (!kHasColor) {
      static_cast<void>(point);
      return compat::nullopt;
    } else if constexpr (requires(const Access& access) {
                           access.color(point);
                         }) {
      return Access::color(point);
    } else {
      const Vec3f grid = grid_coords(point);
      if (!this->view().resolution().contains(grid)) {
        return compat::nullopt;
      }

      const trilinear::GridSample sample = trilinear::grid_sample_at(grid);
      return this->with_color_corners(
          sample.base, [&](const auto& corner_voxel) {
            return trilinear::blend_color(sample.fraction, corner_voxel);
          });
    }
  }

  // Normal from the analytic gradient of the trilinear blend at `point`:
  // one 8-corner gather instead of six extra trilinear samples. The gradient
  // needs the full cell, so this falls back to the central-difference
  // stencil when a corner is missing or unobserved.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> cell_normal(
      const Vec3f& point) const {
    if constexpr (requires(const Access& access) {
                    access.cell_normal(point);
                  }) {
      return Access::cell_normal(point);
    } else {
      const Vec3f grid = grid_coords(point);
      if (!this->view().resolution().contains(grid)) {
        return compat::nullopt;
      }

      const trilinear::GridSample sample = trilinear::grid_sample_at(grid);
      std::array<std::array<std::array<float, 2>, 2>, 2> corners{};
      bool complete = true;
      this->with_corners(sample.base, [&](const auto& corner_voxel) {
        trilinear::for_each_corner([&](int offset_x, int offset_y,
                                       int offset_z) {
          const auto* voxel_ptr = corner_voxel(offset_x, offset_y, offset_z);

          if (voxel_ptr == nullptr) {
            complete = false;
            return;
          }

          const auto voxel = *voxel_ptr;
          if (!voxel.observed() || !voxel.finite_distance()) {
            complete = false;
            return;
          }

          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
          corners[static_cast<std::size_t>(offset_z)][static_cast<std::size_t>(
              offset_y)][static_cast<std::size_t>(offset_x)] = voxel.tsdf();
        });
      });

      if (!complete) {
        return normal(point, CornerPolicy::kSkipMissing);
      }

      const Vec3f gradient = trilinear::cell_gradient(corners, sample.fraction);
      const float length = norm(gradient);

      if (length <= 0.0F) {
        return compat::nullopt;
      }

      return gradient / length;
    }
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> normal(
      const Vec3f& point, CornerPolicy tsdf_corner_policy) const {
    return trilinear::central_normal(*this, point, tsdf_corner_policy);
  }

 private:
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec3f
  grid_coords(const Vec3f& point) const {
    // Reciprocal multiply: div.rn.f32 is a slow macro sequence, and this
    // runs three times per trilinear sample.
    return (point - this->view().origin()) * inv_voxel_size_;
  }

  float inv_voxel_size_;
};

template <MemorySpace Space = MemorySpace::kHost, typename GeomVoxel = Voxel,
          typename Color = FloatColorFacet>
using VolumeSampler = TrilinearSampler<
    DenseCornerAccess<ConstVolumeView<Space, GeomVoxel, Color>>>;

using HostVolumeSampler = VolumeSampler<MemorySpace::kHost>;

// The TsdfVolumeSampler contract lives in trilinear.hpp. Samplers cross into
// kernels by value.
static_assert(TsdfVolumeSampler<HostVolumeSampler>);
static_assert(std::is_trivially_copyable_v<HostVolumeSampler>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_SAMPLER_HPP */
