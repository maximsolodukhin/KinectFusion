#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_REPRESENTATION_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_REPRESENTATION_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_sampler.hpp>
#include <optional>
#include <type_traits>

namespace kinectfusion {

// The storage contract of a pipeline. A model owns the buffers, gives
// raycasting a TsdfVolumeSampler, owns the integrate sweep, and
// materializes a dense float host view for the comparison harness.
template <typename R, MemorySpace Space>
concept VolumeRepresentation =
    TsdfVolumeSampler<typename R::Sampler> &&
    requires(R rep, const R crep, const DepthFrameView<Space>& frame,
             const TsdfIntegrationOptions& options, const TsdfRuleVariant& rule,
             std::optional<HostVolume>& staging) {
      { crep.sampler() } -> std::same_as<typename R::Sampler>;
      { crep.geometry() } -> std::same_as<const VolumeGeometry&>;
      { crep.observed_voxel_count() } -> std::same_as<std::size_t>;
      rep.integrate(frame, options, rule);
      { crep.host_dense_view(staging) } -> std::same_as<ConstHostVolumeView>;
    };

// Storage whose view is the dense flat voxel array. Only such storage can
// feed the occupancy and band bitmap rebuilds.
template <typename R>
concept FlatVoxelRepresentation = requires(const R& rep) {
  { rep.view() } -> DenseVoxelGridView;
};

struct RepresentationFactory {
  template <typename Rep>
  [[nodiscard]] static Rep make(const VolumeGeometry& geometry,
                                std::size_t sparse_block_capacity) {
    if constexpr (std::constructible_from<Rep, const VolumeGeometry&,
                                          std::size_t>) {
      return Rep{geometry, sparse_block_capacity};
    } else {
      return Rep{geometry};
    }
  }
};

// The dense voxel-grid model: BasicVolume storage under the representation
// contract.
template <MemorySpace Space, typename GeomVoxel = Voxel,
          typename Color = FloatColorFacet>
class DenseRep {
  static constexpr bool kDefaultStorage =
      std::is_same_v<GeomVoxel, Voxel> &&
      std::is_same_v<Color, FloatColorFacet>;

 public:
  using Sampler = VolumeSampler<Space, GeomVoxel, Color>;
  using Volume = BasicVolume<Space, GeomVoxel, Color>;

  // Throws std::invalid_argument
  explicit DenseRep(const VolumeGeometry& geometry) : volume_(geometry) {}

  [[nodiscard]] Sampler sampler() const { return Sampler{volume_.view()}; }

  [[nodiscard]] const VolumeGeometry& geometry() const {
    return volume_.geometry();
  }

  [[nodiscard]] auto view() { return volume_.view(); }
  [[nodiscard]] auto view() const { return volume_.view(); }

  [[nodiscard]] std::size_t observed_voxel_count() const {
    return VolumeReduction<Space>::observed_voxel_count(volume_.view());
  }

  void integrate(const DepthFrameView<Space>& frame,
                 const TsdfIntegrationOptions& options,
                 const TsdfRuleVariant& rule) {
    const IntegrationContext<Space> context{frame, options};
    if (options.mode == IntegrationMode::kBand) {
      band_sweep_.run(volume_.view(), context, rule);
      return;
    }
    IntegrationSweep<Space>::run(volume_.view(), context, rule);
  }

  // Dense float materialization for the compare harness. The host default
  // storage is its own dense view. The device default stages a copy. Other
  // storages convert voxel by voxel, so comparisons always run in float.
  [[nodiscard]] ConstHostVolumeView host_dense_view(
      std::optional<HostVolume>& staging) const {
    if constexpr (Space == MemorySpace::kHost && kDefaultStorage) {
      return volume_.view();
    } else if constexpr (kDefaultStorage) {
      staging.emplace(volume_.geometry());
      staging->copy_from(volume_);
      return staging->view();
    } else if constexpr (Space == MemorySpace::kHost) {
      staging.emplace(volume_.geometry());
      materialize(volume_.view(), staging->view());
      return staging->view();
    } else {
      BasicVolume<MemorySpace::kHost, GeomVoxel, Color> native{
          volume_.geometry()};
      native.copy_from(volume_);
      staging.emplace(volume_.geometry());
      materialize(native.view(), staging->view());
      return staging->view();
    }
  }

 private:
  template <DenseVoxelGridView SrcView>
  static void materialize(const SrcView& source, HostVolumeView dense) {
    const auto voxels = source.voxel_span();
    auto out = dense.voxel_span();
    for (std::size_t i = 0; i < voxels.size(); ++i) {
      out[i] = Voxel{.distance = voxels[i].tsdf(),
                     .weight = voxels[i].weight_value()};
    }
    if constexpr (SrcView::ColorFacet::kEnabled) {
      std::copy_n(source.colors, source.voxel_count(), dense.colors);
    }
  }

  Volume volume_;
  BandIntegrationSweep<Space> band_sweep_;
};

using HostDenseRep = DenseRep<MemorySpace::kHost>;
using DeviceDenseRep = DenseRep<MemorySpace::kDevice>;

static_assert(VolumeRepresentation<HostDenseRep, MemorySpace::kHost>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_REPRESENTATION_HPP */
