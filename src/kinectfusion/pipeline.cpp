#include <cstddef>
#include <kinectfusion/block_rep.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/marching_cubes.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_representation.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>  // NOLINT(misc-include-cleaner)

namespace kinectfusion {
namespace {

template <typename Rep>
  requires VolumeRepresentation<Rep, MemorySpace::kHost>
class BasicHostPipeline final : public Pipeline {
  static constexpr MemorySpace kSpace = MemorySpace::kHost;

 public:
  explicit BasicHostPipeline(const PipelineConfig& config)
      : Pipeline(config.name),
        rep_(RepresentationFactory::make<Rep>(config.volume,
                                              config.sparse_block_capacity)),
        index_(make_index(config)),
        integrator_(config.tsdf_rule, config.integration),
        raycaster_(config.raycast) {}

  using Pipeline::integrate;

  void integrate(const DepthFrame& frame) override {
    TsdfIntegrator::validate_frame(frame);
    rep_.integrate(frame.view(), integrator_.options(), integrator_.rule());
    if constexpr (FlatVoxelRepresentation<Rep>) {
      index_.rebuild(rep_.view());
    }
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastCamera& camera) override {
    return render_model(camera);
  }

  [[nodiscard]] TrackingSurfacesVariant tracking_surfaces(
      const RaycastCamera& camera, const LiveViewsVariant& live) override {
    tracking_model_ = render_model(camera);
    return HostTrackingSurfaces::from_render(
        std::get<ConstHostSurfaceView>(live), view(tracking_model_));
  }

  [[nodiscard]] std::size_t observed_voxel_count() const override {
    return rep_.observed_voxel_count();
  }

  [[nodiscard]] ConstHostVolumeView host_view(
      std::optional<HostVolume>& staging) const override {
    return rep_.host_dense_view(staging);
  }

  [[nodiscard]] MarchingCubes::Mesh extract_mesh(
      float min_weight) const override {
    return MarchingCubes::extract(rep_.view(), min_weight);
  }

 private:
  [[nodiscard]] static EmptySpaceIndex<kSpace> make_index(
      const PipelineConfig& config) {
    if constexpr (FlatVoxelRepresentation<Rep>) {
      return {config.raycast_backend, config.volume};
    } else {
      return {};
    }
  }

  // Sparse storage has no empty-space index, so the pipeline instantiates
  // only the NoSkip march.
  [[nodiscard]] SurfaceMaps render_model(const RaycastCamera& camera) {
    if constexpr (FlatVoxelRepresentation<Rep>) {
      return index_.visit([&](const auto& skip) {
        return raycaster_.render(rep_.sampler(), camera, skip);
      });
    } else {
      return raycaster_.render(rep_.sampler(), camera);
    }
  }

  Rep rep_;
  EmptySpaceIndex<kSpace> index_;
  TsdfIntegrator integrator_;
  Raycaster raycaster_;
  SurfaceMaps tracking_model_;
};

// The host half of the registry: one explicit combination per registered
// (voxel, color) pair.
template <TsdfVoxel GeomVoxel, typename Color>
std::unique_ptr<Pipeline> make_host(const PipelineConfig& config) {
  if (config.storage == StorageLayout::kSparse) {
    return std::make_unique<
        BasicHostPipeline<BlockRep<MemorySpace::kHost, GeomVoxel, Color>>>(
        config);
  }
  return std::make_unique<
      BasicHostPipeline<DenseRep<MemorySpace::kHost, GeomVoxel, Color>>>(
      config);
}

std::unique_ptr<Pipeline> create_host(const PipelineConfig& config) {
  return Pipeline::visit_storage(
      config, [&config]<TsdfVoxel GeomVoxel, typename Color>() {
        return make_host<GeomVoxel, Color>(config);
      });
}

}  // namespace

Pipeline::Pipeline(std::string name) : name_(std::move(name)) {}

void Pipeline::require_valid_storage(const PipelineConfig& config) {
  if (config.storage != StorageLayout::kSparse) {
    return;
  }
  require(config.integration.mode == IntegrationMode::kBand,
          "Sparse storage implies band integration (nothing is allocated far "
          "from surfaces); set integration = 'band'");
  require(config.raycast_backend == RaycastBackend::kMarch,
          "Sparse storage supports the plain march raycast backend");
}

#ifndef KINECTFUSION_HAS_CUDA
bool Pipeline::device_available() { return false; }

std::unique_ptr<Pipeline> Pipeline::create_device(
    const PipelineConfig& /*config*/) {
  throw std::logic_error("KinectFusion was built without CUDA support");
}
#endif

Pipeline::Creation Pipeline::create(const PipelineConfig& config) {
  require(!config.name.empty(), "Pipeline requires a non-empty name");
  require_valid_storage(config);
  std::string fallback_reason{};

  if (config.space == MemorySpace::kDevice) {
    if (device_available()) {
      return Creation{.pipeline = create_device(config),
                      .space = MemorySpace::kDevice,
                      .fallback_reason = {}};
    }

    fallback_reason = "device memory space unavailable; running on host";
  }

  return Creation{.pipeline = create_host(config),
                  .space = MemorySpace::kHost,
                  .fallback_reason = fallback_reason};
}

}  // namespace kinectfusion
