#include <cstddef>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace kinectfusion {
namespace {

class HostPipeline final : public Pipeline {
 public:
  explicit HostPipeline(const PipelineConfig& config)
      : Pipeline(config.name),
        volume_(config.volume),
        integrator_(config.tsdf_rule, config.integration),
        raycaster_(config.raycast) {}

  using Pipeline::integrate;

  void integrate(const DepthFrame& frame) override {
    integrator_.integrate(volume_.view(), frame);
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastCamera& camera) override {
    return raycaster_.raycast(volume_.view(), camera);
  }

  [[nodiscard]] TrackingSurfacesVariant tracking_surfaces(
      const RaycastCamera& camera,
      const ConstHostVertexNormalMapsView& live) override {
    tracking_model_ = raycaster_.raycast(volume_.view(), camera);
    return HostTrackingSurfaces::from_render(live, view(tracking_model_));
  }

  [[nodiscard]] std::size_t observed_voxel_count() const override {
    return HostVolumeReduction::observed_voxel_count(volume_.view());
  }

  [[nodiscard]] ConstHostVolumeView host_view(
      std::optional<HostVolume>& /*staging*/) const override {
    return volume_.view();
  }

 private:
  HostVolume volume_;
  TsdfIntegrator integrator_;
  Raycaster raycaster_;
  SurfaceMaps tracking_model_;
};

}  // namespace

Pipeline::Pipeline(std::string name) : name_(std::move(name)) {}

#ifndef KINECTFUSION_HAS_CUDA
bool Pipeline::device_available() { return false; }

std::unique_ptr<Pipeline> Pipeline::create_device(
    const PipelineConfig& /*config*/) {
  throw std::logic_error("KinectFusion was built without CUDA support");
}
#endif

Pipeline::Creation Pipeline::create(const PipelineConfig& config) {
  require(!config.name.empty(), "Pipeline requires a non-empty name");

  if (config.space == MemorySpace::kDevice) {
    if (device_available()) {
      return Creation{.pipeline = create_device(config), .fallback_reason = {}};
    }
    return Creation{
        .pipeline = std::make_unique<HostPipeline>(config),
        .fallback_reason = "device memory space unavailable; running on host"};
  }
  return Creation{.pipeline = std::make_unique<HostPipeline>(config),
                  .fallback_reason = {}};
}

}  // namespace kinectfusion
