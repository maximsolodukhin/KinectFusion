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

  void integrate(const DepthFrame& frame) override {
    integrator_.integrate(volume_.view(), frame);
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastCamera& camera) override {
    return raycaster_.raycast(volume_.view(), camera);
  }

  [[nodiscard]] std::size_t observed_voxel_count() const override {
    return volume_.observed_voxel_count();
  }

  [[nodiscard]] ConstHostVolumeView host_view(
      std::optional<HostVolume>& /*staging*/) const override {
    return volume_.view();
  }

 private:
  HostVolume volume_;
  TsdfIntegrator integrator_;
  Raycaster raycaster_;
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
