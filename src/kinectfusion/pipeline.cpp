#include <cstddef>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <optional>
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

bool Pipeline::device_available() {
  // Placeholder for CUDA pr.
  return false;
}

Pipeline::Creation Pipeline::create(const PipelineConfig& config) {
  require(!config.name.empty(), "Pipeline requires a non-empty name");

  std::string fallback_reason;
  if (config.space == MemorySpace::kDevice && !device_available()) {
    fallback_reason = "device memory space unavailable; running on host";
  }
  return Creation{.pipeline = std::make_unique<HostPipeline>(config),
                  .fallback_reason = std::move(fallback_reason)};
}

}  // namespace kinectfusion
