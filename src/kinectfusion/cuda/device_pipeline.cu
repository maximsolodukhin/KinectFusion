#include <cuda_runtime_api.h>

#include <cstddef>
#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.cuh>
#include <memory>
#include <optional>

namespace kinectfusion {
namespace {

// Integrates on the device; raycasting stages a host copy of the volume
// until the raycast kernel lands.
class DevicePipeline final : public Pipeline {
 public:
  explicit DevicePipeline(const PipelineConfig& config)
      : Pipeline(config.name),
        volume_(config.volume),
        integrator_(config.tsdf_rule, config.integration),
        raycaster_(config.raycast) {}

  void integrate(const DepthFrame& frame) override {
    TsdfIntegrator::validate_frame(frame);
    const DeviceDepthFrame device_frame = DeviceDepthFrame::upload(frame);
    const IntegrationContext<MemorySpace::kDevice> context{
        device_frame.view(), integrator_.options()};
    DeviceIntegrationSweep::run(volume_.view(), context, integrator_.rule());
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastCamera& camera) override {
    std::optional<HostVolume> staging;
    return raycaster_.raycast(host_view(staging), camera);
  }

  [[nodiscard]] std::size_t observed_voxel_count() const override {
    HostVolume staging{volume_.geometry()};
    staging.copy_from(volume_);
    return staging.observed_voxel_count();
  }

  [[nodiscard]] ConstHostVolumeView host_view(
      std::optional<HostVolume>& staging) const override {
    staging.emplace(volume_.geometry());
    staging->copy_from(volume_);
    return staging->view();
  }

 private:
  DeviceVolume volume_;
  TsdfIntegrator integrator_;
  Raycaster raycaster_;
};

}  // namespace

bool Pipeline::device_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<Pipeline> Pipeline::create_device(
    const PipelineConfig& config) {
  return std::make_unique<DevicePipeline>(config);
}

}  // namespace kinectfusion
