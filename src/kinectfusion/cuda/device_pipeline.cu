#include <cuda_runtime_api.h>

#include <cstddef>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.cuh>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.cuh>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace kinectfusion {
namespace {

// Integrates, raycasts, and counts on the device; the volume only crosses to
// the host through host_view's explicit staging seam. Per-extent map buffers
// are cached across frames instead of reallocated per call.
class DevicePipeline final : public Pipeline {
 public:
  explicit DevicePipeline(const PipelineConfig& config)
      : Pipeline(config.name),
        volume_(config.volume),
        integrator_(config.tsdf_rule, config.integration),
        raycaster_(config.raycast) {}

  void integrate(const DepthFrame& frame) override {
    const DeviceDepthFrame* own_upload = nullptr;
    integrate(frame, own_upload);
  }

  void integrate(const DepthFrame& frame,
                 const DeviceDepthFrame*& shared_upload) override {
    TsdfIntegrator::validate_frame(frame);
    if (shared_upload == nullptr) {
      fallback_upload_.assign(frame);
      shared_upload = &fallback_upload_;
    }
    const DeviceIntegrationContext context{shared_upload->view(),
                                           integrator_.options()};
    DeviceIntegrationSweep::run(volume_.view(), context, integrator_.rule());
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastCamera& camera) override {
    Raycaster::validate_camera(camera);
    DeviceSurfaceMaps& maps = surface_maps_for(camera.width, camera.height);
    DeviceRaycastSweep::run(device_raycast(camera), maps.view());
    return maps.download();
  }

  [[nodiscard]] TrackingSurfacesVariant tracking_surfaces(
      const RaycastCamera& camera, const LiveViewsVariant& live) override {
    Raycaster::validate_camera(camera);
    DeviceSurfaceMaps& model = surface_maps_for(camera.width, camera.height);
    DeviceRaycastSweep::run(device_raycast(camera), model.view());
    return DeviceTrackingSurfaces::from_render(device_live(live), model.view());
  }

  [[nodiscard]] std::size_t observed_voxel_count() const override {
    return DeviceVolumeReduction::observed_voxel_count(volume_.view());
  }

  [[nodiscard]] ConstHostVolumeView host_view(
      std::optional<HostVolume>& staging) const override {
    staging.emplace(volume_.geometry());
    staging->copy_from(volume_);
    return staging->view();
  }

 private:
  using MapExtent = std::pair<std::size_t, std::size_t>;

  // Device-built pyramids hand over device views directly; host pyramids are
  // staged through the per-extent upload cache.
  [[nodiscard]] ConstDeviceSurfaceView device_live(
      const LiveViewsVariant& live) {
    if (const auto* device = std::get_if<ConstDeviceSurfaceView>(&live)) {
      return *device;
    }
    return view(staged_live(std::get<ConstHostSurfaceView>(live)));
  }

  [[nodiscard]] DeviceSurfaceRaycast device_raycast(
      const RaycastCamera& camera) const {
    return DeviceSurfaceRaycast::from_camera(volume_.view(),
                                             raycaster_.options(), camera);
  }

  [[nodiscard]] DeviceSurfaceMaps& surface_maps_for(std::size_t width,
                                                    std::size_t height) {
    return surface_maps_.try_emplace(MapExtent{width, height}, width, height)
        .first->second;
  }

  [[nodiscard]] const DeviceSurface& staged_live(
      const ConstHostSurfaceView& live) {
    const auto [entry, inserted] = live_maps_.try_emplace(
        MapExtent{live.vertices.width, live.vertices.height});
    DeviceSurface& staged = entry->second;
    if (inserted) {
      staged.vertices = DeviceVec3fImg::uploaded(live.vertices);
      staged.normals = DeviceVec3fImg::uploaded(live.normals);
    } else {
      staged.vertices.copy_from(live.vertices);
      staged.normals.copy_from(live.normals);
    }
    return staged;
  }

  DeviceVolume volume_;
  TsdfIntegrator integrator_;
  Raycaster raycaster_;
  // Own upload for integrate calls no pyramid or earlier pipeline seeded.
  DeviceDepthFrame fallback_upload_;
  std::map<MapExtent, DeviceSurfaceMaps> surface_maps_;
  std::map<MapExtent, DeviceSurface> live_maps_;
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
