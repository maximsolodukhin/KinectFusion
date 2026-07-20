#include <cuda_runtime_api.h>

#include <cstddef>
#include <kinectfusion/block_rep.cuh>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/occupancy.cuh>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.cuh>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.cuh>
#include <kinectfusion/volume_representation.hpp>
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
template <typename Rep>
  requires VolumeRepresentation<Rep, MemorySpace::kDevice>
class BasicDevicePipeline final : public Pipeline {
  static constexpr MemorySpace kSpace = MemorySpace::kDevice;

 public:
  explicit BasicDevicePipeline(const PipelineConfig& config)
      : Pipeline(config.name),
        rep_(RepresentationFactory::make<Rep>(config.volume,
                                              config.sparse_block_capacity)),
        index_(make_index(config)),
        integrator_(config.tsdf_rule, config.integration),
        raycaster_(config.raycast) {}

  void integrate(const DepthFrame& frame,
                 const DeviceDepthFrame* upload) override {
    TsdfIntegrator::validate_frame(frame);
    if (upload == nullptr) {
      fallback_upload_.assign(frame);
      upload = &fallback_upload_;
    }
    // The upload carries only the shared frame data; the pose is per pipeline,
    // so a set can share one upload across members that tracked apart.
    DeviceDepthFrameView view = upload->view();
    view.world_to_camera = from_eigen(frame.world_to_camera);
    rep_.integrate(view, integrator_.options(), integrator_.rule());
    if constexpr (FlatVoxelRepresentation<Rep>) {
      index_.rebuild(rep_.view());
    }
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastCamera& camera) override {
    Raycaster::validate_camera(camera);
    DeviceSurfaceMaps& maps = surface_maps_for(camera.width, camera.height);
    render_model(camera, maps);

    return maps.download();
  }

  void track(const RaycastCamera& camera, const PyramidLevel& live,
             TrackingSurfaceConsumer& consumer) override {
    Raycaster::validate_camera(camera);
    DeviceSurfaceMaps& model = surface_maps_for(camera.width, camera.height);
    render_model(camera, model);

    consumer.consume(DeviceTrackingSurfaces::from_render(
        device_live(live.surface), model.view()));
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
    if constexpr (requires { rep_.host_snapshot(); }) {
      return MarchingCubes::extract(rep_.host_snapshot().view(), min_weight);
    } else {
      std::optional<HostVolume> staging;
      return MarchingCubes::extract(rep_.host_dense_view(staging), min_weight);
    }
  }

 private:
  using MapExtent = std::pair<std::size_t, std::size_t>;

  [[nodiscard]] static EmptySpaceIndex<kSpace> make_index(
      const PipelineConfig& config) {
    if constexpr (FlatVoxelRepresentation<Rep>) {
      return {config.raycast_backend, config.volume};
    } else {
      return {};
    }
  }

  // Device-built pyramids hand over device views directly; host pyramids are
  // staged through the per-extent upload cache.
  [[nodiscard]] ConstDeviceSurfaceView device_live(
      const LiveViewsVariant& live) {
    if (const auto* device = std::get_if<ConstDeviceSurfaceView>(&live)) {
      return *device;
    }
    return view(staged_live(std::get<ConstHostSurfaceView>(live)));
  }

  // Sparse storage has no empty-space index, so the pipeline instantiates
  // only the NoSkip raycast.
  void render_model(const RaycastCamera& camera, DeviceSurfaceMaps& maps) {
    if constexpr (FlatVoxelRepresentation<Rep>) {
      index_.visit(
          [&](const auto& skip) { launch_raycast(skip, camera, maps); });
    } else {
      launch_raycast(NoSkip{}, camera, maps);
    }
  }

  template <SkipPolicy Skip>
  void launch_raycast(const Skip& skip, const RaycastCamera& camera,
                      DeviceSurfaceMaps& maps) {
    const SurfaceRaycast<kSpace, typename Rep::Sampler, Skip> raycast{
        rep_.sampler(), raycaster_.options(), camera.pose(), camera.intrinsics,
        skip};

    DeviceRaycastSweep::run(raycast, maps.view());
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

  Rep rep_;
  EmptySpaceIndex<kSpace> index_;
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
  return visit_representation<MemorySpace::kDevice>(
      config, [&config]<typename Rep>() -> std::unique_ptr<Pipeline> {
        return std::make_unique<BasicDevicePipeline<Rep>>(config);
      });
}

}  // namespace kinectfusion
