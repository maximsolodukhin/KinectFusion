#include <kinectfusion/grid.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_sampler.hpp>

namespace kinectfusion {

Raycaster::Raycaster(RaycastOptions options) : options_(validated(options)) {}

SurfaceMaps Raycaster::raycast(ConstHostVolumeView volume,
                               const RaycastCamera& camera) const {
  validate_camera(camera);
  using Vec3fImg = image_proc::Vector3fImage;
  using ColorImg = image_proc::ColorImage;
  SurfaceMaps maps{
      .points = Vec3fImg{camera.width, camera.height, invalid_vec3f()},
      .normals = Vec3fImg{camera.width, camera.height, invalid_vec3f()},
      .colors = ColorImg{camera.width, camera.height}};

  const SurfaceRaycast<MemorySpace::kHost> raycast{
      HostVolumeSampler{volume}, options_, camera.rotation(), camera.origin(),
      camera.intrinsics};
  const HostSurfaceMapsView maps_view = view(maps);
  for (const auto [col, row] : PixelIndices{camera.width, camera.height}) {
    raycast.render_pixel(maps_view, col, row);
  }
  return maps;
}

RaycastOptions Raycaster::validated(RaycastOptions options) {
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Raycast depth range is invalid");
  require(options.step_scale > 0.0F, "Raycast step scale must be positive");
  return options;
}

void Raycaster::validate_camera(const RaycastCamera& camera) {
  require(camera.intrinsics.fx > 0.0F && camera.intrinsics.fy > 0.0F,
          "Raycast intrinsics must have positive focal lengths");
  require(camera.width > 0U && camera.height > 0U,
          "Raycast dimensions must be positive");
  require(camera.camera_to_world.allFinite(),
          "Raycast camera_to_world must be finite");
}

}  // namespace kinectfusion
