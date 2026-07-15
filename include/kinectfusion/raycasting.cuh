#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_CUH

#include <cstddef>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {

// Owns the device memory for one rendered surface
class DeviceSurfaceMaps {
 public:
  DeviceSurfaceMaps(std::size_t width, std::size_t height);

  [[nodiscard]] DeviceSurfaceMapsView view();
  [[nodiscard]] ConstDeviceSurfaceMapsView view() const;

  // download() is the device-to-host seam handing the maps back to the host API
  [[nodiscard]] SurfaceMaps download() const;

 private:
  image_proc::DeviceVector3fImage points_;
  image_proc::DeviceVector3fImage normals_;
  image_proc::DeviceColorImage colors_;
};

// Mirror of Raycaster::raycast host loop.
// Synchronous. throws std::runtime_error on CUDA failures
class DeviceRaycastSweep {
 public:
  static void run(const DeviceSurfaceRaycast& raycast,
                  const DeviceSurfaceMapsView& maps);
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_CUH */
