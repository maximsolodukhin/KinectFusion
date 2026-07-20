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
  DeviceVec3fImg points_;
  DeviceVec3fImg normals_;
  DeviceColorImg colors_;
};

// The CUDA backend defines and instantiates this for each registered
// sampler.
class DeviceRaycastSweep {
 public:
  template <PixelRenderer<MemorySpace::kDevice> Raycast>
  static void run(const Raycast& raycast, const DeviceSurfaceMapsView& maps);
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_CUH */
