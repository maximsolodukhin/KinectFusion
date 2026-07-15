#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH

#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/tsdf_integration.hpp>

namespace kinectfusion {
// Owns the device memory for a single depth frame, including the depth image,
// color image, and normals.
class DeviceDepthFrame {
 public:
  // host to device copy on each frame
  [[nodiscard]] static DeviceDepthFrame upload(const DepthFrame& frame);

  [[nodiscard]] DeviceDepthFrameView view() const;

 private:
  DeviceDepthFrame() = default;

  image_proc::DeviceDepthImage depth_;
  image_proc::DeviceColorImage color_;
  image_proc::DeviceVector3fImage normals_;
  CameraIntrinsics intrinsics_{};
  RigidTransform world_to_camera_{};
};

// Device driver of the per-element TSDF layer, the mirror image of the
// IntegrationSweep
// One kernel instantiation per rule in the variant.
// Synchronous. throws std::runtime_error on CUDA failures
class DeviceIntegrationSweep {
 public:
  static void run(const DeviceVolumeView& volume,
                  const DeviceIntegrationContext& context,
                  const TsdfRuleVariant& rule);
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH */
