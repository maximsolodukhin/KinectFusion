#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH

#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/tsdf_integration.hpp>

namespace kinectfusion {
// Owns the device memory for a single depth frame, including the depth image,
// color image, and normals.
class DeviceDepthFrame {
 public:
  DeviceDepthFrame() = default;
 
  // Reuse memory of another frame, if the dimensions match.
  void assign(const DepthFrame& frame);
  void assign_from_pyramid(const DepthFrame& frame,
                           const DeviceDepthImg& raw_depth,
                           const DeviceVec3fImg& normals);

  [[nodiscard]] DeviceDepthFrameView view() const;

 private:
  DeviceDepthImg depth_;
  DeviceColorImg color_;
  DeviceVec3fImg normals_;
  CameraIntrinsics intrinsics_{};
  RigidTransform world_to_camera_{};
};

// Device driver of the per-element TSDF layer, the mirror image of the
// IntegrationSweep
// One kernel instantiation per rule in the variant.
// Stream-ordered: launch errors throw here, execution errors surface at
// the next device-to-host copy.
class DeviceIntegrationSweep {
 public:
  static void run(const DeviceVolumeView& volume,
                  const DeviceIntegrationContext& context,
                  const TsdfRuleVariant& rule);
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH */
