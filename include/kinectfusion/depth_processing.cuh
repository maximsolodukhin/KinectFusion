#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUH

#include <cstdint>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/image_proc/image.hpp>

namespace kinectfusion {

template <>
class DepthProcessor<MemorySpace::kDevice> {
 public:
  using SurfacePyramid = SurfacePyramidFor<MemorySpace::kDevice>;

  explicit DepthProcessor(DepthProcessingOptions options = {});

  [[nodiscard]] SurfacePyramid build_surface_pyramid(
      image_proc::DeviceImageView<const std::uint16_t> depth,
      const CameraIntrinsics& intrinsics) const;

 private:
  DepthProcessingOptions options_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUH */
