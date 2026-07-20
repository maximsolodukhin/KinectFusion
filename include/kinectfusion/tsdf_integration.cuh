#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH

#include <cstdint>
#include <kinectfusion/cuda/pinned_buffer.cuh>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/tsdf_integration.hpp>
#include <memory>

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
  cuda::PinnedBuffer<std::uint32_t> color_staging_;
  DeviceVec3fImg normals_;
  CameraIntrinsics intrinsics_{};
  RigidTransform world_to_camera_{};
};

// Device sweep: one kernel instantiation per (rule, volume view) pair.
// Stream-ordered: launch errors throw here, execution errors surface at the
// next device-to-host copy. The CUDA backend defines and instantiates `run`
// for each registered volume view.
template <>
struct IntegrationSweep<MemorySpace::kDevice> {
  template <VoxelGridView VolumeViewT>
  static void run(const VolumeViewT& volume,
                  const DeviceIntegrationContext& context,
                  const TsdfRuleVariant& rule);
};

using DeviceIntegrationSweep = IntegrationSweep<MemorySpace::kDevice>;

// Stateful band sweep: the allocation bitmap and the compact work list live
// on the device. The CUDA backend instantiates `run` for each registered
// volume view.
template <>
class BandIntegrationSweep<MemorySpace::kDevice> {
 public:
  BandIntegrationSweep();
  ~BandIntegrationSweep();

  BandIntegrationSweep(const BandIntegrationSweep&) = delete;
  BandIntegrationSweep& operator=(const BandIntegrationSweep&) = delete;
  BandIntegrationSweep(BandIntegrationSweep&&) noexcept;
  BandIntegrationSweep& operator=(BandIntegrationSweep&&) noexcept;

  template <VoxelGridView VolumeViewT>
  void run(const VolumeViewT& volume, const DeviceIntegrationContext& context,
           const TsdfRuleVariant& rule);

 private:
  struct Scratch;

  std::unique_ptr<Scratch> scratch_;
};

using DeviceBandIntegrationSweep = BandIntegrationSweep<MemorySpace::kDevice>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_CUH */
