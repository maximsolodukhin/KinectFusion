#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_OCCUPANCY_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_OCCUPANCY_CUH

#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/occupancy.hpp>

namespace kinectfusion {

// The CUDA backend defines and instantiates this for each registered volume
// view.
template <>
struct OccupancyRebuild<MemorySpace::kDevice> {
  template <DenseVoxelGridView VolumeViewT>
  static void run(std::uint32_t* raw, std::uint32_t* dilated,
                  std::size_t word_count, const Size3& blocks,
                  const VolumeViewT& volume);
};

using DeviceOccupancyBitmap = BasicOccupancyBitmap<MemorySpace::kDevice>;

template <>
struct BandRebuild<MemorySpace::kDevice> {
  template <DenseVoxelGridView VolumeViewT>
  static void run(std::uint32_t* raw, std::uint32_t* dilated,
                  std::size_t word_count, const Size3& blocks,
                  const VolumeViewT& volume);
};

using DeviceBandBitmap = BasicBandBitmap<MemorySpace::kDevice>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_OCCUPANCY_CUH */
