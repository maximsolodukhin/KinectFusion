#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEVICE_VOLUME_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEVICE_VOLUME_CUH

#include <cstddef>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {
// Only CUDA translation units include this header!
// Device specialization of the volume storage mechanics: BasicVolume<kDevice>
// owns its voxel and colour data through cuda::DeviceBuffer, the same owner
// the device image sits on. volume.hpp declares the primary template.
//
// The device Transfer directions and the device observed_voxel_count
// reduction land with the volume CUDA PR.
template <>
struct SpaceTraits<MemorySpace::kDevice> {
  template <typename T>
  using Buffer = cuda::DeviceBuffer<T>;

  template <typename T>
  [[nodiscard]] static Buffer<T> allocate(std::size_t count) {
    return Buffer<T>{count};
  }
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEVICE_VOLUME_CUH */
