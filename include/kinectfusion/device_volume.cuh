#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEVICE_VOLUME_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEVICE_VOLUME_CUH

#include <concepts>
#include <cstddef>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {
// Only CUDA translation units include this header!
// Device specialization of the volume storage mechanics: BasicVolume<kDevice>
// owns its voxel and colour data through cuda::DeviceBuffer, the same owner
// the device image sits on. volume.hpp declares the primary template.
template <>
struct SpaceTraits<MemorySpace::kDevice> {
  template <typename T>
  using Buffer = cuda::DeviceBuffer<T>;

  // Matches host uninitialized voxel behavior
  // (1.0F for tsdf, 0 for color & weight)
  template <typename T>
  [[nodiscard]] static Buffer<T> allocate(std::size_t count) {
    auto buffer = Buffer<T>::uninitialized(count);
    buffer.fill(T{});
    return buffer;
  }
};

// Cross-space volume copies
// BasicVolume::copy_from has already checked the geometries match, so raw
// direction-labelled memcpys over the views suffice.
template <>
struct Transfer<MemorySpace::kHost, MemorySpace::kDevice> {
  template <DenseVoxelGridView DstView, DenseVoxelGridView SrcView>
    requires std::same_as<typename DstView::GeometryVoxel,
                          typename SrcView::GeometryVoxel>
  static void copy(DstView destination, SrcView source) {
    using GeomVoxel = typename DstView::GeometryVoxel;
    cuda::check(cudaMemcpy(destination.voxels, source.voxels,
                           source.voxel_count() * sizeof(GeomVoxel),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(volume voxels device to host)");
    if constexpr (DstView::ColorFacet::kEnabled) {
      cuda::check(cudaMemcpy(destination.colors, source.colors,
                             source.voxel_count() *
                                 sizeof(typename DstView::ColorFacet::Voxel),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(volume colors device to host)");
    }
  }
};

// The CUDA backend defines and instantiates this for each registered
// representation.
template <>
class VolumeReduction<MemorySpace::kDevice> {
 public:
  template <DenseVoxelGridView VolumeViewT>
  [[nodiscard]] static std::size_t observed_voxel_count(
      const VolumeViewT& volume);
};

using DeviceVolumeReduction = VolumeReduction<MemorySpace::kDevice>;

template <>
struct Transfer<MemorySpace::kDevice, MemorySpace::kHost> {
  template <DenseVoxelGridView DstView, DenseVoxelGridView SrcView>
    requires std::same_as<typename DstView::GeometryVoxel,
                          typename SrcView::GeometryVoxel>
  static void copy(DstView destination, SrcView source) {
    using GeomVoxel = typename DstView::GeometryVoxel;
    cuda::check(cudaMemcpy(destination.voxels, source.voxels,
                           source.voxel_count() * sizeof(GeomVoxel),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(volume voxels host to device)");
    if constexpr (DstView::ColorFacet::kEnabled) {
      cuda::check(cudaMemcpy(destination.colors, source.colors,
                             source.voxel_count() *
                                 sizeof(typename DstView::ColorFacet::Voxel),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(volume colors host to device)");
    }
  }
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEVICE_VOLUME_CUH */
