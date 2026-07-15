#include <algorithm>
#include <cstddef>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {

std::size_t HostVolumeReduction::observed_voxel_count(
    const ConstHostVolumeView& volume) {
  std::size_t count = 0;
  for (const Voxel& voxel : volume.voxel_span()) {
    if (voxel.weight > 0.0F) {
      ++count;
    }
  }
  return count;
}

void Transfer<MemorySpace::kHost, MemorySpace::kHost>::copy(
    HostVolumeView destination, ConstHostVolumeView source) {
  std::copy_n(source.voxels, source.voxel_count(), destination.voxels);
  std::copy_n(source.colors, source.voxel_count(), destination.colors);
}

}  // namespace kinectfusion
