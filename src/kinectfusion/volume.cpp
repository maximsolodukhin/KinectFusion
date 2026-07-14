#include <algorithm>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {

void Transfer<MemorySpace::kHost, MemorySpace::kHost>::copy(
    HostVolumeView destination, ConstHostVolumeView source) {
  std::copy_n(source.voxels, source.voxel_count(), destination.voxels);
  std::copy_n(source.colors, source.voxel_count(), destination.colors);
}

}  // namespace kinectfusion
