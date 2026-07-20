#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_REGISTERED_STORAGES_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_REGISTERED_STORAGES_HPP

// The registered (geometric voxel, color facet) storage combinations. Every
// explicit device instantiation list expands from this one list. The names
// come from volume.hpp. Add a combination here and in the registry switch.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KINECTFUSION_FOR_EACH_REGISTERED_STORAGE(X) \
  X(Voxel, FloatColorFacet)                         \
  X(Voxel, NoColorFacet)                            \
  X(QuantizedVoxel, FloatColorFacet)                \
  X(QuantizedVoxel, NoColorFacet)                   \
  X(Bf16Voxel, FloatColorFacet)                     \
  X(Bf16Voxel, NoColorFacet)

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_REGISTERED_STORAGES_HPP */
