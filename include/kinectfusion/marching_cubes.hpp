#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_MARCHING_CUBES_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_MARCHING_CUBES_HPP

#include <array>
#include <cstdint>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <vector>

namespace kinectfusion {

// Extracts the TSDF zero crossing as one welded triangle mesh. Dense views
// sweep every cell; sparse views sweep only allocated blocks, so large
// volumes mesh without a dense staging copy. The CPP defines and
// instantiates `extract` for each registered host view.
class MarchingCubes {
 public:
  struct Mesh {
    std::vector<Vec3f> positions;
    std::vector<Vec3f> normals;
    std::vector<Vec3f> colors;
    std::vector<std::array<std::uint32_t, 3>> triangles;
  };

  // `min_weight` keeps low-confidence cells out of the mesh: a cell is
  // meshed only when all 8 corners have at least that TSDF weight. Zero
  // meshes every observed cell. ViewT must model VoxelGridView; the
  // template is unconstrained because gcc and clang mangle constrained
  // templates differently, and the CUDA TUs reference these symbols
  // through gcc.
  template <typename ViewT>
  [[nodiscard]] static Mesh extract(const ViewT& volume,
                                    float min_weight = 0.0F);
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_MARCHING_CUBES_HPP */
