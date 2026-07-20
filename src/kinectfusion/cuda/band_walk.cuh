#ifndef KINECTFUSION_SRC_KINECTFUSION_CUDA_BAND_WALK_CUH
#define KINECTFUSION_SRC_KINECTFUSION_CUDA_BAND_WALK_CUH

#include <cstddef>
#include <cstdint>
#include <kinectfusion/block_rep.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/occupancy.hpp>
#include <kinectfusion/tsdf_integration.cuh>

// Device side of the truncation-band walk, shared by the dense band sweep
// and the sparse allocation pass.
namespace kinectfusion::band_walk {

inline constexpr unsigned int kLinearBlock = 256;
inline constexpr unsigned int kMaxGrid = 1024;
inline constexpr auto kBlockThreads =
    static_cast<unsigned int>(kVoxelBlockVolume);

// Sets bit `index`. Warp lanes that hit the same word merge their bits into
// one atomic operation.
struct WarpAggregatedSet {
  std::uint32_t* bitmap;

  __device__ void operator()(std::size_t index) const {
    const auto word = static_cast<unsigned int>(index / kBitmapWordBits);
    const std::uint32_t bit = 1U << (index % kBitmapWordBits);

    const unsigned int peers = __match_any_sync(__activemask(), word);
    const std::uint32_t merged = __reduce_or_sync(peers, bit);
    const unsigned int lane = ((threadIdx.y * blockDim.x) + threadIdx.x) % 32U;

    if (lane == static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1)) {
      atomicOr(&bitmap[word], merged);
    }
  }
};

// One thread per depth pixel: walk the truncation band and mark each block
// it crosses.
template <typename Visitor>
__global__ void mark_band_blocks_kernel(DeviceIntegrationContext context,
                                        RigidTransform camera_to_world,
                                        VolumeGeometry geometry,
                                        BlockGrid blocks, Visitor visitor) {
  const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto& frame = context.frame();
  if (col >= frame.depth.width || row >= frame.depth.height) {
    return;
  }
  TruncationBandWalk::visit_pixel(frame, context.options(), geometry, blocks,
                                  TruncationBandWalk::half_block_step(geometry),
                                  camera_to_world, col, row, visitor);
}

// One CUDA block per work-list entry (grid-strided), one thread per voxel of
// the 8^3 block. Sparse views skip unallocated (overflowed) blocks.
template <VoxelGridView VolumeViewT, TsdfUpdateRule<MemorySpace::kDevice> Rule>
__global__ void integrate_listed_blocks_kernel(VolumeViewT volume,
                                               DeviceIntegrationContext context,
                                               Rule rule, BlockGrid blocks,
                                               const std::uint32_t* list,
                                               const unsigned int* count) {
  const Size3 resolution = volume.resolution();
  const Size3 offset =
      Size3{.x = kVoxelBlockEdge, .y = kVoxelBlockEdge, .z = kVoxelBlockEdge}
          .unflatten(threadIdx.x);

  for (unsigned int item = blockIdx.x; item < *count; item += gridDim.x) {
    const std::uint32_t block = list[item];
    if constexpr (requires { volume.grid; }) {
      if (volume.grid[block] == kUnallocatedBlock) {
        continue;
      }
    }

    const GridIndex base = blocks.voxel_base(block);
    const std::size_t x = base.x + offset.x;
    const std::size_t y = base.y + offset.y;
    const std::size_t z = base.z + offset.z;

    if (x >= resolution.x || y >= resolution.y || z >= resolution.z) {
      continue;
    }
    rule.update(volume, context, x, y, z);
  }
}

}  // namespace kinectfusion::band_walk

#endif /* KINECTFUSION_SRC_KINECTFUSION_CUDA_BAND_WALK_CUH */
