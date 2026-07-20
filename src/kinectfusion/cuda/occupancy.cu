#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/occupancy.cuh>
#include <kinectfusion/registered_storages.hpp>

namespace kinectfusion {

namespace {

constexpr unsigned int kBlock = 256;
constexpr unsigned int kMaxGrid = 1024;

// Grid-stride over voxels. A voxel in the filter set marks the bit of its
// block in the layout's bit order.
template <BlockBitmapView LayoutView, typename Filter,
          DenseVoxelGridView VolumeViewT>
__global__ void mark_blocks_kernel(std::uint32_t* raw, LayoutView layout,
                                   VolumeViewT volume, Filter filter) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const Size3 resolution = volume.resolution();
  const std::size_t count = volume.voxel_count();
  for (std::size_t index = first; index < count; index += stride) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (!filter(volume.voxels[index])) {
      continue;
    }
    const Size3 voxel = resolution.unflatten(index);
    const std::size_t block =
        layout.block_index(voxel.x / kVoxelBlockEdge, voxel.y / kVoxelBlockEdge,
                           voxel.z / kVoxelBlockEdge);
    atomicOr(&raw[block / kBitmapWordBits], 1U << (block % kBitmapWordBits));
  }
}

// One thread per block: OR the 3^3 neighborhood of raw bits into dilated.
template <BlockBitmapView LayoutView>
__global__ void dilate_kernel(const std::uint32_t* raw, std::uint32_t* dilated,
                              LayoutView raw_view) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t count =
      raw_view.blocks.x * raw_view.blocks.y * raw_view.blocks.z;
  for (std::size_t index = first; index < count; index += stride) {
    const GridIndex block = raw_view.block_coords(index);
    if (BlockBitmapOps::neighborhood_occupied(raw_view, block.x, block.y,
                                              block.z)) {
      atomicOr(&dilated[index / kBitmapWordBits],
               1U << (index % kBitmapWordBits));
    }
  }
}

template <BlockBitmapView LayoutView, typename Filter,
          DenseVoxelGridView VolumeViewT>
void rebuild_device(std::uint32_t* raw, std::uint32_t* dilated,
                    std::size_t word_count, const Size3& blocks,
                    const VolumeViewT& volume) {
  const std::size_t bytes = word_count * sizeof(std::uint32_t);
  cuda::check(cudaMemsetAsync(raw, 0, bytes), "bitmap raw clear");
  cuda::check(cudaMemsetAsync(dilated, 0, bytes), "bitmap dilated clear");

  const LayoutView raw_view{.words = raw, .blocks = blocks, .geometry = {}};
  const unsigned int mark_grid =
      std::min(cuda::ceil_div(volume.voxel_count(), kBlock), kMaxGrid);
  mark_blocks_kernel<<<mark_grid, kBlock>>>(raw, raw_view, volume, Filter{});
  cuda::check(cudaGetLastError(), "mark_blocks_kernel launch");

  const std::size_t block_count = blocks.x * blocks.y * blocks.z;
  const unsigned int dilate_grid =
      std::min(cuda::ceil_div(block_count, kBlock), kMaxGrid);
  dilate_kernel<<<dilate_grid, kBlock>>>(raw, dilated, raw_view);
  cuda::check(cudaGetLastError(), "dilate_kernel launch");
}

}  // namespace

template <DenseVoxelGridView VolumeViewT>
void OccupancyRebuild<MemorySpace::kDevice>::run(std::uint32_t* raw,
                                                 std::uint32_t* dilated,
                                                 std::size_t word_count,
                                                 const Size3& blocks,
                                                 const VolumeViewT& volume) {
  rebuild_device<OccupancyView, ObservedFilter>(raw, dilated, word_count,
                                                blocks, volume);
}

template <DenseVoxelGridView VolumeViewT>
void BandRebuild<MemorySpace::kDevice>::run(std::uint32_t* raw,
                                            std::uint32_t* dilated,
                                            std::size_t word_count,
                                            const Size3& blocks,
                                            const VolumeViewT& volume) {
  rebuild_device<BandView, BandFilter>(raw, dilated, word_count, blocks,
                                       volume);
}

// One instantiation per registered storage combination.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KINECTFUSION_INSTANTIATE(GeomVoxel, Color)                       \
  template void OccupancyRebuild<MemorySpace::kDevice>::run(             \
      std::uint32_t*, std::uint32_t*, std::size_t, const Size3&,         \
      const VolumeView<MemorySpace::kDevice, false, GeomVoxel, Color>&); \
  template void BandRebuild<MemorySpace::kDevice>::run(                  \
      std::uint32_t*, std::uint32_t*, std::size_t, const Size3&,         \
      const VolumeView<MemorySpace::kDevice, false, GeomVoxel, Color>&);
KINECTFUSION_FOR_EACH_REGISTERED_STORAGE(KINECTFUSION_INSTANTIATE)
#undef KINECTFUSION_INSTANTIATE

}  // namespace kinectfusion
