#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/block_rep.cuh>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/registered_storages.hpp>
#include <variant>
#include <vector>

#include "band_walk.cuh"

namespace kinectfusion {

namespace {

// Enumerate marked blocks into the work list and allocate the new ones.
__global__ void allocate_blocks_kernel(
    const std::uint32_t* bitmap, std::size_t word_count, std::uint32_t* grid,
    std::uint32_t capacity, std::uint32_t* work_list, unsigned int* work_count,
    std::uint32_t* new_list, unsigned int* new_count, unsigned int* allocated) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  for (std::size_t word = first; word < word_count; word += stride) {
    BlockBitmapOps::for_each_set_bit(
        bitmap[word], word * kBitmapWordBits, [&](std::size_t index) {
          const auto block = static_cast<std::uint32_t>(index);
          if (grid[block] == kUnallocatedBlock) {
            const unsigned int slot = atomicAdd(allocated, 1U);
            if (slot >= capacity) {
              return;  // overflow: leave unallocated, drop the updates
            }
            grid[block] = slot;
            new_list[atomicAdd(new_count, 1U)] = block;
          }
          work_list[atomicAdd(work_count, 1U)] = block;
        });
  }
}

// One CUDA block per newly allocated pool slot: default-initialize voxels.
template <TsdfVoxel GeomVoxel, typename ColorVoxelT>
__global__ void init_blocks_kernel(const std::uint32_t* new_list,
                                   const unsigned int* new_count,
                                   const std::uint32_t* grid, GeomVoxel* pool,
                                   ColorVoxelT* color_pool, bool has_color) {
  for (unsigned int item = blockIdx.x; item < *new_count; item += gridDim.x) {
    const std::uint32_t slot = grid[new_list[item]];
    const std::size_t base = static_cast<std::size_t>(slot) * kVoxelBlockVolume;
    pool[base + threadIdx.x] = GeomVoxel{};
    if (has_color) {
      color_pool[base + threadIdx.x] = ColorVoxelT{};
    }
  }
}

template <TsdfVoxel VoxelT>
__global__ void count_observed_pool_kernel(const VoxelT* pool,
                                           std::size_t voxel_count,
                                           unsigned long long* result) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  unsigned long long count = 0;
  for (std::size_t index = first; index < voxel_count; index += stride) {
    if (pool[index].observed()) {
      ++count;
    }
  }
  if (count > 0) {
    atomicAdd(result, count);
  }
}

}  // namespace

template <typename GeomVoxel, typename Color>
BlockRep<MemorySpace::kDevice, GeomVoxel, Color>::BlockRep(
    const VolumeGeometry& geometry, std::size_t block_capacity)
    : geometry_(geometry),
      blocks_(BlockGrid::for_resolution(geometry.resolution)),
      capacity_(block_capacity != 0 ? block_capacity : blocks_.count() / 4),
      word_count_(BlockBitmapOps::word_count(blocks_.count())),
      grid_(blocks_.count()),
      pool_(cuda::DeviceBuffer<GeomVoxel>::uninitialized(capacity_ *
                                                         kVoxelBlockVolume)),
      color_pool_(cuda::DeviceBuffer<typename Color::Voxel>::uninitialized(
          Color::kEnabled ? capacity_ * kVoxelBlockVolume : 0)),
      bitmap_(word_count_),
      work_list_(blocks_.count()),
      new_list_(blocks_.count()) {
  cuda::check(
      cudaMemset(grid_.data(), 0xFF, grid_.size() * sizeof(std::uint32_t)),
      "sparse grid init");
  cuda::check(cudaMemset(allocated_.data(), 0, sizeof(unsigned int)),
              "sparse allocated init");
}

template <typename GeomVoxel, typename Color>
void BlockRep<MemorySpace::kDevice, GeomVoxel, Color>::integrate(
    const DepthFrameView<MemorySpace::kDevice>& frame,
    const TsdfIntegrationOptions& options, const TsdfRuleVariant& rule) {
  const IntegrationContext<MemorySpace::kDevice> context{frame, options};
  cuda::check(
      cudaMemsetAsync(bitmap_.data(), 0, word_count_ * sizeof(std::uint32_t)),
      "sparse bitmap clear");
  cuda::check(cudaMemsetAsync(work_count_.data(), 0, sizeof(unsigned int)),
              "sparse work count clear");
  cuda::check(cudaMemsetAsync(new_count_.data(), 0, sizeof(unsigned int)),
              "sparse new count clear");

  constexpr dim3 kBlock2d{16, 16};
  const dim3 mark_grid{cuda::ceil_div(frame.depth.width, kBlock2d.x),
                       cuda::ceil_div(frame.depth.height, kBlock2d.y)};

  band_walk::mark_band_blocks_kernel<<<mark_grid, kBlock2d>>>(
      context, inverse(frame.world_to_camera), geometry_, blocks_,
      band_walk::WarpAggregatedSet{bitmap_.data()});

  cuda::check(cudaGetLastError(), "mark_band_blocks_kernel launch");

  const unsigned int linear_grid =
      std::min(cuda::ceil_div(word_count_, band_walk::kLinearBlock),
               band_walk::kMaxGrid);

  allocate_blocks_kernel<<<linear_grid, band_walk::kLinearBlock>>>(
      bitmap_.data(), word_count_, grid_.data(),
      static_cast<std::uint32_t>(capacity_), work_list_.data(),
      work_count_.data(), new_list_.data(), new_count_.data(),
      allocated_.data());

  cuda::check(cudaGetLastError(), "allocate_blocks_kernel launch");

  init_blocks_kernel<<<band_walk::kMaxGrid, band_walk::kBlockThreads>>>(
      new_list_.data(), new_count_.data(), grid_.data(), pool_.data(),
      color_pool_.data(), Color::kEnabled);

  cuda::check(cudaGetLastError(), "init_blocks_kernel launch");

  const View volume = view();
  std::visit(
      [&](const auto& concrete) {
        band_walk::integrate_listed_blocks_kernel<<<band_walk::kMaxGrid,
                                                    band_walk::kBlockThreads>>>(
            volume, context, concrete, blocks_, work_list_.data(),
            work_count_.data());
      },
      rule);

  cuda::check(cudaGetLastError(), "integrate_listed_blocks_kernel launch");
}

template <typename GeomVoxel, typename Color>
std::size_t
BlockRep<MemorySpace::kDevice, GeomVoxel, Color>::observed_voxel_count() const {
  unsigned int allocated = 0;
  allocated_.copy_to_host(&allocated, 1);
  const std::size_t used =
      std::min<std::size_t>(allocated, capacity_) * kVoxelBlockVolume;

  cuda::DeviceBuffer<unsigned long long> result{1};
  const unsigned int grid = std::min(
      cuda::ceil_div(used, band_walk::kLinearBlock), band_walk::kMaxGrid);

  if (used > 0) {
    count_observed_pool_kernel<<<grid, band_walk::kLinearBlock>>>(
        pool_.data(), used, result.data());
    cuda::check(cudaGetLastError(), "count_observed_pool_kernel launch");
  }

  unsigned long long count = 0;
  result.copy_to_host(&count, 1);

  return static_cast<std::size_t>(count);
}

template <typename GeomVoxel, typename Color>
SparseVolumeSnapshot<GeomVoxel, Color>
BlockRep<MemorySpace::kDevice, GeomVoxel, Color>::host_snapshot() const {
  unsigned int allocated = 0;
  allocated_.copy_to_host(&allocated, 1);
  const std::size_t used = std::min<std::size_t>(allocated, capacity_);

  SparseVolumeSnapshot<GeomVoxel, Color> snapshot{
      .grid = std::vector<std::uint32_t>(grid_.size()),
      .pool = std::vector<GeomVoxel>(used * kVoxelBlockVolume),
      .color_pool = {},
      .geometry = geometry_,
      .blocks = blocks_};
  grid_.copy_to_host(snapshot.grid.data(), snapshot.grid.size());
  // Download only the allocated prefix of each pool, so plain cudaMemcpy
  // instead of the whole-buffer DeviceBuffer helper.
  if (!snapshot.pool.empty()) {
    cuda::check(cudaMemcpy(snapshot.pool.data(), pool_.data(),
                           snapshot.pool.size() * sizeof(GeomVoxel),
                           cudaMemcpyDeviceToHost),
                "sparse pool download");
  }
  if constexpr (Color::kEnabled) {
    snapshot.color_pool.resize(used * kVoxelBlockVolume);
    if (!snapshot.color_pool.empty()) {
      cuda::check(
          cudaMemcpy(snapshot.color_pool.data(), color_pool_.data(),
                     snapshot.color_pool.size() * sizeof(typename Color::Voxel),
                     cudaMemcpyDeviceToHost),
          "sparse color pool download");
    }
  }
  return snapshot;
}

template <typename GeomVoxel, typename Color>
ConstHostVolumeView
BlockRep<MemorySpace::kDevice, GeomVoxel, Color>::host_dense_view(
    std::optional<HostVolume>& staging) const {
  const SparseVolumeSnapshot<GeomVoxel, Color> snapshot = host_snapshot();
  const auto sparse = snapshot.view();

  staging.emplace(geometry_);
  auto dense = staging->view();
  for (const auto [x, y, z] : GridIndices{geometry_.resolution}) {
    const GeomVoxel* voxel = sparse.find_voxel(x, y, z);
    if (voxel == nullptr) {
      continue;
    }
    dense.voxel_at(x, y, z) =
        Voxel{.distance = voxel->tsdf(), .weight = voxel->weight_value()};
    if constexpr (Color::kEnabled) {
      dense.color_at(x, y, z) = *sparse.find_color_voxel(x, y, z);
    }
  }
  return staging->view();
}

// One explicit instantiation per registered storage combination.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KINECTFUSION_INSTANTIATE(GeomVoxel, Color) \
  template class BlockRep<MemorySpace::kDevice, GeomVoxel, Color>;
KINECTFUSION_FOR_EACH_REGISTERED_STORAGE(KINECTFUSION_INSTANTIATE)
#undef KINECTFUSION_INSTANTIATE

}  // namespace kinectfusion
