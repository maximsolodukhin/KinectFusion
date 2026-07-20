#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_BLOCK_REP_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_BLOCK_REP_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/trilinear.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_representation.hpp>
#include <kinectfusion/volume_sampler.hpp>
#include <optional>
#include <type_traits>
#include <vector>

namespace kinectfusion {

// Sparse storage: a dense grid of block slots over pools of 8^3 voxel
// blocks. The truncation band of the depth image drives allocation, so the
// sweep never touches far free space.
inline constexpr std::uint32_t kUnallocatedBlock = 0xFFFFFFFFU;

template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false,
          typename GeomVoxel = Voxel, typename Color = FloatColorFacet>
struct SparseVolumeView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  using GeometryVoxel = GeomVoxel;
  using ColorFacet = Color;

  Pointee<std::uint32_t>* grid{};  // pool slot per block, or kUnallocatedBlock
  Pointee<GeomVoxel>* pool{};
  Pointee<typename Color::Voxel>* color_pool{};
  VolumeGeometry geometry{};
  BlockGrid blocks;
  std::uint32_t capacity{};

  static constexpr MemorySpace kMemorySpace = Space;

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Size3& resolution() const {
    return geometry.resolution;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float voxel_size() const {
    return geometry.voxel_size;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE const Vec3f& origin() const {
    return geometry.origin;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float truncation_distance() const {
    return geometry.truncation_distance;
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec3f
  cell_center(std::size_t x, std::size_t y, std::size_t z) const {
    return geometry.cell_center(x, y, z);
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t block_index(
      std::size_t x, std::size_t y, std::size_t z) const {
    return blocks.block_of_voxel(x, y, z);
  }

  // Pointer to the voxel of an in-bounds coordinate, or nullptr when its
  // block is not allocated.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<GeomVoxel>* find_voxel(
      std::size_t x, std::size_t y, std::size_t z) const {
    return pool_at(pool, x, y, z);
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<typename Color::Voxel>*
  find_color_voxel(std::size_t x, std::size_t y, std::size_t z) const {
    return pool_at(color_pool, x, y, z);
  }

  // Precondition: the block of (x, y, z) is allocated.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<GeomVoxel>& voxel_at(
      std::size_t x, std::size_t y, std::size_t z) const {
    return *find_voxel(x, y, z);
  }
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Pointee<typename Color::Voxel>&
  color_at(std::size_t x, std::size_t y, std::size_t z) const {
    return *find_color_voxel(x, y, z);
  }

  template <typename T>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE T* pool_at(T* target_pool,
                                                    std::size_t x,
                                                    std::size_t y,
                                                    std::size_t z) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::uint32_t slot = grid[block_index(x, y, z)];
    if (slot == kUnallocatedBlock) {
      return nullptr;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return target_pool + (static_cast<std::size_t>(slot) * kVoxelBlockVolume) +
           BlockGrid::intra_of_voxel(x, y, z);
  }

  template <bool TargetConst = true>
    requires(TargetConst && !IsConst)
  [[nodiscard]] KINECTFUSION_HOST_DEVICE
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator SparseVolumeView<Space, TargetConst, GeomVoxel, Color>() const {
    return {.grid = grid,
            .pool = pool,
            .color_pool = color_pool,
            .geometry = geometry,
            .blocks = blocks,
            .capacity = capacity};
  }
};

// Corner access for the sparse pointer grid: each corner is one clipped
// block lookup. The trilinear blends read an unallocated corner as a
// missing corner.
template <typename ViewT>
class SparseCornerAccess {
 public:
  using View = ViewT;

  explicit KINECTFUSION_HOST_DEVICE SparseCornerAccess(const ViewT& view)
      : view_(view) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE const ViewT& view() const {
    return view_;
  }

  template <typename Use>
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE decltype(auto) with_corners(
      const Vec3i& base, const Use& use) const {
    return use(
        trilinear::clipped_corners(view_.resolution(), base, VoxelAt{&view_}));
  }

  template <typename Use>
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE decltype(auto)
  with_color_corners(const Vec3i& base, const Use& use) const {
    return use(trilinear::clipped_corners(view_.resolution(), base,
                                          ColorVoxelAt{&view_}));
  }

 private:
  struct VoxelAt {
    const ViewT* view;
    [[nodiscard]] KINECTFUSION_HOST_DEVICE const ViewT::GeometryVoxel*
    operator()(std::size_t x, std::size_t y, std::size_t z) const {
      return view->find_voxel(x, y, z);
    }
  };

  struct ColorVoxelAt {
    const ViewT* view;
    [[nodiscard]] KINECTFUSION_HOST_DEVICE const ViewT::ColorFacet::Voxel*
    operator()(std::size_t x, std::size_t y, std::size_t z) const {
      return view->find_color_voxel(x, y, z);
    }
  };

  ViewT view_;
};

template <MemorySpace Space = MemorySpace::kHost, typename GeomVoxel = Voxel,
          typename Color = FloatColorFacet>
using SparseVolumeSampler = TrilinearSampler<
    SparseCornerAccess<SparseVolumeView<Space, true, GeomVoxel, Color>>>;

template <MemorySpace Space, typename GeomVoxel = Voxel,
          typename Color = FloatColorFacet>
class BlockRep;

template <typename GeomVoxel, typename Color>
class BlockRep<MemorySpace::kHost, GeomVoxel, Color> {
  static constexpr MemorySpace kSpace = MemorySpace::kHost;

 public:
  using Sampler = SparseVolumeSampler<kSpace, GeomVoxel, Color>;
  using View = SparseVolumeView<kSpace, false, GeomVoxel, Color>;

  explicit BlockRep(const VolumeGeometry& geometry,
                    std::size_t block_capacity = 0)
      : geometry_(geometry),
        blocks_(BlockGrid::for_resolution(geometry.resolution)),
        capacity_(block_capacity != 0 ? block_capacity : blocks_.count() / 4),
        grid_(blocks_.count(), kUnallocatedBlock),
        pool_(capacity_ * kVoxelBlockVolume),
        color_pool_(Color::kEnabled ? capacity_ * kVoxelBlockVolume : 0),
        marked_frame_(grid_.size(), 0) {}

  [[nodiscard]] Sampler sampler() const { return Sampler{view()}; }

  [[nodiscard]] const VolumeGeometry& geometry() const { return geometry_; }

  [[nodiscard]] SparseVolumeView<kSpace, true, GeomVoxel, Color> view() const {
    return {.grid = grid_.data(),
            .pool = pool_.data(),
            .color_pool = color_pool_.data(),
            .geometry = geometry_,
            .blocks = blocks_,
            .capacity = static_cast<std::uint32_t>(capacity_)};
  }
  [[nodiscard]] View view() {
    return {.grid = grid_.data(),
            .pool = pool_.data(),
            .color_pool = color_pool_.data(),
            .geometry = geometry_,
            .blocks = blocks_,
            .capacity = static_cast<std::uint32_t>(capacity_)};
  }

  [[nodiscard]] std::size_t observed_voxel_count() const {
    std::size_t count = 0;
    const std::size_t used =
        std::min(allocated_, capacity_) * kVoxelBlockVolume;
    for (std::size_t index = 0; index < used; ++index) {
      if (pool_[index].observed()) {
        ++count;
      }
    }
    return count;
  }

  void integrate(const DepthFrameView<kSpace>& frame,
                 const TsdfIntegrationOptions& options,
                 const TsdfRuleVariant& rule) {
    const IntegrationContext<kSpace> context{frame, options};
    mark_and_allocate(frame, options);
    const View volume = view();
    std::visit(
        [&](const auto& chosen) {
          for (const std::uint32_t block : work_list_) {
            integrate_block(volume, context, chosen, block);
          }
        },
        rule);
  }

  // Unallocated voxels stay unobserved in the dense output.
  [[nodiscard]] ConstHostVolumeView host_dense_view(
      std::optional<HostVolume>& staging) const {
    staging.emplace(geometry_);
    auto dense = staging->view();
    const auto sparse = view();
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

  [[nodiscard]] bool overflowed() const { return allocated_ > capacity_; }

 private:
  void mark_and_allocate(const DepthFrameView<kSpace>& frame,
                         const TsdfIntegrationOptions& options) {
    ++frame_stamp_;
    work_list_.clear();
    TruncationBandWalk::visit(frame, options, geometry_, blocks_,
                              TruncationBandWalk::half_block_step(geometry_),
                              [this](std::size_t block) { allocate(block); });
  }

  void allocate(std::size_t block) {
    if (grid_[block] != kUnallocatedBlock) {
      if (marked_frame_[block] != frame_stamp_) {
        marked_frame_[block] = frame_stamp_;
        work_list_.push_back(static_cast<std::uint32_t>(block));
      }
      return;
    }
    if (allocated_ >= capacity_) {
      ++allocated_;  // records the overflow
      return;
    }
    const std::size_t slot = allocated_++;
    grid_[block] = static_cast<std::uint32_t>(slot);
    marked_frame_[block] = frame_stamp_;
    work_list_.push_back(static_cast<std::uint32_t>(block));
  }

  template <TsdfUpdateRule<kSpace> Rule>
  void integrate_block(const View& volume,
                       const IntegrationContext<kSpace>& context,
                       const Rule& rule, std::uint32_t block) const {
    for (const auto [x, y, z] :
         BlockVoxels{block, blocks_, geometry_.resolution}) {
      rule.update(volume, context, x, y, z);
    }
  }

  VolumeGeometry geometry_;
  BlockGrid blocks_;
  std::size_t capacity_;
  std::vector<std::uint32_t> grid_;
  std::vector<GeomVoxel> pool_;
  std::vector<typename Color::Voxel> color_pool_;
  std::vector<std::uint32_t> marked_frame_;
  std::vector<std::uint32_t> work_list_;
  std::size_t allocated_{0};
  std::uint32_t frame_stamp_{1};
};

using HostBlockRep = BlockRep<MemorySpace::kHost>;

static_assert(VolumeRepresentation<HostBlockRep, MemorySpace::kHost>);
static_assert(CornerAccessPolicy<SparseCornerAccess<SparseVolumeView<>>>);
static_assert(
    TsdfVolumeSampler<SparseVolumeSampler<MemorySpace::kHost, Bf16Voxel>>);
static_assert(VoxelGridView<SparseVolumeView<>>);
static_assert(VoxelGridView<SparseVolumeView<MemorySpace::kHost, true>>);
static_assert(!DenseVoxelGridView<SparseVolumeView<>>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_BLOCK_REP_HPP */
