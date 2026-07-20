#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_OCCUPANCY_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_OCCUPANCY_HPP

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <span>
#include <type_traits>

namespace kinectfusion {

// One bit for each kVoxelBlockEdge^3 voxel block. At 512^3 voxels the bitmap
// is 32 KB and stays in the L2 cache.
inline constexpr std::size_t kBitmapWordBits = 32;

// A block bitmap view. Layouts differ only in the bit order of block_index.
template <typename V>
concept BlockBitmapView = requires(const V view, std::size_t index) {
  { view.blocks } -> std::convertible_to<const Size3&>;
  { view.block_index(index, index, index) } -> std::same_as<std::size_t>;
  { view.block_coords(index) } -> std::same_as<GridIndex>;
  { view.occupied(index, index, index) } -> std::same_as<bool>;
};

// Bit primitives and the mark and dilate passes shared by all block bitmaps.
struct BlockBitmapOps {
  [[nodiscard]] static constexpr std::size_t word_count(std::size_t bits) {
    return (bits + kBitmapWordBits - 1) / kBitmapWordBits;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE static bool test(
      const std::uint32_t* words, std::size_t index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return (words[index / kBitmapWordBits] &
            (1U << (index % kBitmapWordBits))) != 0U;
  }

  KINECTFUSION_HOST_DEVICE static void set(std::uint32_t* words,
                                           std::size_t index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    words[index / kBitmapWordBits] |= 1U << (index % kBitmapWordBits);
  }

  template <std::invocable<std::size_t> Visit>
  KINECTFUSION_HOST_DEVICE static void for_each_set_bit(std::uint32_t word,
                                                        std::size_t base,
                                                        const Visit& visit) {
    while (word != 0U) {
#ifdef __CUDA_ARCH__
      const int bit = __ffs(static_cast<int>(word)) - 1;
#else
      const int bit = std::countr_zero(word);
#endif
      word &= word - 1U;
      visit(base + static_cast<std::size_t>(bit));
    }
  }

  // True if a block of the clipped 3^3 neighborhood is occupied. A trilinear
  // stencil can read corners from an adjacent block, so an unset dilated bit
  // guarantees nullopt samples.
  template <BlockBitmapView View>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE static bool neighborhood_occupied(
      const View& view, std::size_t block_x, std::size_t block_y,
      std::size_t block_z) {
    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
      for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
          const auto neighbor_x = static_cast<long long>(block_x) + offset_x;
          const auto neighbor_y = static_cast<long long>(block_y) + offset_y;
          const auto neighbor_z = static_cast<long long>(block_z) + offset_z;

          if (!view.blocks.contains(neighbor_x, neighbor_y, neighbor_z)) {
            continue;
          }

          if (view.occupied(static_cast<std::size_t>(neighbor_x),
                            static_cast<std::size_t>(neighbor_y),
                            static_cast<std::size_t>(neighbor_z))) {
            return true;
          }
        }
      }
    }
    return false;
  }

  template <BlockBitmapView View>
  static void dilate(const View& raw, std::uint32_t* dilated) {
    for (const auto [block_x, block_y, block_z] : GridIndices{raw.blocks}) {
      if (neighborhood_occupied(raw, block_x, block_y, block_z)) {
        set(dilated, raw.block_index(block_x, block_y, block_z));
      }
    }
  }

  template <BlockBitmapView LayoutView, DenseVoxelGridView VolumeViewT,
            std::predicate<const typename VolumeViewT::GeometryVoxel&> InSet>
  static void rebuild(std::uint32_t* raw, std::uint32_t* dilated,
                      std::size_t word_count, const LayoutView& raw_view,
                      const VolumeViewT& volume, const InSet& in_set) {
    std::ranges::fill(std::span{raw, word_count}, 0U);
    std::ranges::fill(std::span{dilated, word_count}, 0U);

    for (const auto [x, y, z] : GridIndices{volume.resolution()}) {
      if (in_set(volume.voxel_at(x, y, z))) {
        set(raw, raw_view.block_index(x / kVoxelBlockEdge, y / kVoxelBlockEdge,
                                      z / kVoxelBlockEdge));
      }
    }

    dilate(raw_view, dilated);
  }
};

struct OccupancyView {
  const std::uint32_t* words{};
  Size3 blocks{};
  VolumeGeometry geometry{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t block_index(
      std::size_t block_x, std::size_t block_y, std::size_t block_z) const {
    return BlockGrid{blocks}.flat_of(
        {.x = block_x, .y = block_y, .z = block_z});
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE GridIndex
  block_coords(std::size_t flat) const {
    return blocks.unflatten(flat);
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool occupied(
      std::size_t block_x, std::size_t block_y, std::size_t block_z) const {
    return BlockBitmapOps::test(words, block_index(block_x, block_y, block_z));
  }
};

static_assert(BlockBitmapView<OccupancyView>);

// The projective TSDF writes exactly 1.0 beyond the truncation band. Thus
// kBandEpsilon must only exceed the storage quantum near 1.0 (bf16: 1/256).
inline constexpr float kBandEpsilon = 1.0F / 64.0F;

template <TsdfVoxel VoxelT>
[[nodiscard]] KINECTFUSION_HOST_DEVICE bool in_band(const VoxelT& voxel) {
  return voxel.observed() && voxel.tsdf() < (1.0F - kBandEpsilon);
}

// The bitmap membership predicates, shared by the host and device rebuilds.
struct ObservedFilter {
  template <TsdfVoxel VoxelT>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool operator()(
      const VoxelT& voxel) const {
    return voxel.observed();
  }
};

struct BandFilter {
  template <TsdfVoxel VoxelT>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool operator()(
      const VoxelT& voxel) const {
    return in_band(voxel);
  }
};

// z-major bit layout: 32 consecutive z-blocks share one word. A ray with a
// dominant z axis tests 32 blocks with one load.
struct BandView {
  const std::uint32_t* words{};
  Size3 blocks{};
  VolumeGeometry geometry{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::size_t block_index(
      std::size_t block_x, std::size_t block_y, std::size_t block_z) const {
    return (((block_x * blocks.y) + block_y) * blocks.z) + block_z;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE GridIndex
  block_coords(std::size_t flat) const {
    const Size3 zmajor =
        Size3{.x = blocks.z, .y = blocks.y, .z = blocks.x}.unflatten(flat);
    return {.x = zmajor.z, .y = zmajor.y, .z = zmajor.x};
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE bool occupied(
      std::size_t block_x, std::size_t block_y, std::size_t block_z) const {
    return BlockBitmapOps::test(words, block_index(block_x, block_y, block_z));
  }

  // The word that holds bit `index`. The bit offset (index % 32) walks +z.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE std::uint32_t word_at(
      std::size_t index) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return words[index / kBitmapWordBits];
  }
};

static_assert(BlockBitmapView<BandView>);

// Marks and dilates in one call. The CUDA backend defines the device
// specialization.
template <MemorySpace Space>
struct OccupancyRebuild;

template <>
struct OccupancyRebuild<MemorySpace::kHost> {
  template <DenseVoxelGridView VolumeViewT>
  static void run(std::uint32_t* raw, std::uint32_t* dilated,
                  std::size_t word_count, const Size3& blocks,
                  const VolumeViewT& volume) {
    const OccupancyView raw_view{
        .words = raw, .blocks = blocks, .geometry = volume.geometry};
    BlockBitmapOps::rebuild(raw, dilated, word_count, raw_view, volume,
                            ObservedFilter{});
  }
};

// Band variant: z-major words over the in_band predicate.
template <MemorySpace Space>
struct BandRebuild;

template <>
struct BandRebuild<MemorySpace::kHost> {
  template <DenseVoxelGridView VolumeViewT>
  static void run(std::uint32_t* raw, std::uint32_t* dilated,
                  std::size_t word_count, const Size3& blocks,
                  const VolumeViewT& volume) {
    const BandView raw_view{
        .words = raw, .blocks = blocks, .geometry = volume.geometry};
    BlockBitmapOps::rebuild(raw, dilated, word_count, raw_view, volume,
                            BandFilter{});
  }
};

// Owns the raw and dilated word buffers. The raycast reads the dilated view.
// rebuild() refreshes both buffers after integration.
template <MemorySpace Space, typename ViewT,
          template <MemorySpace> class Rebuild>
class BasicBlockBitmap {
  using WordBuffer = SpaceTraits<Space>::template Buffer<std::uint32_t>;

 public:
  explicit BasicBlockBitmap(const VolumeGeometry& geometry)
      : blocks_(BlockGrid::for_resolution(geometry.resolution)),
        geometry_(geometry),
        word_count_(BlockBitmapOps::word_count(blocks_.count())),
        raw_(allocate_words(word_count_)),
        dilated_(allocate_words(word_count_)) {}

  template <DenseVoxelGridView VolumeViewT>
  void rebuild(const VolumeViewT& volume) {
    Rebuild<Space>::run(raw_.data(), dilated_.data(), word_count_,
                        blocks_.extent(), volume);
  }

  [[nodiscard]] ViewT view() const {
    return {.words = dilated_.data(),
            .blocks = blocks_.extent(),
            .geometry = geometry_};
  }

 private:
  [[nodiscard]] static WordBuffer allocate_words(std::size_t count) {
    return SpaceTraits<Space>::template allocate<std::uint32_t>(count);
  }

  BlockGrid blocks_;
  VolumeGeometry geometry_;
  std::size_t word_count_;
  WordBuffer raw_;
  WordBuffer dilated_;
};

template <MemorySpace Space>
using BasicOccupancyBitmap =
    BasicBlockBitmap<Space, OccupancyView, OccupancyRebuild>;
template <MemorySpace Space>
using BasicBandBitmap = BasicBlockBitmap<Space, BandView, BandRebuild>;

using HostOccupancyBitmap = BasicOccupancyBitmap<MemorySpace::kHost>;
using HostBandBitmap = BasicBandBitmap<MemorySpace::kHost>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_OCCUPANCY_HPP */
