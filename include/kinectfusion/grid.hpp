#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_GRID_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_GRID_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <kinectfusion/vector.hpp>
#include <ranges>
#include <type_traits>

namespace kinectfusion {

using GridIndex = Size3;

inline constexpr std::size_t kVoxelBlockEdge = 8;
inline constexpr std::size_t kVoxelBlockVolume =
    kVoxelBlockEdge * kVoxelBlockEdge * kVoxelBlockEdge;

// The grid of kVoxelBlockEdge^3 voxel blocks over a voxel resolution. All
// block consumers share its x-major flat block indexing.
class BlockGrid {
 public:
  BlockGrid() = default;
  explicit KINECTFUSION_FORCEINLINE_DEVICE BlockGrid(const Size3& blocks)
      : blocks_(blocks) {}

  [[nodiscard]] static BlockGrid for_resolution(const Size3& resolution) {
    const auto blocks_along = [](std::size_t voxels) {
      return (voxels + kVoxelBlockEdge - 1) / kVoxelBlockEdge;
    };
    return BlockGrid{Size3{.x = blocks_along(resolution.x),
                           .y = blocks_along(resolution.y),
                           .z = blocks_along(resolution.z)}};
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE const Size3& extent() const {
    return blocks_;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE std::size_t count() const {
    return blocks_.x * blocks_.y * blocks_.z;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE std::size_t flat_of(
      const GridIndex& block) const {
    return blocks_.flatten(block.x, block.y, block.z);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE GridIndex
  coords_of(std::size_t flat) const {
    return blocks_.unflatten(flat);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE std::size_t block_of_voxel(
      std::size_t x, std::size_t y, std::size_t z) const {
    return flat_of({.x = x / kVoxelBlockEdge,
                    .y = y / kVoxelBlockEdge,
                    .z = z / kVoxelBlockEdge});
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static std::size_t
  intra_of_voxel(std::size_t x, std::size_t y, std::size_t z) {
    constexpr std::size_t kBlockSz = kVoxelBlockEdge;

    const std::size_t zidx = z % kBlockSz;
    const std::size_t yidx = y % kBlockSz;

    return (((zidx * kBlockSz) + yidx) * kBlockSz) + (x % kBlockSz);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE GridIndex
  voxel_base(std::size_t flat) const {
    const GridIndex block = coords_of(flat);
    return {.x = block.x * kVoxelBlockEdge,
            .y = block.y * kVoxelBlockEdge,
            .z = block.z * kVoxelBlockEdge};
  }

 private:
  Size3 blocks_;
};

static_assert(std::is_trivially_copyable_v<BlockGrid>);

// Bounds-checked integer pixel coordinate for image lookups.
struct Pixel {
  std::size_t x{};
  std::size_t y{};

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE Vec2f as_vector() const {
    return Vec2f{.x = static_cast<float>(x), .y = static_cast<float>(y)};
  }

  friend bool operator==(const Pixel&, const Pixel&) = default;
};

// The 3D index space of a voxel grid in storage order, x fastest.
class GridIndices {
 public:
  class Iterator {
   public:
    using value_type = GridIndex;
    using difference_type = std::ptrdiff_t;

    Iterator() = default;
    Iterator(GridIndex current, Size3 extent)
        : current_(current), extent_(extent) {}

    [[nodiscard]] GridIndex operator*() const { return current_; }

    Iterator& operator++() {
      if (++current_.x < extent_.x) {
        return *this;
      }
      current_.x = 0;
      if (++current_.y < extent_.y) {
        return *this;
      }
      current_.y = 0;
      ++current_.z;
      return *this;
    }

    Iterator operator++(int) {
      Iterator copy = *this;
      ++*this;
      return copy;
    }

    [[nodiscard]] friend bool operator==(const Iterator& lhs,
                                         const Iterator& rhs) {
      return lhs.current_ == rhs.current_;
    }

   private:
    GridIndex current_{};
    Size3 extent_{};
  };

  explicit GridIndices(Size3 extent) : extent_(normalized(extent)) {}

  [[nodiscard]] Iterator begin() const {
    return Iterator{GridIndex{}, extent_};
  }

  [[nodiscard]] Iterator end() const {
    return Iterator{GridIndex{.z = extent_.z}, extent_};
  }

 private:
  // An empty dimension empties the whole range instead of trapping the
  // x/y carry logic in the iterator
  [[nodiscard]] static Size3 normalized(Size3 extent) {
    if (extent.x == 0 || extent.y == 0 || extent.z == 0) {
      return Size3{};
    }
    return extent;
  }

  Size3 extent_;
};

// The 2D index space of an image in storage order, x fastest
// GridIndices but z=1
class PixelIndices {
 public:
  class Iterator {
   public:
    using value_type = Pixel;
    using difference_type = std::ptrdiff_t;

    Iterator() = default;
    explicit Iterator(GridIndices::Iterator inner) : inner_(inner) {}

    [[nodiscard]] Pixel operator*() const {
      const GridIndex index = *inner_;
      return Pixel{.x = index.x, .y = index.y};
    }

    Iterator& operator++() {
      ++inner_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator copy = *this;
      ++*this;
      return copy;
    }

    [[nodiscard]] friend bool operator==(const Iterator&,
                                         const Iterator&) = default;

   private:
    GridIndices::Iterator inner_;
  };

  PixelIndices(std::size_t width, std::size_t height)
      : grid_({.x = width, .y = height, .z = 1}) {}

  [[nodiscard]] Iterator begin() const { return Iterator{grid_.begin()}; }
  [[nodiscard]] Iterator end() const { return Iterator{grid_.end()}; }

 private:
  GridIndices grid_;
};

class BlockVoxels {
 public:
  class Iterator {
   public:
    using value_type = GridIndex;
    using difference_type = std::ptrdiff_t;

    Iterator() = default;
    Iterator(GridIndices::Iterator inner, GridIndex base)
        : inner_(inner), base_(base) {}

    [[nodiscard]] GridIndex operator*() const {
      const GridIndex offset = *inner_;

      return GridIndex{.x = base_.x + offset.x,
                       .y = base_.y + offset.y,
                       .z = base_.z + offset.z};
    }

    Iterator& operator++() {
      ++inner_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator copy = *this;
      ++*this;

      return copy;
    }

    [[nodiscard]] friend bool operator==(const Iterator& lhs,
                                         const Iterator& rhs) {
      return lhs.inner_ == rhs.inner_;
    }

   private:
    GridIndices::Iterator inner_;
    GridIndex base_{};
  };

  BlockVoxels(std::size_t block, const BlockGrid& blocks,
              const Size3& resolution)
      : base_(blocks.voxel_base(block)),
        inner_(clamped_extent(base_, resolution)) {}

  [[nodiscard]] Iterator begin() const {
    return Iterator{inner_.begin(), base_};
  }
  [[nodiscard]] Iterator end() const { return Iterator{inner_.end(), base_}; }

 private:
  [[nodiscard]] static Size3 clamped_extent(const GridIndex& base,
                                            const Size3& resolution) {
    const auto axis = [](std::size_t start, std::size_t bound) {
      return start < bound ? std::min(kVoxelBlockEdge, bound - start)
                           : std::size_t{0};
    };

    return Size3{.x = axis(base.x, resolution.x),
                 .y = axis(base.y, resolution.y),
                 .z = axis(base.z, resolution.z)};
  }

  GridIndex base_;
  GridIndices inner_;
};

static_assert(std::forward_iterator<GridIndices::Iterator>);
static_assert(std::ranges::forward_range<GridIndices>);
static_assert(std::forward_iterator<PixelIndices::Iterator>);
static_assert(std::ranges::forward_range<PixelIndices>);
static_assert(std::forward_iterator<BlockVoxels::Iterator>);
static_assert(std::ranges::forward_range<BlockVoxels>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_GRID_HPP */
