#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_GRID_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_GRID_HPP

#include <cstddef>
#include <iterator>
#include <kinectfusion/vector.hpp>
#include <ranges>

namespace kinectfusion {

using GridIndex = Size3;

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

static_assert(std::forward_iterator<GridIndices::Iterator>);
static_assert(std::ranges::forward_range<GridIndices>);
static_assert(std::forward_iterator<PixelIndices::Iterator>);
static_assert(std::ranges::forward_range<PixelIndices>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_GRID_HPP */
