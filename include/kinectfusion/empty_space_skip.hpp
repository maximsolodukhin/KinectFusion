#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_EMPTY_SPACE_SKIP_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_EMPTY_SPACE_SKIP_HPP

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/occupancy.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <type_traits>

namespace kinectfusion {

// Ray math over the block grid of one volume. One instance is one ray in
// one volume, so the direction is examined in one place (axis_exit) and the
// queries take only block coordinates.
class BlockMarch {
 public:
  static constexpr float kInfinity = 1.0e30F;

  KINECTFUSION_HOST_DEVICE BlockMarch(const VolumeGeometry& geometry,
                                      const Vec3f& origin,
                                      const Vec3f& direction)
      : geometry_(geometry),
        block_size_(static_cast<float>(kVoxelBlockEdge) * geometry.voxel_size),
        origin_(origin),
        direction_(direction) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float block_size() const {
    return block_size_;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE Vec3f
  to_grid(const Vec3f& point) const {
    return (point - geometry_.origin) / geometry_.voxel_size;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE bool in_grid(
      const Vec3f& grid) const {
    return geometry_.resolution.contains(grid);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static GridIndex block_of(
      const Vec3f& grid) {
    return {.x = static_cast<std::size_t>(grid.x) / kVoxelBlockEdge,
            .y = static_cast<std::size_t>(grid.y) / kVoxelBlockEdge,
            .z = static_cast<std::size_t>(grid.z) / kVoxelBlockEdge};
  }

  // True when the ray walks up the z axis.
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE bool forward_z() const {
    return direction_.z >= 0.0F;
  }

  // Ray parameter where the ray enters z-block layer `block_z`.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float z_entry(
      std::size_t block_z) const {
    const float side = forward_z() ? 0.0F : 1.0F;
    const float plane = geometry_.origin.z +
                        ((static_cast<float>(block_z) + side) * block_size_);
    return (plane - origin_.z) / direction_.z;
  }

  // Ray parameter where the ray leaves the block.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float block_exit(
      const GridIndex& block) const {
    const Vec3f low = geometry_.origin +
                      (make_vec3f(block.x, block.y, block.z) * block_size_);
    const Vec3f high = low + make_vec3f(1.0F, 1.0F, 1.0F) * block_size_;
    return compat::min(
        axis_exit(origin_.x, direction_.x, low.x, high.x),
        compat::min(axis_exit(origin_.y, direction_.y, low.y, high.y),
                    axis_exit(origin_.z, direction_.z, low.z, high.z)));
  }

  // Ray parameter where the ray leaves the (block_x, block_y) block column.
  // Axes parallel to the ray never exit: kInfinity.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float column_exit(
      std::size_t block_x, std::size_t block_y) const {
    const float low_x =
        geometry_.origin.x + (static_cast<float>(block_x) * block_size_);
    const float low_y =
        geometry_.origin.y + (static_cast<float>(block_y) * block_size_);
    float exit = kInfinity;
    if (direction_.x != 0.0F) {
      exit = compat::min(
          exit, axis_exit(origin_.x, direction_.x, low_x, low_x + block_size_));
    }
    if (direction_.y != 0.0F) {
      exit = compat::min(
          exit, axis_exit(origin_.y, direction_.y, low_y, low_y + block_size_));
    }
    return exit;
  }

  // Slab-test entry into the volume. Returns kInfinity on a miss.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float volume_entry() const {
    const Vec3f low = geometry_.origin;
    const Vec3f high =
        low + (make_vec3f(geometry_.resolution.x, geometry_.resolution.y,
                          geometry_.resolution.z) *
               geometry_.voxel_size);
    float entry = 0.0F;
    float exit = kInfinity;
    slab(origin_.x, direction_.x, low.x, high.x, entry, exit);
    slab(origin_.y, direction_.y, low.y, high.y, entry, exit);
    slab(origin_.z, direction_.z, low.z, high.z, entry, exit);
    return entry <= exit ? entry : kInfinity;
  }

 private:
  // The one place that selects an exit face: the high face for a ray that
  // goes up the axis, the low face for a ray that goes down.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE static float axis_exit(float origin,
                                                                float direction,
                                                                float low,
                                                                float high) {
    return ((direction >= 0.0F ? high : low) - origin) / direction;
  }

  KINECTFUSION_HOST_DEVICE static void slab(float origin, float direction,
                                            float low, float high, float& entry,
                                            float& exit) {
    if (direction == 0.0F) {
      if (origin < low || origin > high) {
        entry = kInfinity;
      }
      return;
    }
    const float t_low = (low - origin) / direction;
    const float t_high = (high - origin) / direction;
    entry = compat::max(entry, compat::min(t_low, t_high));
    exit = compat::min(exit, compat::max(t_low, t_high));
  }

  VolumeGeometry geometry_;
  float block_size_;
  Vec3f origin_;
  Vec3f direction_;
};

// Default skip policy: the plain march.
struct NoSkip {
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static float advance(
      const Vec3f& /*origin*/, const Vec3f& /*direction*/, float ray_length,
      float /*base_step*/, float /*max_depth*/) {
    return ray_length;
  }
};

// Moves the ray across empty dilated blocks. An empty dilated block contains
// no zero crossing. The advance replays the base_step additions of the plain
// march, so the returned ray lengths are bit-identical to it.
struct BitmapSkip {
  OccupancyView occupancy{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE float advance(const Vec3f& origin,
                                                       const Vec3f& direction,
                                                       float ray_length,
                                                       float base_step,
                                                       float max_depth) const {
    const BlockMarch march{occupancy.geometry, origin, direction};
    while (ray_length <= max_depth) {
      const Vec3f point = origin + (ray_length * direction);
      const Vec3f grid = march.to_grid(point);
      float target = 0.0F;
      if (march.in_grid(grid)) {
        const GridIndex block = BlockMarch::block_of(grid);
        if (occupancy.occupied(block.x, block.y, block.z)) {
          return ray_length;
        }
        target = march.block_exit(block);
      } else {
        target = march.volume_entry();
      }

      const float before = ray_length;
      // false for the miss sentinel and NaN edges
      if (target < BlockMarch::kInfinity) {
        // Replay base_step additions to keep the sample phase bit-identical.
        while (ray_length < target && ray_length <= max_depth) {
          ray_length += base_step;
        }
      }
      if (ray_length == before) {
        ray_length += base_step;  // numeric edge (boundary/NaN): plain step
      }
    }
    return ray_length;
  }
};

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE int find_set_at_or_after(
    std::uint32_t word, unsigned int offset) {
  const std::uint32_t masked = word & ~((1U << offset) - 1U);
#ifdef __CUDA_ARCH__
  return masked == 0U ? -1 : __ffs(masked) - 1;
#else
  return masked == 0U ? -1 : std::countr_zero(masked);
#endif
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE int find_set_at_or_before(
    std::uint32_t word, unsigned int offset) {
  constexpr unsigned int kTopBit = kBitmapWordBits - 1U;
  const std::uint32_t masked =
      word & (offset >= kTopBit ? ~0U : ((1U << (offset + 1U)) - 1U));
#ifdef __CUDA_ARCH__
  return masked == 0U ? -1 : static_cast<int>(kTopBit) - __clz(masked);
#else
  return masked == 0U ? -1
                      : static_cast<int>(kTopBit) - std::countl_zero(masked);
#endif
}

// Approximate skip over the dilated band bitmap: the ray moves to the entry
// of the next band block. Skipped space is far-positive or unobserved, so
// only the interpolation endpoints shift by a sub-voxel amount. The output
// is not bit-identical to the plain march.
struct BandSkip {
  BandView band{};

  [[nodiscard]] KINECTFUSION_HOST_DEVICE float advance(const Vec3f& origin,
                                                       const Vec3f& direction,
                                                       float ray_length,
                                                       float base_step,
                                                       float max_depth) const {
    const BlockMarch march{band.geometry, origin, direction};
    const float nudge = 0.25F * band.geometry.voxel_size;
    const bool z_major = std::abs(direction.z) >= std::abs(direction.x) &&
                         std::abs(direction.z) >= std::abs(direction.y) &&
                         (band.blocks.z % kBitmapWordBits) == 0U;
    while (ray_length <= max_depth) {
      const Vec3f point = origin + (ray_length * direction);
      const Vec3f grid = march.to_grid(point);
      if (!march.in_grid(grid)) {
        const float entry = march.volume_entry();
        if (entry >= BlockMarch::kInfinity || entry > max_depth) {
          return max_depth + base_step;
        }
        ray_length = compat::max(ray_length + base_step, entry + nudge);
        continue;
      }
      const GridIndex block = BlockMarch::block_of(grid);
      if (band.occupied(block.x, block.y, block.z)) {
        return ray_length;
      }
      float target = BlockMarch::kInfinity;
      if (z_major) {
        target = word_scan_z(block, march);
      }
      if (target >= BlockMarch::kInfinity) {
        target = march.block_exit(block);
      }
      ray_length = compat::max(ray_length + base_step, target + nudge);
    }
    return ray_length;
  }

 private:
  // Move to the next band bit along z in one word. Clamp the result to the
  // exit of the (x, y) block column: only z may vary inside one scanned
  // word.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE float word_scan_z(
      const GridIndex& block, const BlockMarch& march) const {
    const std::size_t index = band.block_index(block.x, block.y, block.z);
    const std::uint32_t word = band.word_at(index);
    const auto offset = static_cast<unsigned int>(index % kBitmapWordBits);
    const bool forward = march.forward_z();
    const int found = forward ? find_set_at_or_after(word, offset)
                              : find_set_at_or_before(word, offset);

    std::size_t target_z = 0;
    if (found >= 0) {
      target_z = block.z + static_cast<std::size_t>(found) - offset;
    } else {
      // No set bit in the scan direction: move to the far end of the word.
      target_z = forward ? block.z + (kBitmapWordBits - offset)
                         : block.z - compat::min<std::size_t>(offset, block.z);
    }
    return compat::min(march.z_entry(target_z),
                       march.column_exit(block.x, block.y));
  }
};

// advance() returns ray_length unchanged in occupied space. In empty space
// it returns the next ray length to sample.
template <typename S>
concept SkipPolicy = std::is_trivially_copyable_v<S> &&
                     requires(const S skip, const Vec3f& point, float value) {
                       {
                         skip.advance(point, point, value, value, value)
                       } -> std::same_as<float>;
                     };

static_assert(SkipPolicy<NoSkip>);
static_assert(SkipPolicy<BitmapSkip>);
static_assert(SkipPolicy<BandSkip>);

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_EMPTY_SPACE_SKIP_HPP */
