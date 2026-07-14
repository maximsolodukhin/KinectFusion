#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_COMPARISON_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_COMPARISON_HPP

#include <algorithm>
#include <cstddef>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {

struct SurfaceMapsComparison {
  std::size_t compared_pixels{};
  std::size_t only_primary{};
  std::size_t only_reference{};
  float max_point_distance{};
  float mean_point_distance{};
  float max_normal_angle{};
  float mean_normal_angle{};
};

struct VolumeComparison {
  std::size_t compared_voxels{};
  std::size_t only_primary{};
  std::size_t only_reference{};
  float max_distance_delta{};
  float mean_distance_delta{};
  float max_weight_delta{};
};

class Comparator {
 public:
  // Throws std::invalid_argument on resolution mismatch.
  [[nodiscard]] static VolumeComparison compare(ConstHostVolumeView primary,
                                                ConstHostVolumeView reference);

  // Throws std::invalid_argument on shape mismatch.
  [[nodiscard]] static SurfaceMapsComparison compare(
      const SurfaceMaps& primary, const SurfaceMaps& reference);

 private:
  struct PairTally {
    std::size_t compared{};
    std::size_t only_primary{};
    std::size_t only_reference{};

    // Classifies one element pair
    // True when both sides are valid and the pair's deviation should be
    // measured.
    [[nodiscard]] bool classify(bool primary_valid, bool reference_valid) {
      if (primary_valid && reference_valid) {
        ++compared;
        return true;
      }
      if (primary_valid) {
        ++only_primary;
      } else if (reference_valid) {
        ++only_reference;
      }
      return false;
    }
  };

  // Running max/mean of one deviation metric
  class Deviation {
   public:
    void add(float value) {
      sum_ += static_cast<double>(value);
      max_ = std::max(max_, value);
    }

    [[nodiscard]] float max() const { return max_; }
    [[nodiscard]] float mean(std::size_t count) const {
      if (count == 0) {
        return 0.0F;
      }
      return static_cast<float>(sum_ / static_cast<double>(count));
    }

   private:
    double sum_{};
    float max_{};
  };

  // A raycast pixel takes part in the comparison only when its point and
  // normal are both finite
  [[nodiscard]] static bool valid_surface_pixel(const Vec3f& point,
                                                const Vec3f& normal) {
    return all_finite(point) && all_finite(normal);
  }
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_COMPARISON_HPP */
