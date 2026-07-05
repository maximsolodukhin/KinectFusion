#ifndef KINECTFUSION_SRC_KINECTFUSION_DEPTH_PROCESSING_COMMON_HPP
#define KINECTFUSION_SRC_KINECTFUSION_DEPTH_PROCESSING_COMMON_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/util.hpp>

#if defined(__CUDACC__)
#define KINECTFUSION_DEPTH_HOST_DEVICE __host__ __device__
#define KINECTFUSION_DEPTH_FORCE_INLINE __forceinline__
#else
#define KINECTFUSION_DEPTH_HOST_DEVICE
#define KINECTFUSION_DEPTH_FORCE_INLINE inline
#endif

namespace kinectfusion::depth_processing_detail {

inline constexpr std::size_t downsample_factor = 2U;

inline void validate_options(const DepthProcessingOptions& options) {
  if (options.levels == 0U) {
    throw std::invalid_argument("Depth pyramid must have at least one level");
  }
  if (options.depth_scale <= 0.0F) {
    throw std::invalid_argument("Depth scale must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Depth range is invalid");
  }
  if (options.max_normal_depth_jump < 0.0F) {
    throw std::invalid_argument(
        "Normal depth jump threshold must be non-negative");
  }
  if (options.max_downsample_depth_jump < 0.0F) {
    throw std::invalid_argument(
        "Downsample depth jump threshold must be non-negative");
  }
  if (options.bilateral_radius < 0) {
    throw std::invalid_argument("Bilateral filter radius must be non-negative");
  }
  if (options.bilateral_spatial_sigma <= 0.0F) {
    throw std::invalid_argument("Bilateral spatial sigma must be positive");
  }
  if (options.bilateral_depth_sigma <= 0.0F) {
    throw std::invalid_argument("Bilateral depth sigma must be positive");
  }
}

[[nodiscard]] inline float bilateral_weight_scale(float sigma) {
  return -0.5F / (sigma * sigma);
}

template <typename Options>
[[nodiscard]] KINECTFUSION_DEPTH_HOST_DEVICE
    KINECTFUSION_DEPTH_FORCE_INLINE float
    usable_depth_meters(std::uint16_t raw, const Options& options) {
  if (raw == 0) {
    return 0.0F;
  }
  const float meters = depth_to_meters(raw, options.depth_scale);
  return meters >= options.min_depth && meters <= options.max_depth ? meters
                                                                    : 0.0F;
}

}  // namespace kinectfusion::depth_processing_detail

#undef KINECTFUSION_DEPTH_FORCE_INLINE
#undef KINECTFUSION_DEPTH_HOST_DEVICE

#endif  // KINECTFUSION_SRC_KINECTFUSION_DEPTH_PROCESSING_COMMON_HPP
