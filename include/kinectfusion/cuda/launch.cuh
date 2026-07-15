#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_LAUNCH_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_LAUNCH_CUH

#include <cstddef>

namespace kinectfusion::cuda {

// Grid dimension covering `count` elements in blocks of `divisor`.
[[nodiscard]] constexpr unsigned int ceil_div(std::size_t count,
                                              unsigned int divisor) {
  return static_cast<unsigned int>((count + divisor - 1) / divisor);
}

}  // namespace kinectfusion::cuda

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_LAUNCH_CUH */
