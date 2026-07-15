#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_COMPAT_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_COMPAT_HPP

#include <type_traits>

#ifdef __CUDACC__
#include <cuda/std/cmath>
#include <cuda/std/optional>
#else
#include <cmath>
#include <optional>
#endif

#ifdef __CUDACC__
#define KINECTFUSION_HOST_DEVICE __host__ __device__
#else
#define KINECTFUSION_HOST_DEVICE
#endif

// Portable spelling of C99 `restrict`. Use on local pointers and function
// parameters only (not on struct members). Examples:
//
// Local pointer inside a kernel:
//
//   __global__ void copy_kernel(image_proc::DeviceImageView<const float> in,
//                               image_proc::DeviceImageView<float>       out) {
//     const float* KINECTFUSION_RESTRICT src = in.data;
//     float*       KINECTFUSION_RESTRICT dst = out.data;
//     const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
//     dst[i] = src[i];
//   }
//
// Function parameter:
//
//   void axpy(std::size_t n, float alpha,
//             const float* KINECTFUSION_RESTRICT x,
//             float*       KINECTFUSION_RESTRICT y) {
//     for (std::size_t i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
//   }
#if defined(_MSC_VER) && !defined(__clang__)
#define KINECTFUSION_RESTRICT __restrict
#else
#define KINECTFUSION_RESTRICT __restrict__
#endif

#ifdef __CUDACC__
#define KINECTFUSION_FORCEINLINE __forceinline__
#elif defined(_MSC_VER) && !defined(__clang__)
#define KINECTFUSION_FORCEINLINE __forceinline
#else
#define KINECTFUSION_FORCEINLINE inline __attribute__((always_inline))
#endif

#define KINECTFUSION_FORCEINLINE_DEVICE \
  KINECTFUSION_FORCEINLINE KINECTFUSION_HOST_DEVICE

// One vocabulary for the std pieces kernel-reachable code needs: libcu++
// replaces <optional> and lround, and lacks min/max/clamp entirely.
namespace kinectfusion::compat {

// ::cuda, not cuda: inside kinectfusion::compat a bare `cuda` finds the
// sibling namespace kinectfusion::cuda instead of libcu++.
#ifdef __CUDACC__
template <typename T>
using optional = ::cuda::std::optional<T>;
using ::cuda::std::lround;
using ::cuda::std::nullopt;
#else
template <typename T>
using optional = std::optional<T>;
using std::lround;
using std::nullopt;
#endif

template <typename T>
  requires std::is_arithmetic_v<T>
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr T min(T lhs, T rhs) {
  return rhs < lhs ? rhs : lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr T max(T lhs, T rhs) {
  return lhs < rhs ? rhs : lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr T clamp(T value, T low,
                                                                T high) {
  if (value < low) {
    return low;
  }
  if (high < value) {
    return high;
  }
  return value;
}

}  // namespace kinectfusion::compat

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_COMPAT_HPP */
