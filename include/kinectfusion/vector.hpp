#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VECTOR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VECTOR_HPP

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
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

namespace kinectfusion {

enum class MemorySpace : std::uint8_t { Host, Device };

struct Vec3f {
  float x{};
  float y{};
  float z{};
};

struct Size3 {
  std::size_t x{};
  std::size_t y{};
  std::size_t z{};
};

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
make_vec3f(float x, float y, float z) {
  return Vec3f{.x = x, .y = y, .z = z};
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f zero_vec3f() {
  return {};
}

[[nodiscard]] inline Vec3f invalid_vec3f() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  return make_vec3f(nan, nan, nan);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
operator+(const Vec3f lhs, const Vec3f rhs) {
  return make_vec3f(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
operator-(const Vec3f lhs, const Vec3f rhs) {
  return make_vec3f(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
operator-(const Vec3f value) {
  return make_vec3f(-value.x, -value.y, -value.z);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
operator*(const Vec3f value, const float scale) {
  return make_vec3f(value.x * scale, value.y * scale, value.z * scale);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
operator*(const float scale, const Vec3f value) {
  return value * scale;
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f
operator/(const Vec3f value, const float scale) {
  return make_vec3f(value.x / scale, value.y / scale, value.z / scale);
}

KINECTFUSION_HOST_DEVICE constexpr Vec3f &operator+=(Vec3f &lhs,
                                                     const Vec3f rhs) {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr float dot(const Vec3f lhs,
                                                           const Vec3f rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr Vec3f cross(const Vec3f lhs,
                                                             const Vec3f rhs) {
  return make_vec3f(lhs.y * rhs.z - lhs.z * rhs.y,
                    lhs.z * rhs.x - lhs.x * rhs.z,
                    lhs.x * rhs.y - lhs.y * rhs.x);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE constexpr float
squared_norm(const Vec3f value) {
  return dot(value, value);
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE inline float norm(const Vec3f value) {
  return std::sqrt(squared_norm(value));
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE inline Vec3f
normalized(const Vec3f value) {
  const float length = norm(value);
  return length > 0.0F ? value / length : zero_vec3f();
}

[[nodiscard]] KINECTFUSION_HOST_DEVICE inline bool
all_finite(const Vec3f value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] inline Vec3f from_eigen(const Eigen::Vector3f &value) {
  return make_vec3f(value.x(), value.y(), value.z());
}

[[nodiscard]] inline Eigen::Vector3f to_eigen(const Vec3f value) {
  return Eigen::Vector3f{value.x, value.y, value.z};
}

} // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VECTOR_HPP */
