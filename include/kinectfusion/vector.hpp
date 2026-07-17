#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VECTOR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VECTOR_HPP

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <limits>
#include <type_traits>

namespace kinectfusion {

enum class MemorySpace : std::uint8_t { kHost, kDevice };

struct alignas(2 * sizeof(float)) Vec2f {
  float x{};
  float y{};
};

static_assert(sizeof(Vec2f) == 2 * sizeof(float));

struct Vec3f {
  float x{};
  float y{};
  float z{};

  friend bool operator==(const Vec3f&, const Vec3f&) = default;
};

struct Vec3i {
  int x{};
  int y{};
  int z{};
};

struct Size3 {
  std::size_t x{};
  std::size_t y{};
  std::size_t z{};

  friend bool operator==(const Size3&, const Size3&) = default;
};

using Vector3s = Eigen::Matrix<std::size_t, 3, 1>;

[[nodiscard]] KINECTFUSION_FORCEINLINE Size3 to_size3(const Vector3s& value) {
  return Size3{.x = value.x(), .y = value.y(), .z = value.z()};
}

// To prevent casts everywhere
template <typename X, typename Y, typename Z>
  requires std::is_arithmetic_v<X> && std::is_arithmetic_v<Y> &&
           std::is_arithmetic_v<Z>
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f make_vec3f(X x,
                                                                         Y y,
                                                                         Z z) {
  return Vec3f{.x = static_cast<float>(x),
               .y = static_cast<float>(y),
               .z = static_cast<float>(z)};
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f zero_vec3f() {
  return {};
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE Vec3f invalid_vec3f() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  return make_vec3f(nan, nan, nan);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator+(
    const Vec3f lhs, const Vec3f rhs) {
  return make_vec3f(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator-(
    const Vec3f lhs, const Vec3f rhs) {
  return make_vec3f(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator-(
    const Vec3f value) {
  return make_vec3f(-value.x, -value.y, -value.z);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator*(
    const Vec3f value, const float scale) {
  return make_vec3f(value.x * scale, value.y * scale, value.z * scale);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator*(
    const float scale, const Vec3f value) {
  return value * scale;
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator/(
    const Vec3f value, const float scale) {
  return make_vec3f(value.x / scale, value.y / scale, value.z / scale);
}

KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f& operator+=(Vec3f& lhs,
                                                            const Vec3f rhs) {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr float dot(
    const Vec3f lhs, const Vec3f rhs) {
  return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f cross(
    const Vec3f lhs, const Vec3f rhs) {
  return make_vec3f((lhs.y * rhs.z) - (lhs.z * rhs.y),
                    (lhs.z * rhs.x) - (lhs.x * rhs.z),
                    (lhs.x * rhs.y) - (lhs.y * rhs.x));
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr float squared_norm(
    const Vec3f value) {
  return dot(value, value);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float norm(const Vec3f value) {
  return std::sqrt(squared_norm(value));
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE Vec3f
normalized(const Vec3f value) {
  const float length = norm(value);
  return length > 0.0F ? value / length : zero_vec3f();
}

// Angle between two directions in radians, in [0, pi]. atan2 over
// (|cross|, dot) instead of acos(dot): acos loses ~sqrt(epsilon) precision
// near parallel vectors, which would put a false floor under small-deviation
// statistics; this form is exact at 0 and stable everywhere.
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float angle_between(
    const Vec3f lhs, const Vec3f rhs) {
  const Vec3f source = normalized(lhs);
  const Vec3f target = normalized(rhs);
  return std::atan2(norm(cross(source, target)), dot(source, target));
}

// Row-major 3x3 matrix; the POD counterpart of an Eigen::Matrix3f rotation
// block for kernel-bound math.
struct Mat3f {
  Vec3f row_x{};
  Vec3f row_y{};
  Vec3f row_z{};
};

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator*(
    const Mat3f& matrix, const Vec3f& vector) {
  return make_vec3f(dot(matrix.row_x, vector), dot(matrix.row_y, vector),
                    dot(matrix.row_z, vector));
}

// Rigid body transform
struct RigidTransform {
  Mat3f rotation{};
  Vec3f translation{};

  [[nodiscard]] static constexpr RigidTransform identity() {
    return {.rotation = {.row_x = make_vec3f(1, 0, 0),
                         .row_y = make_vec3f(0, 1, 0),
                         .row_z = make_vec3f(0, 0, 1)},
            .translation = {}};
  }
};

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f operator*(
    const RigidTransform& transform, const Vec3f& point) {
  return (transform.rotation * point) + transform.translation;
}

// Weighted average of two samples; the caller guarantees a positive total
// weight.
template <typename ValueT>
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr ValueT weighted_average(
    const ValueT& value, float value_weight, const ValueT& observed,
    float observed_weight) {
  return ((value * value_weight) + (observed * observed_weight)) /
         (value_weight + observed_weight);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE bool all_finite(
    const Vec3f value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] KINECTFUSION_FORCEINLINE Vec3f
from_eigen(const Eigen::Vector3f& value) {
  return make_vec3f(value.x(), value.y(), value.z());
}

[[nodiscard]] KINECTFUSION_FORCEINLINE Mat3f
from_eigen(const Eigen::Matrix3f& value) {
  return Mat3f{.row_x = make_vec3f(value(0, 0), value(0, 1), value(0, 2)),
               .row_y = make_vec3f(value(1, 0), value(1, 1), value(1, 2)),
               .row_z = make_vec3f(value(2, 0), value(2, 1), value(2, 2))};
}

[[nodiscard]] KINECTFUSION_FORCEINLINE RigidTransform
from_eigen(const Eigen::Matrix4f& transform) {
  return {
      .rotation = from_eigen(Eigen::Matrix3f{transform.block<3, 3>(0, 0)}),
      .translation = from_eigen(Eigen::Vector3f{transform.block<3, 1>(0, 3)})};
}

[[nodiscard]] KINECTFUSION_FORCEINLINE Eigen::Vector3f to_eigen(
    const Vec3f value) {
  return Eigen::Vector3f{value.x, value.y, value.z};
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VECTOR_HPP */
