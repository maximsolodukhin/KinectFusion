#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <limits>

namespace kinectfusion {

using ColorRgba = Eigen::Vector<std::uint8_t, 4>;

inline constexpr float kTumDepthScale = 5000.0F;

struct CameraIntrinsics {
  float fx{525.0F};
  float fy{525.0F};
  float cx{319.5F};
  float cy{239.5F};

  [[nodiscard]] Eigen::Matrix3f matrix() const {
    Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
    intrinsics(0, 0) = fx;
    intrinsics(1, 1) = fy;
    intrinsics(0, 2) = cx;
    intrinsics(1, 2) = cy;
    return intrinsics;
  }
};

[[nodiscard]] inline Eigen::Vector3f invalid_vector() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  return Eigen::Vector3f{nan, nan, nan};
}

[[nodiscard]] inline bool is_valid_depth(std::uint16_t depth) {
  return depth != 0;
}

[[nodiscard]] inline float depth_to_meters(std::uint16_t depth,
                                           float depth_scale = kTumDepthScale) {
  return static_cast<float>(depth) / depth_scale;
}

[[nodiscard]] inline ColorRgba rgba_from_pixel(std::uint32_t pixel) {
  return ColorRgba{static_cast<std::uint8_t>(pixel & 0xFFU),
                   static_cast<std::uint8_t>((pixel >> 8U) & 0xFFU),
                   static_cast<std::uint8_t>((pixel >> 16U) & 0xFFU),
                   static_cast<std::uint8_t>((pixel >> 24U) & 0xFFU)};
}

[[nodiscard]] inline Eigen::Matrix4f make_transform_matrix(
    const Eigen::Matrix3f& rotation,
    const Eigen::Vector3f& translation) {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = translation;
  return transform;
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP */
