#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <kinectfusion/image_proc/image.hpp>

namespace kinectfusion {

// Fixed scene/sensor constants shared across the library.
inline constexpr float kTumDepthScale = 5000.0F;
inline constexpr float kMinDepth = 0.4F;
inline constexpr float kMaxDepth = 8.0F;

using ColorRgba = Eigen::Vector<std::uint8_t, 4>;

struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};

  [[nodiscard]] Eigen::Matrix3f matrix() const {
    Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
    intrinsics <<   fx, 0.0F,   cx, 
                  0.0F,   fy,   cy, 
                  0.0F, 0.0F, 1.0F;
    return intrinsics;
  }
};

[[nodiscard]] inline Eigen::Matrix4f make_transform_matrix(
    const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = translation;
  return transform;
}

// Unpacks an RGBA8 pixel (as stored by image_proc::ColorImage) into bytes.
[[nodiscard]] inline ColorRgba rgba_from_pixel(std::uint32_t pixel) {
  return ColorRgba{static_cast<std::uint8_t>(pixel & 0xFFU),
                   static_cast<std::uint8_t>((pixel >> 8U) & 0xFFU),
                   static_cast<std::uint8_t>((pixel >> 16U) & 0xFFU),
                   static_cast<std::uint8_t>((pixel >> 24U) & 0xFFU)};
}

[[nodiscard]] inline float depth_to_meters(std::uint16_t depth,
                                           float depth_scale = kTumDepthScale) {
  return static_cast<float>(depth) / depth_scale;
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP */
