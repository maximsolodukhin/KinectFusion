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

using Vector3s = Eigen::Matrix<size_t, 3, 1>;

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

  [[nodiscard]] Eigen::Vector2f project(
      const Eigen::Vector3f& camera_point) const {
    return Eigen::Vector2f{
        fx * camera_point.x() / camera_point.z() + cx,
        fy * camera_point.y() / camera_point.z() + cy};
  }

  // Back-projects pixel (x, y) at depth z into a camera-space point.
  [[nodiscard]] Eigen::Vector3f back_project(const Eigen::Vector2f& pixel,
                                             float depth) const {
    return Eigen::Vector3f{(pixel.x() - cx) * depth / fx,
                           (pixel.y() - cy) * depth / fy, depth};
  }

  // Intrinsics for pyramid level `level`, where each level halves the
  // resolution (level 0 returns the unscaled intrinsics).
  [[nodiscard]] CameraIntrinsics scaled(unsigned int level) const {
    const float scale = 1.0F / static_cast<float>(1U << level);
    return CameraIntrinsics{.fx = fx * scale,
                            .fy = fy * scale,
                            .cx = cx * scale,
                            .cy = cy * scale};
  }
};

[[nodiscard]] inline Eigen::Vector3f invalid_vector() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  return Eigen::Vector3f{nan, nan, nan};
}

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
                                           float depth_scale) {
  return static_cast<float>(depth) / depth_scale;
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_UTIL_HPP */
