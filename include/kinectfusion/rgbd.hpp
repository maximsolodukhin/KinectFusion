#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_RGBD_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_RGBD_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>

namespace kinectfusion {

using ColorRgba = Eigen::Vector<std::uint8_t, 4>;

inline constexpr std::uint32_t color_channel_mask =
    0xFFU;  // select lowest 8 bits
inline constexpr std::uint32_t max_color_channel_value = 0xFFU;
inline constexpr float max_color_channel_value_f = 255.0F;
inline constexpr unsigned int color_red_shift = 0U;     // bits 0 - 7
inline constexpr unsigned int color_green_shift = 8U;   // bits 8 - 15
inline constexpr unsigned int color_blue_shift = 16U;   // bits 16 - 23
inline constexpr unsigned int color_alpha_shift = 24U;  // bits 24 - 31

struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};

  [[nodiscard]] Eigen::Matrix3f matrix() const {
    Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
    intrinsics << fx, 0.0F, cx, 0.0F, fy, cy, 0.0F, 0.0F, 1.0F;
    return intrinsics;
  }

  [[nodiscard]] Eigen::Vector2f project(
      const Eigen::Vector3f& camera_point) const {
    return Eigen::Vector2f{(fx * camera_point.x() / camera_point.z()) + cx,
                           (fy * camera_point.y() / camera_point.z()) + cy};
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
    return CameraIntrinsics{
        .fx = fx * scale, .fy = fy * scale, .cx = cx * scale, .cy = cy * scale};
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
  return ColorRgba{static_cast<std::uint8_t>((pixel >> color_red_shift) &
                                             color_channel_mask),
                   static_cast<std::uint8_t>((pixel >> color_green_shift) &
                                             color_channel_mask),
                   static_cast<std::uint8_t>((pixel >> color_blue_shift) &
                                             color_channel_mask),
                   static_cast<std::uint8_t>((pixel >> color_alpha_shift) &
                                             color_channel_mask)};
}

[[nodiscard]] inline float depth_to_meters(std::uint16_t depth,
                                           float depth_scale) {
  return static_cast<float>(depth) / depth_scale;
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_RGBD_HPP */
