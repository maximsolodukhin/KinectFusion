#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_RGBD_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_RGBD_HPP

#include <Eigen/Core>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/vector.hpp>

namespace kinectfusion {

using ColorRgba = Eigen::Vector<std::uint8_t, 4>;

// Default TUM RGB-D sensor conventions. Single source of truth shared by
// depth processing, TSDF fusion, and the app options.
inline constexpr float kDefaultTumDepthScale = 5000.0F;
inline constexpr float kDefaultMinDepthMeters = 0.4F;
inline constexpr float kDefaultMaxDepthMeters = 8.0F;

inline constexpr std::uint32_t kColorChannelMask =
    0xFFU;  // select lowest 8 bits
inline constexpr std::uint32_t kMaxColorChannelValue = 0xFFU;
inline constexpr float kMaxColorChannelValueF = 255.0F;
inline constexpr unsigned int kColorRedShift = 0U;     // bits 0 - 7
inline constexpr unsigned int kColorGreenShift = 8U;   // bits 8 - 15
inline constexpr unsigned int kColorBlueShift = 16U;   // bits 16 - 23
inline constexpr unsigned int kColorAlphaShift = 24U;  // bits 24 - 31

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

  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec2f
  project(const Vec3f& camera_point) const {
    return Vec2f{.x = (fx * camera_point.x / camera_point.z) + cx,
                 .y = (fy * camera_point.y / camera_point.z) + cy};
  }

  [[nodiscard]] Eigen::Vector2f project(
      const Eigen::Vector3f& camera_point) const {
    const Vec2f pixel = project(from_eigen(camera_point));
    return Eigen::Vector2f{pixel.x, pixel.y};
  }

  // Back-projects pixel (x, y) at depth z into a camera-space point.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE Vec3f back_project(const Vec2f& pixel,
                                                            float depth) const {
    return make_vec3f((pixel.x - cx) * depth / fx, (pixel.y - cy) * depth / fy,
                      depth);
  }

  [[nodiscard]] Eigen::Vector3f back_project(const Eigen::Vector2f& pixel,
                                             float depth) const {
    return to_eigen(back_project(Vec2f{.x = pixel.x(), .y = pixel.y()}, depth));
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

// Packs a colour with 0..255 float channels into the RGBA8 pixel layout used
// by image_proc::ColorImage (alpha fully opaque). Inverse of
// `rgba_from_pixel` up to channel clamping.
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE std::uint32_t pixel_from_color(
    const Vec3f& color) {
  const auto to_byte = [](float value) {
    return static_cast<std::uint32_t>(
        compat::clamp(value, 0.0F, kMaxColorChannelValueF));
  };
  return to_byte(color.x) | (to_byte(color.y) << kColorGreenShift) |
         (to_byte(color.z) << kColorBlueShift) |
         (std::uint32_t{kMaxColorChannelValue} << kColorAlphaShift);
}

// RGB channels of an RGBA8 pixel as 0..255 floats;
// `rgba_from_pixel` but for kernel-bound math.
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr Vec3f color_from_pixel(
    std::uint32_t pixel) {
  return make_vec3f((pixel >> kColorRedShift) & kColorChannelMask,
                    (pixel >> kColorGreenShift) & kColorChannelMask,
                    (pixel >> kColorBlueShift) & kColorChannelMask);
}

// Unpacks an RGBA8 pixel (as stored by image_proc::ColorImage) into bytes.
[[nodiscard]] inline ColorRgba rgba_from_pixel(std::uint32_t pixel) {
  return ColorRgba{
      static_cast<std::uint8_t>((pixel >> kColorRedShift) & kColorChannelMask),
      static_cast<std::uint8_t>((pixel >> kColorGreenShift) &
                                kColorChannelMask),
      static_cast<std::uint8_t>((pixel >> kColorBlueShift) & kColorChannelMask),
      static_cast<std::uint8_t>((pixel >> kColorAlphaShift) &
                                kColorChannelMask)};
}

[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE constexpr float depth_to_meters(
    std::uint16_t depth, float depth_scale) {
  return static_cast<float>(depth) / depth_scale;
}

// Converts a raw sensor sample to meters like `depth_to_meters`, but yields
// nullopt for missing samples (zero) and depths outside [min_depth, max_depth].
[[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE compat::optional<float>
depth_in_range(std::uint16_t depth, float depth_scale, float min_depth,
               float max_depth) {
  if (depth == 0) {
    return compat::nullopt;
  }
  const float meters = depth_to_meters(depth, depth_scale);
  if (meters < min_depth || meters > max_depth) {
    return compat::nullopt;
  }
  return meters;
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_RGBD_HPP */
