#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP

#include <Eigen/Core>

#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

// Fixed depth-processing constants.
inline constexpr unsigned int kPyramidLevels = 3U;
inline constexpr int kBilateralRadius = 2;
inline constexpr float kBilateralSpatialSigma = 2.0F;
inline constexpr float kBilateralDepthSigma = 0.08F;
inline constexpr float kMaxDownsampleDepthJump = 0.1F;

struct VertexNormalMaps {
  image_proc::Image<Eigen::Vector3f> vertices;
  image_proc::Image<Eigen::Vector3f> normals;
};

struct DepthProcessingLevel {
  image_proc::DepthImage depth_image;
  CameraIntrinsics intrinsics;
  VertexNormalMaps maps;
};

[[nodiscard]] inline CameraIntrinsics scale_intrinsics(
    const CameraIntrinsics& base, unsigned int level) {
  const float scale = 1.0F / static_cast<float>(1U << level);
  return CameraIntrinsics{.fx = base.fx * scale,
                          .fy = base.fy * scale,
                          .cx = base.cx * scale,
                          .cy = base.cy * scale};
}

// Central-difference surface normals. Border pixels and pixels with any
// non-finite neighbour or depth discontinuity are left as NaN so downstream
// code can skip them.
[[nodiscard]] image_proc::Image<Eigen::Vector3f>
compute_normals_central_differences(
    const image_proc::Image<Eigen::Vector3f>& vertices);

// Edge-preserving bilateral filter on raw depth. Samples outside the usable
// depth range (and zeros) are ignored; invalid output pixels stay zero.
[[nodiscard]] image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image);

// Conservative 2x2 reduction: ignore zero depths, reject neighbourhoods with a
// large depth jump, and average the remaining valid raw depths.
[[nodiscard]] image_proc::DepthImage build_depth_pyramid_level(
    const image_proc::DepthImage& depth_image);

// Back-project one depth image into world space (camera_pose = Identity for
// camera-space maps). Invalid pixels are set to quiet NaN.
[[nodiscard]] image_proc::Image<Eigen::Vector3f> project_depth_to_vertices(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose);

// Build the bilateral-filtered depth pyramid with image-aligned vertex/normal
// maps per level for projective ICP.
[[nodiscard]] std::vector<DepthProcessingLevel> build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity());

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP */
