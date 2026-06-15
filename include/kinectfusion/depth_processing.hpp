#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP

#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

enum class NormalComputation {
  paper_forward,
  central_difference,
};

struct DepthProcessingOptions {
  unsigned int levels{3};
  float depth_scale{kTumDepthScale};
  float min_depth{0.4F};
  float max_depth{8.0F};
  float max_normal_depth_jump{0.1F};
  float max_downsample_depth_jump{0.1F};
  bool bilateral_filter{true};
  int bilateral_radius{2};
  float bilateral_spatial_sigma{2.0F};
  float bilateral_depth_sigma{0.08F};
  NormalComputation normal_computation{NormalComputation::paper_forward};
};

struct VertexNormalMaps {
  image_proc::Image<Eigen::Vector3f> vertices;
  image_proc::Image<Eigen::Vector3f> normals;
};

struct DepthProcessingLevel {
  image_proc::DepthImage depth;
  CameraIntrinsics intrinsics;
  VertexNormalMaps maps;
};

using DepthPyramid = std::vector<image_proc::DepthImage>;
using SurfacePyramid = std::vector<DepthProcessingLevel>;

// Apply a CPU bilateral filter to raw depth. Invalid input pixels remain zero.
// Spatial distance is measured in pixels and range distance in meters.
[[nodiscard]] image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options = {});

// Build a depth pyramid where level 0 is the provided depth image and each
// next level is half resolution. Prefer a conservative 2x2 reduction: ignore
// zero depth values, reject mixed neighborhoods with large depth jumps, and
// average valid depths in raw uint16 units so the image remains compatible with
// image_proc::DepthImage.
[[nodiscard]] DepthPyramid build_depth_pyramid(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options = {});

// Return intrinsics scaled for a pyramid level. fx, fy, cx, and cy should be
// multiplied by 1 / 2^level. Keep this helper separate so projective ICP,
// raycasting, and back-projection use the same convention.
[[nodiscard]] CameraIntrinsics scale_intrinsics(const CameraIntrinsics& base,
                                                unsigned int level);

// Back-project one depth image into camera or world space. Use
// camera_to_world = Identity for camera-space vertex maps. Invalid pixels
// should be set to quiet NaN, not zero, so later correspondence code can use
// Eigen::allFinite().
[[nodiscard]] image_proc::Image<Eigen::Vector3f> back_project_depth(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_to_world = Eigen::Matrix4f::Identity(),
    const DepthProcessingOptions& options = {});

// Compute normal maps from an image-aligned vertex map. For each interior
// pixel, use central differences:
//   dx = vertex(x + 1, y) - vertex(x - 1, y)
//   dy = vertex(x, y + 1) - vertex(x, y - 1)
//   normal = normalize(dy.cross(dx))
// Reject borders, NaN neighbors, zero-length normals, and depth discontinuities.
[[nodiscard]] image_proc::Image<Eigen::Vector3f> compute_normals(
    const image_proc::Image<Eigen::Vector3f>& vertices,
    const DepthProcessingOptions& options = {});

// Convenience wrapper for one level. It should back-project depth first, then
// compute normals from the resulting vertex map.
[[nodiscard]] VertexNormalMaps build_vertex_normal_maps(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_to_world = Eigen::Matrix4f::Identity(),
    const DepthProcessingOptions& options = {});

// Full preprocessing entry point for tracking. Build the depth pyramid, scale
// intrinsics per level, then build image-aligned vertex/normal maps per level.
// Projective ICP should consume SurfacePyramid rather than unstructured point
// clouds so pixel-to-pixel correspondences remain available.
[[nodiscard]] SurfacePyramid build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_to_world = Eigen::Matrix4f::Identity(),
    const DepthProcessingOptions& options = {});

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP */
