#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP

#include <Eigen/Core>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <utility>
#include <vector>

namespace kinectfusion {

inline constexpr unsigned int default_depth_pyramid_levels = 3U;
inline constexpr float default_depth_processing_depth_scale = 5000.0F;
inline constexpr float default_depth_processing_min_depth = 0.4F;
inline constexpr float default_depth_processing_max_depth = 8.0F;
inline constexpr float default_max_normal_depth_jump_meters = 0.1F;
inline constexpr float default_max_downsample_depth_jump_meters = 0.1F;
inline constexpr int default_bilateral_radius_pixels = 2;
inline constexpr float default_bilateral_spatial_sigma_pixels = 2.0F;
inline constexpr float default_bilateral_depth_sigma_meters = 0.08F;

struct DepthProcessingOptions {
  unsigned int levels{default_depth_pyramid_levels};
  float depth_scale{default_depth_processing_depth_scale};
  float min_depth{default_depth_processing_min_depth};
  float max_depth{default_depth_processing_max_depth};
  float max_normal_depth_jump{default_max_normal_depth_jump_meters};
  float max_downsample_depth_jump{default_max_downsample_depth_jump_meters};
  bool bilateral_filter{true};
  int bilateral_radius{default_bilateral_radius_pixels};
  float bilateral_spatial_sigma{default_bilateral_spatial_sigma_pixels};
  float bilateral_depth_sigma{default_bilateral_depth_sigma_meters};
};

struct VertexNormalMaps {
  image_proc::Vector3fImage vertices;
  image_proc::Vector3fImage normals;
};

template <MemorySpace Space = MemorySpace::Host>
struct VertexNormalMapsView {
  image_proc::ImageView<Vec3f, Space> vertices;
  image_proc::ImageView<Vec3f, Space> normals;

  static constexpr MemorySpace memory_space = Space;
};

template <MemorySpace Space = MemorySpace::Host>
struct ConstVertexNormalMapsView {
  image_proc::ImageView<const Vec3f, Space> vertices;
  image_proc::ImageView<const Vec3f, Space> normals;

  static constexpr MemorySpace memory_space = Space;
};

using HostVertexNormalMapsView = VertexNormalMapsView<MemorySpace::Host>;
using DeviceVertexNormalMapsView = VertexNormalMapsView<MemorySpace::Device>;
using ConstHostVertexNormalMapsView =
    ConstVertexNormalMapsView<MemorySpace::Host>;
using ConstDeviceVertexNormalMapsView =
    ConstVertexNormalMapsView<MemorySpace::Device>;

[[nodiscard]] inline HostVertexNormalMapsView view(VertexNormalMaps& maps) {
  return HostVertexNormalMapsView{.vertices = maps.vertices.view(),
                                  .normals = maps.normals.view()};
}

[[nodiscard]] inline ConstHostVertexNormalMapsView view(
    const VertexNormalMaps& maps) {
  return ConstHostVertexNormalMapsView{.vertices = maps.vertices.view(),
                                       .normals = maps.normals.view()};
}

struct DepthProcessingLevel {
  DepthProcessingLevel(image_proc::DepthImage depth,
                       const CameraIntrinsics& camera_intrinsics,
                       image_proc::Vector3fImage vertices,
                       image_proc::Vector3fImage normals)
      : depth_image(std::move(depth)),
        intrinsics(camera_intrinsics),
        maps{.vertices = std::move(vertices), .normals = std::move(normals)} {}

  image_proc::DepthImage depth_image;
  CameraIntrinsics intrinsics;
  VertexNormalMaps maps;
};

// Central-difference surface normals. Border pixels and pixels with any
// non-finite neighbour or depth discontinuity are left as NaN so downstream
// code can skip them.
[[nodiscard]] image_proc::Vector3fImage compute_normals_central_differences(
    const image_proc::Vector3fImage& vertices,
    const DepthProcessingOptions& options = {});

// Edge-preserving bilateral filter on raw depth. Samples outside the usable
// depth range (and zeros) are ignored; invalid output pixels stay zero.
[[nodiscard]] image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options = {});

// Conservative 2x2 reduction: ignore zero depths, reject neighbourhoods with a
// large depth jump, and average the remaining valid raw depths.
[[nodiscard]] image_proc::DepthImage build_depth_pyramid_level(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options = {});

// Back-project one depth image into world space (camera_pose = Identity for
// camera-space maps). Invalid pixels are set to quiet NaN.
[[nodiscard]] image_proc::Vector3fImage project_depth_to_vertices(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options = {});

// Build the bilateral-filtered depth pyramid with image-aligned vertex/normal
// maps per level for projective ICP.
[[nodiscard]] std::vector<DepthProcessingLevel> build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity(),
    const DepthProcessingOptions& options = {});

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP */
