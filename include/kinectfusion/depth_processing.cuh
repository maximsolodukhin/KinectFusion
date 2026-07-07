#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUH

#include <Eigen/Core>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <utility>
#include <vector>
using namespace kinectfusion;

namespace kinectfusion_cuda {

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

}  // namespace kinectfusion_cuda

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUH */
