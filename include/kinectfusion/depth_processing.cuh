#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUDA_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUDA_HPP

#include <Eigen/Core>

#include <vector>

#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion::cuda {

[[nodiscard]] image_proc::Vector3fImage
compute_normals_central_differences(
    const image_proc::Vector3fImage& vertices,
    const DepthProcessingOptions& options = {});

[[nodiscard]] image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options = {});

[[nodiscard]] image_proc::DepthImage build_depth_pyramid_level(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options = {});

[[nodiscard]] image_proc::Vector3fImage project_depth_to_vertices(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options = {});

[[nodiscard]] std::vector<DepthProcessingLevel> build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics,
    const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity(),
    const DepthProcessingOptions& options = {});

}  // namespace kinectfusion::cuda

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_CUDA_HPP */
