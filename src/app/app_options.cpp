#include "app_options.hpp"

#include <stdexcept>
#include <string>

namespace app {

kinectfusion::Vec3f AppOptions::volume_origin() const {
  const auto resolution = static_cast<float>(volume_resolution);
  const float half_extent = 0.5F * resolution * voxel_size;
  return kinectfusion::Vec3f{
      .x = -half_extent, .y = -half_extent, .z = -volume_camera_margin};
}

kinectfusion::RaycastOptions AppOptions::raycast_options(
    const kinectfusion::VirtualSensor& sensor,
    const Eigen::Matrix4f& camera_to_world, unsigned int level) const {
  const auto scale = 1U << level;
  return kinectfusion::RaycastOptions{
      .intrinsics = sensor.depth_intrinsics().scaled(level),
      .width = sensor.depth_image().width() / scale,
      .height = sensor.depth_image().height() / scale,
      .camera_to_world = camera_to_world,
      .min_depth = min_depth,
      .max_depth = max_depth,
      .tsdf_from_valid_corners = raycast_tsdf_from_valid_corners};
}

kinectfusion::DepthProcessingOptions AppOptions::depth_options() const {
  return kinectfusion::DepthProcessingOptions{
      .levels = pyramid_levels,
      .depth_scale = depth_scale,
      .min_depth = min_depth,
      .max_depth = max_depth,
      .bilateral_filter = bilateral_filter,
      .bilateral_radius = bilateral_radius,
      .bilateral_spatial_sigma = bilateral_spatial_sigma,
      .bilateral_depth_sigma = bilateral_depth_sigma};
}

kinectfusion::TsdfIntegrationOptions AppOptions::tsdf_options() const {
  return kinectfusion::TsdfIntegrationOptions{
      .depth_scale = depth_scale,
      .min_depth = min_depth,
      .max_depth = max_depth,
      .projective_distance = projective_tsdf_distance,
      .distance_scaled_truncation = distance_scaled_truncation,
      .truncation_distance_scale = truncation_distance_scale,
      .view_angle_weighting = view_angle_weighting};
}

kinectfusion::ProjectiveIcpOptions AppOptions::icp_options() const {
  // Fields left unset keep ProjectiveIcpOptions' defaults (these matched the
  // values the call site used to set explicitly).
  return kinectfusion::ProjectiveIcpOptions{
      .max_point_distance = matching_distance,
      .min_normal_dot = min_normal_dot,
      .max_update_translation = max_pose_update_translation,
      .max_update_rotation = max_pose_update_rotation,
      .min_system_eigenvalue = min_icp_eigenvalue,
      .max_condition_number = max_icp_condition_number};
}

unsigned int AppOptions::icp_iterations_for_level(unsigned int level) const {
  if (icp_iterations > 0) {
    return static_cast<unsigned int>(icp_iterations);
  }
  if (level == 0) {
    return 10U;
  }
  if (level == 1) {
    return 5U;
  }
  return 4U;
}

void configure_cli(CLI::App& app, AppOptions& app_options) {
  app.set_version_flag("--version", std::string{KINECTFUSION_VERSION});
  app.add_option("dataset", app_options.dataset_dir,
                 "TUM RGB-D dataset directory");
  app.add_option("--frames", app_options.max_frames,
                 "Maximum number of frames to process. Use -1 for all.");
  app.add_option("--volume-resolution", app_options.volume_resolution,
                 "Cubic TSDF volume resolution in voxels.");
  app.add_option("--voxel-size", app_options.voxel_size,
                 "Voxel size in meters.");
  app.add_option("--truncation-distance", app_options.truncation_distance,
                 "TSDF truncation distance in meters.");
  app.add_option("--volume-camera-margin", app_options.volume_camera_margin,
                 "Extra space behind the initial camera in meters.");
  app.add_option("--depth-scale", app_options.depth_scale,
                 "Raw depth units per meter (TUM default 5000).");
  app.add_option("--min-depth", app_options.min_depth,
                 "Minimum usable depth in meters.");
  app.add_option("--max-depth", app_options.max_depth,
                 "Maximum usable depth in meters.");
  app.add_option("--pyramid-levels", app_options.pyramid_levels,
                 "Number of depth pyramid levels to build.");
  app.add_option("--icp-iterations", app_options.icp_iterations,
                 "Fixed ICP iterations per level, or -1 for [10, 5, 4].");
  app.add_option("--matching-distance", app_options.matching_distance,
                 "Maximum ICP correspondence distance in meters.");
  app.add_option("--min-normal-dot", app_options.min_normal_dot,
                 "Minimum ICP normal dot product. cos(15deg)=0.9659258.");
  app.add_option("--max-pose-update-translation",
                 app_options.max_pose_update_translation,
                 "Maximum accepted ICP translation update in meters.");
  app.add_option("--max-pose-update-rotation",
                 app_options.max_pose_update_rotation,
                 "Maximum accepted ICP rotation update in radians.");
  app.add_option("--min-icp-eigenvalue", app_options.min_icp_eigenvalue,
                 "Minimum accepted ICP normal-system eigenvalue.");
  app.add_option("--max-icp-condition-number",
                 app_options.max_icp_condition_number,
                 "Maximum accepted ICP normal-system condition number.");
  app.add_flag("--projective-tsdf-distance,!--no-projective-tsdf-distance",
               app_options.projective_tsdf_distance,
               "Use lambda-corrected projective TSDF distance.");
  app.add_flag("--distance-scaled-truncation,!--no-distance-scaled-truncation",
               app_options.distance_scaled_truncation,
               "Increase TSDF truncation support linearly with depth.");
  app.add_option("--truncation-distance-scale",
                 app_options.truncation_distance_scale,
                 "Linear TSDF truncation support increase per meter.");
  app.add_flag("--view-angle-weighting,!--no-view-angle-weighting",
               app_options.view_angle_weighting,
               "Weight TSDF observations by view angle and depth.");
  app.add_flag("--bilateral-filter,!--no-bilateral-filter",
               app_options.bilateral_filter,
               "Enable bilateral depth filtering for tracking.");
  app.add_option("--bilateral-radius", app_options.bilateral_radius,
                 "Bilateral filter radius in pixels.");
  app.add_option("--bilateral-spatial-sigma",
                 app_options.bilateral_spatial_sigma,
                 "Bilateral filter spatial sigma in pixels.");
  app.add_option("--bilateral-depth-sigma", app_options.bilateral_depth_sigma,
                 "Bilateral filter depth sigma in meters.");
  app.add_flag("--interactive-relocalization",
               app_options.interactive_relocalization,
               "Pause after tracking failure before continuing.");
  app.add_option("--output-dir", app_options.output_dir,
                 "Directory for raycast images and point clouds.");
  app.add_flag("--write-raycast-images,!--no-write-raycast-images",
               app_options.write_raycast_images,
               "Write raycast color images as PNG files.");
  app.add_flag("--write-point-clouds,!--no-write-point-clouds",
               app_options.write_point_clouds,
               "Write raycast point clouds as ASCII PLY files.");
  app.add_flag(
      "--raycast-tsdf-from-valid-corners,!--no-raycast-tsdf-from-valid-corners",
      app_options.raycast_tsdf_from_valid_corners,
      "Interpolate raycast TSDF only from fully valid voxel corners.");
}

void validate_options(const AppOptions& app_options) {
  if (app_options.volume_resolution <= 0) {
    throw std::invalid_argument("volume resolution must be positive");
  }
  const float extent = static_cast<float>(app_options.volume_resolution) *
                       app_options.voxel_size;
  if (app_options.volume_camera_margin < 0.0F ||
      app_options.volume_camera_margin > extent) {
    throw std::invalid_argument(
        "volume camera margin must be within [0, volume extent]");
  }
  if (app_options.voxel_size <= 0.0F) {
    throw std::invalid_argument("voxel size must be positive");
  }
  if (app_options.truncation_distance <= 0.0F) {
    throw std::invalid_argument("truncation distance must be positive");
  }
  if (app_options.depth_scale <= 0.0F) {
    throw std::invalid_argument("depth scale must be positive");
  }
  if (app_options.min_depth < 0.0F ||
      app_options.max_depth <= app_options.min_depth) {
    throw std::invalid_argument("depth range is invalid");
  }
  if (app_options.pyramid_levels == 0U) {
    throw std::invalid_argument("pyramid levels must be positive");
  }
  if (app_options.icp_iterations == 0 || app_options.icp_iterations < -1) {
    throw std::invalid_argument("ICP iterations must be positive or -1");
  }
  if (app_options.matching_distance <= 0.0F) {
    throw std::invalid_argument("matching distance must be positive");
  }
  if (app_options.min_normal_dot < -1.0F || app_options.min_normal_dot > 1.0F) {
    throw std::invalid_argument("min normal dot must be within [-1, 1]");
  }
  if (app_options.truncation_distance_scale < 0.0F) {
    throw std::invalid_argument(
        "truncation distance scale must be non-negative");
  }
  if (app_options.bilateral_radius < 0) {
    throw std::invalid_argument("bilateral radius must be non-negative");
  }
  if (app_options.bilateral_spatial_sigma <= 0.0F ||
      app_options.bilateral_depth_sigma <= 0.0F) {
    throw std::invalid_argument("bilateral sigmas must be positive");
  }
}

}  // namespace app
