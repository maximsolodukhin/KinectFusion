#ifndef KINECTFUSION_SRC_APP_APP_OPTIONS_HPP
#define KINECTFUSION_SRC_APP_APP_OPTIONS_HPP

#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>

namespace app {

// Defaults for AppOptions below. Values that mirror library-level defaults
// (in kinectfusion::) are pulled in by name so they stay in sync.
inline constexpr int default_max_frames = 30;
inline constexpr std::size_t default_volume_resolution = 512;
inline constexpr float default_voxel_size_meters = 0.01F;
inline constexpr float default_truncation_distance_meters = 0.05F;
inline constexpr float default_volume_camera_margin_meters = 2.56F;
inline constexpr int default_fixed_icp_iterations_disabled = -1;
inline constexpr float default_matching_distance_meters = 0.05F;
// cos(15 degrees); correspondences whose normals disagree by more are dropped.
inline constexpr float default_min_normal_dot_15_deg = 0.9659258F;
inline constexpr float default_max_pose_update_translation_meters = 0.15F;
inline constexpr float default_max_pose_update_rotation_radians = 0.35F;
inline constexpr float default_min_icp_eigenvalue = 1.0e-6F;
inline constexpr float default_max_icp_condition_number = 1.0e6F;

// Command-line configuration for the reconstruction app. Holds the raw user
// inputs and knows how to turn them into the library option structs.
struct AppOptions {
  std::filesystem::path dataset_dir{"../data/rgbd_dataset_freiburg1_xyz"};
  int max_frames{default_max_frames};
  std::size_t volume_resolution{default_volume_resolution};
  float voxel_size{default_voxel_size_meters};
  float truncation_distance{default_truncation_distance_meters};
  float volume_camera_margin{default_volume_camera_margin_meters};
  float depth_scale{kinectfusion::default_depth_processing_depth_scale};
  float min_depth{kinectfusion::default_depth_processing_min_depth};
  float max_depth{kinectfusion::default_depth_processing_max_depth};
  unsigned int pyramid_levels{kinectfusion::default_depth_pyramid_levels};
  int icp_iterations{default_fixed_icp_iterations_disabled};
  float matching_distance{default_matching_distance_meters};
  float min_normal_dot{default_min_normal_dot_15_deg};
  float max_pose_update_translation{default_max_pose_update_translation_meters};
  float max_pose_update_rotation{default_max_pose_update_rotation_radians};
  float min_icp_eigenvalue{default_min_icp_eigenvalue};
  float max_icp_condition_number{default_max_icp_condition_number};
  bool projective_tsdf_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{
      kinectfusion::default_truncation_distance_scale};
  bool view_angle_weighting{true};
  bool bilateral_filter{true};
  int bilateral_radius{kinectfusion::default_bilateral_radius_pixels};
  float bilateral_spatial_sigma{
      kinectfusion::default_bilateral_spatial_sigma_pixels};
  float bilateral_depth_sigma{
      kinectfusion::default_bilateral_depth_sigma_meters};
  bool interactive_relocalization{false};
  std::filesystem::path output_dir{"kinectfusion_output"};
  bool write_raycast_images{true};
  bool write_point_clouds{true};
  bool raycast_tsdf_from_valid_corners{false};

  // Origin of the TSDF volume in world space, derived from its extent and the
  // requested margin behind the initial camera.
  [[nodiscard]] kinectfusion::Vec3f volume_origin() const;

  [[nodiscard]] kinectfusion::RaycastOptions raycast_options(
      const kinectfusion::VirtualSensor& sensor,
      const Eigen::Matrix4f& camera_to_world, unsigned int level = 0) const;

  [[nodiscard]] kinectfusion::DepthProcessingOptions depth_options() const;

  [[nodiscard]] kinectfusion::TsdfIntegrationOptions tsdf_options() const;

  [[nodiscard]] kinectfusion::ProjectiveIcpOptions icp_options() const;

  // Per-pyramid-level ICP iteration budget: a fixed override, or the default
  // [10, 5, 4] schedule when icp_iterations is left at -1.
  [[nodiscard]] unsigned int icp_iterations_for_level(unsigned int level) const;
};

// Binds every AppOptions field to a CLI11 flag/option.
void configure_cli(CLI::App& app, AppOptions& app_options);

// Throws std::invalid_argument if any option is out of its valid range.
void validate_options(const AppOptions& app_options);

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_APP_OPTIONS_HPP */
