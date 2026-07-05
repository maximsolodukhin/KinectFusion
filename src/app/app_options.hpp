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

// Command-line configuration for the reconstruction app. Holds the raw user
// inputs and knows how to turn them into the library option structs.
struct AppOptions {
  std::filesystem::path dataset_dir{"../data/rgbd_dataset_freiburg1_xyz"};
  int max_frames{30};
  std::size_t volume_resolution{512};
  float voxel_size{0.01F};
  float truncation_distance{0.05F};
  float volume_camera_margin{2.56F};
  float depth_scale{5000.0F};
  float min_depth{0.4F};
  float max_depth{8.0F};
  unsigned int pyramid_levels{3};
  int icp_iterations{-1};
  float matching_distance{0.05F};
  float min_normal_dot{0.9659258F};
  float max_pose_update_translation{0.15F};
  float max_pose_update_rotation{0.35F};
  float min_icp_eigenvalue{1.0e-6F};
  float max_icp_condition_number{1.0e6F};
  bool projective_tsdf_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{0.01F};
  bool view_angle_weighting{true};
  bool bilateral_filter{true};
  int bilateral_radius{2};
  float bilateral_spatial_sigma{2.0F};
  float bilateral_depth_sigma{0.08F};
  bool cuda_depth_processing{false};
  bool interactive_relocalization{false};
  std::filesystem::path output_dir{"kinectfusion_output"};
  bool write_raycast_images{true};
  bool write_point_clouds{true};
  bool raycast_tsdf_from_valid_corners{false};

  // Origin of the TSDF volume in world space, derived from its extent and the
  // requested margin behind the initial camera.
  [[nodiscard]] Eigen::Vector3f volume_origin() const;

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
