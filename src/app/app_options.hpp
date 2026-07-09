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
inline constexpr int kDefaultMaxFrames = 30;
inline constexpr std::size_t kDefaultVolumeResolution = 512;
inline constexpr float kDefaultVoxelSizeMeters = 0.01F;
inline constexpr float kDefaultTruncationDistanceMeters = 0.05F;
inline constexpr float kDefaultVolumeCameraMarginMeters = 2.56F;
inline constexpr int kDefaultFixedIcpIterationsDisabled = -1;

// Command-line configuration for the reconstruction app. Holds the raw user
// inputs and knows how to turn them into the library option structs.
struct AppOptions {
  std::filesystem::path dataset_dir{"../data/rgbd_dataset_freiburg1_xyz"};
  int max_frames{kDefaultMaxFrames};
  std::size_t volume_resolution{kDefaultVolumeResolution};
  float voxel_size{kDefaultVoxelSizeMeters};
  float truncation_distance{kDefaultTruncationDistanceMeters};
  float volume_camera_margin{kDefaultVolumeCameraMarginMeters};
  float depth_scale{kinectfusion::kDefaultTumDepthScale};
  float min_depth{kinectfusion::kDefaultMinDepthMeters};
  float max_depth{kinectfusion::kDefaultMaxDepthMeters};
  unsigned int pyramid_levels{kinectfusion::kDefaultDepthPyramidLevels};
  int icp_iterations{kDefaultFixedIcpIterationsDisabled};
  float matching_distance{kinectfusion::kDefaultMaxIcpPointDistanceMeters};
  float min_normal_dot{kinectfusion::kDefaultMinIcpNormalDot};
  float max_pose_update_translation{
      kinectfusion::kDefaultMaxIcpUpdateTranslationMeters};
  float max_pose_update_rotation{
      kinectfusion::kDefaultMaxIcpUpdateRotationRadians};
  float min_icp_eigenvalue{kinectfusion::kDefaultMinIcpSystemEigenvalue};
  float max_icp_condition_number{kinectfusion::kDefaultMaxIcpConditionNumber};
  bool projective_tsdf_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{
      kinectfusion::kDefaultTruncationDistanceScale};
  bool view_angle_weighting{true};
  bool bilateral_filter{true};
  int bilateral_radius{kinectfusion::kDefaultBilateralRadiusPixels};
  float bilateral_spatial_sigma{
      kinectfusion::kDefaultBilateralSpatialSigmaPixels};
  float bilateral_depth_sigma{kinectfusion::kDefaultBilateralDepthSigmaMeters};
  bool interactive_relocalization{false};
  std::filesystem::path output_dir{"kinectfusion_output"};
  bool write_raycast_images{true};
  bool write_point_clouds{true};
  bool raycast_tsdf_from_valid_corners{false};

  // Origin of the TSDF volume in world space, derived from its extent and the
  // requested margin behind the initial camera.
  [[nodiscard]] kinectfusion::Vec3f volume_origin() const;

  [[nodiscard]] kinectfusion::Volume make_volume() const;

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
