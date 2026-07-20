#ifndef KINECTFUSION_SRC_APP_APP_OPTIONS_HPP
#define KINECTFUSION_SRC_APP_APP_OPTIONS_HPP

#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <string>

namespace app {

// Defaults for AppOptions below. Values that mirror library-level defaults
// (in kinectfusion::) are pulled in by name so they stay in sync.
inline constexpr int kDefaultMaxFrames = 30;
inline constexpr std::size_t kDefaultVolumeResolution = 512;
inline constexpr float kDefaultVoxelSizeMeters = 0.01F;
inline constexpr float kDefaultTruncationDistanceMeters = 0.05F;
inline constexpr float kDefaultVolumeCameraMarginMeters = 2.56F;
inline constexpr int kDefaultFixedIcpIterationsDisabled = -1;
inline constexpr float kDefaultMeshMinWeight = 2.0F;

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
  float icp_lambda{kinectfusion::kDefaultIcpDampingLambda};
  bool icp_capture_graph{false};
  bool icp_device_solve{false};
  bool icp_adaptive_damping{false};
  std::string icp_damping{"none"};
  std::string voxel_store{"float"};
  std::string color_store{"float"};
  std::string raycast_backend{"march"};
  std::string integration_mode{"full"};
  std::string storage_layout{"dense"};
  std::size_t sparse_capacity{0};
  bool projective_tsdf_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{
      kinectfusion::kDefaultTruncationDistanceScale};
  std::string tsdf_variant{"angle-weighted"};
  kinectfusion::MemorySpace space{kinectfusion::MemorySpace::kHost};
  std::filesystem::path pipelines_config;
  int compare_every_n_frames{1};
  bool bilateral_filter{true};
  int bilateral_radius{kinectfusion::kDefaultBilateralRadiusPixels};
  float bilateral_spatial_sigma{
      kinectfusion::kDefaultBilateralSpatialSigmaPixels};
  float bilateral_depth_sigma{kinectfusion::kDefaultBilateralDepthSigmaMeters};
  bool interactive_relocalization{false};
  bool preload_frames{false};
  std::filesystem::path output_dir{"kinectfusion_output"};
  bool write_raycast_images{true};
  bool write_point_clouds{true};
  bool write_mesh{true};
  float mesh_min_weight{kDefaultMeshMinWeight};
  bool raycast_tsdf_from_valid_corners{false};
  bool cell_gradient_normals{false};
  bool raycast_seed_previous{false};

  // Origin of the TSDF volume in world space, derived from its extent and the
  // requested margin behind the initial camera.
  [[nodiscard]] kinectfusion::Vec3f volume_origin() const;

  [[nodiscard]] kinectfusion::VolumeGeometry volume_geometry() const;

  // The TSDF update rule selected by --tsdf-variant.
  [[nodiscard]] kinectfusion::TsdfRuleVariant tsdf_rule() const;

  // The single pipeline described by the CLI options alone.
  [[nodiscard]] kinectfusion::PipelineConfig pipeline_config() const;

  // The pipeline set to run: the --pipelines TOML file when given (CLI
  // options provide the per-pipeline defaults), else the single CLI pipeline.
  [[nodiscard]] kinectfusion::PipelineSetConfig pipeline_set_config() const;

  [[nodiscard]] kinectfusion::RaycastOptions raycast_options() const;

  // Per-call raycast camera at the given pose and pyramid level. Static: the
  // camera is derived from the sensor and pose alone, not from any option.
  [[nodiscard]] static kinectfusion::RaycastCamera raycast_camera(
      const kinectfusion::VirtualSensor& sensor,
      const Eigen::Matrix4f& camera_to_world, unsigned int level = 0);

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
