#include "app_options.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/trilinear.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <string>

#include "pipeline_config.hpp"

namespace app {
namespace {

// Default ICP iteration counts per pyramid level when the user has not
// requested a fixed value. Higher levels use fewer iterations because they
// operate on coarser pyramid levels with fewer correspondences.
constexpr unsigned int kDefaultIcpIterationsLevel0 = 10U;
constexpr unsigned int kDefaultIcpIterationsLevel1 = 5U;
constexpr unsigned int kDefaultIcpIterationsLevelDeep = 4U;

}  // namespace

kinectfusion::Vec3f AppOptions::volume_origin() const {
  const auto resolution = static_cast<float>(volume_resolution);
  const float half_extent = 0.5F * resolution * voxel_size;
  return kinectfusion::Vec3f{
      .x = -half_extent, .y = -half_extent, .z = -volume_camera_margin};
}

kinectfusion::VolumeGeometry AppOptions::volume_geometry() const {
  return kinectfusion::VolumeGeometry{
      .resolution = {.x = volume_resolution,
                     .y = volume_resolution,
                     .z = volume_resolution},
      .voxel_size = voxel_size,
      .origin = volume_origin(),
      .truncation_distance = truncation_distance};
}

kinectfusion::TsdfRuleVariant AppOptions::tsdf_rule() const {
  return tsdf_rule_from_name(tsdf_variant);
}

kinectfusion::PipelineConfig AppOptions::pipeline_config() const {
  return kinectfusion::PipelineConfig{
      .name = "baseline",
      .space = space,
      .tsdf_rule = tsdf_rule(),
      .integration = tsdf_options(),
      .raycast = raycast_options(),
      .volume = volume_geometry(),
      .voxel = voxel_store_from_name(voxel_store),
      .color = color_store_from_name(color_store),
      .raycast_backend = raycast_backend_from_name(raycast_backend),
      .storage = storage_layout_from_name(storage_layout),
      .sparse_block_capacity = sparse_capacity,
      .icp_damping = {.mode = icp_damping_mode_from_name(icp_damping),
                      .lambda = icp_lambda},
      .icp_adaptive_damping = icp_adaptive_damping};
}

kinectfusion::PipelineSetConfig AppOptions::pipeline_set_config() const {
  if (pipelines_config.empty()) {
    return kinectfusion::PipelineSetConfig{
        .pipelines = {pipeline_config()},
        .reference = {},
        .compare_every_n_frames = compare_every_n_frames};
  }
  return parse_pipeline_set(pipelines_config, pipeline_config(),
                            compare_every_n_frames);
}

kinectfusion::RaycastOptions AppOptions::raycast_options() const {
  return kinectfusion::RaycastOptions{
      .min_depth = min_depth,
      .max_depth = max_depth,
      .tsdf_corner_policy = raycast_tsdf_from_valid_corners
                                ? kinectfusion::CornerPolicy::kRequireAll
                                : kinectfusion::CornerPolicy::kSkipMissing,
      .cell_gradient_normals = cell_gradient_normals,
      .seed_from_previous = raycast_seed_previous};
}

kinectfusion::RaycastCamera AppOptions::raycast_camera(
    const kinectfusion::VirtualSensor& sensor,
    const Eigen::Matrix4f& camera_to_world, unsigned int level) {
  const auto scale = 1U << level;
  return kinectfusion::RaycastCamera{
      .intrinsics = sensor.depth_intrinsics().scaled(level),
      .width = sensor.depth_image().width() / scale,
      .height = sensor.depth_image().height() / scale,
      .camera_to_world = camera_to_world};
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
      .mode = integration_mode_from_name(integration_mode)};
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
      .max_condition_number = max_icp_condition_number,
      .device_graph_build = icp_capture_graph
                                ? kinectfusion::IcpGraphBuild::kCaptured
                                : kinectfusion::IcpGraphBuild::kExplicit,
      .device_solve = icp_device_solve,
      .damping = {.mode = icp_damping_mode_from_name(icp_damping),
                  .lambda = icp_lambda},
      .adaptive_damping = icp_adaptive_damping};
}

unsigned int AppOptions::icp_iterations_for_level(unsigned int level) const {
  if (icp_iterations > 0) {
    return static_cast<unsigned int>(icp_iterations);
  }
  if (level == 0) {
    return kDefaultIcpIterationsLevel0;
  }
  if (level == 1) {
    return kDefaultIcpIterationsLevel1;
  }
  return kDefaultIcpIterationsLevelDeep;
}

void configure_cli(CLI::App& app, AppOptions& app_options) {
  app.set_version_flag("--version", std::string{KINECTFUSION_VERSION});
  app.add_option("dataset", app_options.dataset_dir,
                 "TUM RGB-D dataset directory.");
  app.add_option("--frames", app_options.max_frames,
                 "Maximum number of frames to process. Set -1 to process all "
                 "frames.");
  app.add_option("--volume-resolution", app_options.volume_resolution,
                 "Number of voxels along each edge of the cubic TSDF volume.");
  app.add_option("--voxel-size", app_options.voxel_size,
                 "Edge length of one voxel, in meters.");
  app.add_option("--truncation-distance", app_options.truncation_distance,
                 "TSDF truncation distance, in meters.");
  app.add_option("--volume-camera-margin", app_options.volume_camera_margin,
                 "Extra volume space behind the initial camera, in meters.");
  app.add_option("--depth-scale", app_options.depth_scale,
                 "Raw depth units for one meter. The TUM default is 5000.");
  app.add_option("--min-depth", app_options.min_depth,
                 "Minimum usable depth, in meters. The pipeline ignores "
                 "closer measurements.");
  app.add_option("--max-depth", app_options.max_depth,
                 "Maximum usable depth, in meters. The pipeline ignores "
                 "farther measurements.");
  app.add_option("--pyramid-levels", app_options.pyramid_levels,
                 "Number of depth pyramid levels for tracking.");
  app.add_option("--icp-iterations", app_options.icp_iterations,
                 "Fixed number of ICP iterations for each pyramid level. Set "
                 "-1 for the defaults [10, 5, 4].");
  app.add_option("--matching-distance", app_options.matching_distance,
                 "Maximum distance between ICP correspondences, in meters. "
                 "The search rejects pairs that are farther apart.");
  app.add_option("--min-normal-dot", app_options.min_normal_dot,
                 "Minimum dot product between correspondence normals. The "
                 "search rejects lower pairs. cos(15 deg) = 0.9659258.");
  app.add_option("--max-pose-update-translation",
                 app_options.max_pose_update_translation,
                 "Maximum accepted ICP translation update, in meters. A "
                 "larger update fails the frame.");
  app.add_option("--max-pose-update-rotation",
                 app_options.max_pose_update_rotation,
                 "Maximum accepted ICP rotation update, in radians. A larger "
                 "update fails the frame.");
  app.add_option("--min-icp-eigenvalue", app_options.min_icp_eigenvalue,
                 "Minimum eigenvalue of the ICP normal system. A smaller "
                 "value fails the frame as unconstrained.");
  app.add_option("--max-icp-condition-number",
                 app_options.max_icp_condition_number,
                 "Maximum condition number of the ICP normal system. A "
                 "larger value fails the frame as unstable.");
  app.add_option("--voxel", app_options.voxel_store,
                 "Storage of the geometric TSDF voxel. 'float' uses 8 bytes. "
                 "'quantized' (int16) and 'bf16' use 4 bytes.");
  app.add_option("--color", app_options.color_store,
                 "Storage of the volume color. 'float' fuses color into "
                 "16-byte color voxels. 'none' stores no color and renders "
                 "shaded geometry.");
  app.add_flag("--cell-normals", app_options.cell_gradient_normals,
               "Ablation: compute raycast normals from the 8 corners of the "
               "final sample. The default uses six extra trilinear samples.");
  app.add_flag("--raycast-seed-previous", app_options.raycast_seed_previous,
               "Ablation: start each ray just in front of the surface from "
               "the last frame. The default starts at the minimum depth.");
  app.add_option("--storage", app_options.storage_layout,
                 "Volume storage layout. 'dense' stores every voxel. "
                 "'sparse' allocates 8^3 voxel blocks along the truncation "
                 "band and requires --integration band.");
  app.add_option("--sparse-capacity", app_options.sparse_capacity,
                 "Number of blocks in the sparse pool. Set 0 to use one "
                 "quarter of the block count.");
  app.add_option("--integration", app_options.integration_mode,
                 "Integration sweep. 'full' updates all voxels each frame "
                 "and carves free space. 'band' updates only blocks near the "
                 "measured surface and does not carve; it is lossy.");
  app.add_option("--raycast", app_options.raycast_backend,
                 "Raycast backend. 'march' samples every step. "
                 "'bitmap-march' skips empty blocks with identical output. "
                 "'band-march' skips far-from-surface blocks with "
                 "approximate output.");
  app.add_flag("--icp-device-solve", app_options.icp_device_solve,
               "Ablation: run the whole ICP Gauss-Newton loop on the GPU, "
               "with one synchronization for each pyramid level.");
  app.add_flag("--icp-capture-graph", app_options.icp_capture_graph,
               "Ablation: record the device ICP reduction graph with stream "
               "capture. The default builds it with the explicit node API.");
  app.add_option("--icp-damping", app_options.icp_damping,
                 "ICP solver damping. 'none' is Gauss-Newton. 'identity' adds "
                 "lambda*I (Levenberg). 'diagonal' adds lambda*diag(JtJ) "
                 "(Marquardt). Damping replaces the eigenvalue veto.");
  app.add_option("--icp-lambda", app_options.icp_lambda,
                 "Initial damping lambda. Fixed unless "
                 "--icp-adaptive-damping is set.");
  app.add_flag("--icp-adaptive-damping", app_options.icp_adaptive_damping,
               "Ablation: adapt lambda per trial and roll back trials that "
               "raise the cost. Requires --icp-damping. This is the full "
               "Levenberg-Marquardt driver.");
  app.add_flag("--projective-tsdf-distance,!--no-projective-tsdf-distance",
               app_options.projective_tsdf_distance,
               "Use the lambda-corrected projective TSDF distance. Disable "
               "to use the camera z distance.");
  app.add_flag("--distance-scaled-truncation,!--no-distance-scaled-truncation",
               app_options.distance_scaled_truncation,
               "Widen the TSDF truncation band linearly with measured "
               "depth.");
  app.add_option("--truncation-distance-scale",
                 app_options.truncation_distance_scale,
                 "Widening of the truncation band for each meter of depth.");
  app.add_option("--tsdf-variant", app_options.tsdf_variant,
                 "TSDF update rule. 'angle-weighted' weights observations by "
                 "cos(theta)/depth. 'classic' uses a constant weight.")
      ->check(CLI::IsMember({"angle-weighted", "classic"}));
  app.add_option("--space", app_options.space,
                 "Memory space of the pipeline: 'cpu' or 'cuda'. Falls back "
                 "to cpu with a warning when cuda is unavailable.")
      ->transform(
          CLI::CheckedTransformer(memory_space_names(), CLI::ignore_case));
  app.add_option("--pipelines", app_options.pipelines_config,
                 "TOML file that describes an ablation pipeline set "
                 "([[pipeline]] tables). CLI options give the defaults for "
                 "each pipeline.");
  app.add_option("--compare-every", app_options.compare_every_n_frames,
                 "Number of frames between pipeline deviation reports. Set 0 "
                 "or less to disable the reports.");
  app.add_flag("--bilateral-filter,!--no-bilateral-filter",
               app_options.bilateral_filter,
               "Filter the tracking depth with a bilateral filter.");
  app.add_option("--bilateral-radius", app_options.bilateral_radius,
                 "Radius of the bilateral filter, in pixels.");
  app.add_option("--bilateral-spatial-sigma",
                 app_options.bilateral_spatial_sigma,
                 "Spatial sigma of the bilateral filter, in pixels.");
  app.add_option("--bilateral-depth-sigma", app_options.bilateral_depth_sigma,
                 "Depth sigma of the bilateral filter, in meters.");
  app.add_flag("--interactive-relocalization",
               app_options.interactive_relocalization,
               "Pause after a tracking failure. Continue on user input.");
  app.add_flag("--preload", app_options.preload_frames,
               "Decode the whole dataset into memory before the frame loop "
               "(~2 MB for each frame). The loop then measures pure pipeline "
               "throughput.");
  app.add_option("--output-dir", app_options.output_dir,
                 "Directory for raycast images and point clouds.");
  app.add_flag("--write-raycast-images,!--no-write-raycast-images",
               app_options.write_raycast_images,
               "Write the raycast color images as PNG files.");
  app.add_flag("--write-point-clouds,!--no-write-point-clouds",
               app_options.write_point_clouds,
               "Write the raycast point clouds as binary PLY files.");
  app.add_flag("--write-mesh,!--no-write-mesh", app_options.write_mesh,
               "Write the final TSDF surface as a triangle mesh (mesh.ply).");
  app.add_option("--mesh-min-weight", app_options.mesh_min_weight,
                 "Minimum TSDF weight for the mesh. Cells with a corner below "
                 "this weight are not meshed. Set 0 to mesh every observed "
                 "voxel.");
  app.add_flag(
      "--raycast-tsdf-from-valid-corners,!--no-raycast-tsdf-from-valid-corners",
      app_options.raycast_tsdf_from_valid_corners,
      "Interpolate the raycast TSDF only where all 8 corners are valid. The "
      "default reweights the valid corners.");
}

void validate_options(const AppOptions& app_options) {
  using kinectfusion::require;
  require(app_options.volume_resolution > 0,
          "volume resolution must be positive");
  require(app_options.voxel_size > 0.0F, "voxel size must be positive");
  const float extent = static_cast<float>(app_options.volume_resolution) *
                       app_options.voxel_size;
  require(app_options.volume_camera_margin >= 0.0F &&
              app_options.volume_camera_margin <= extent,
          "volume camera margin must be within [0, volume extent]");
  require(app_options.truncation_distance > 0.0F,
          "truncation distance must be positive");
  require(app_options.depth_scale > 0.0F, "depth scale must be positive");
  require(app_options.min_depth >= 0.0F &&
              app_options.max_depth > app_options.min_depth,
          "depth range is invalid");
  require(app_options.pyramid_levels > 0U, "pyramid levels must be positive");
  require(app_options.icp_iterations > 0 || app_options.icp_iterations == -1,
          "ICP iterations must be positive or -1");
  require(app_options.matching_distance > 0.0F,
          "matching distance must be positive");
  require(
      app_options.min_normal_dot >= -1.0F && app_options.min_normal_dot <= 1.0F,
      "min normal dot must be within [-1, 1]");
  require(app_options.truncation_distance_scale >= 0.0F,
          "truncation distance scale must be non-negative");
  require(app_options.bilateral_radius >= 0,
          "bilateral radius must be non-negative");
  require(app_options.bilateral_spatial_sigma > 0.0F &&
              app_options.bilateral_depth_sigma > 0.0F,
          "bilateral sigmas must be positive");
  require(app_options.icp_lambda > 0.0F, "ICP lambda must be positive");
  require(!app_options.icp_adaptive_damping ||
              icp_damping_mode_from_name(app_options.icp_damping) !=
                  kinectfusion::IcpDampingMode::kNone,
          "--icp-adaptive-damping requires --icp-damping");
}

}  // namespace app
