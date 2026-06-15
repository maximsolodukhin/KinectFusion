#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/image_proc/write_png.hpp>
#include <kinectfusion/projective_icp.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <string>
#include <string_view>

namespace {

struct AppOptions {
  std::filesystem::path dataset_dir{"../data/rgbd_dataset_freiburg1_xyz"};
  int max_frames{30};
  int volume_resolution{512};
  float voxel_size{0.01F};
  float truncation_distance{0.05F};
  float volume_camera_margin{2.56F};
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
  std::string normal_computation{"central"};
  bool interactive_relocalization{false};
  std::filesystem::path output_dir{"kinectfusion_output"};
  bool write_raycast_images{true};
  bool write_point_clouds{true};
  bool raycast_tsdf_from_valid_corners{true};
};

[[nodiscard]] Eigen::Vector3f volume_origin(const AppOptions& options) {
  const auto resolution = static_cast<float>(options.volume_resolution);
  const float half_extent = 0.5F * resolution * options.voxel_size;
  return Eigen::Vector3f{-half_extent, -half_extent,
                         -options.volume_camera_margin};
}

[[nodiscard]] kinectfusion::RaycastOptions make_raycast_options(
    const kinectfusion::VirtualSensor& sensor,
    const Eigen::Matrix4f& camera_to_world,
    const AppOptions& app_options,
    unsigned int level = 0) {
  const auto scale = 1U << level;
  return kinectfusion::RaycastOptions{
      .intrinsics = kinectfusion::scale_intrinsics(sensor.depth_intrinsics(),
                                                   level),
      .width = sensor.depth_image().width() / scale,
      .height = sensor.depth_image().height() / scale,
      .camera_to_world = camera_to_world,
      .min_depth = app_options.min_depth,
      .max_depth = app_options.max_depth,
      .step_scale = 1.0F,
      .tsdf_from_valid_corners = app_options.raycast_tsdf_from_valid_corners};
}

[[nodiscard]] kinectfusion::DepthProcessingOptions make_depth_options(
    const AppOptions& app_options) {
  const auto normal_computation =
      app_options.normal_computation == "central"
          ? kinectfusion::NormalComputation::central_difference
          : kinectfusion::NormalComputation::paper_forward;
  return kinectfusion::DepthProcessingOptions{
      .levels = app_options.pyramid_levels,
      .depth_scale = kinectfusion::kTumDepthScale,
      .min_depth = app_options.min_depth,
      .max_depth = app_options.max_depth,
      .max_normal_depth_jump = 0.1F,
      .max_downsample_depth_jump = 0.1F,
      .bilateral_filter = app_options.bilateral_filter,
      .bilateral_radius = app_options.bilateral_radius,
      .bilateral_spatial_sigma = app_options.bilateral_spatial_sigma,
      .bilateral_depth_sigma = app_options.bilateral_depth_sigma,
      .normal_computation = normal_computation};
}

[[nodiscard]] kinectfusion::TsdfIntegrationOptions make_tsdf_options(
    const AppOptions& app_options) {
  return kinectfusion::TsdfIntegrationOptions{
      .depth_scale = kinectfusion::kTumDepthScale,
      .observation_weight = 1.0F,
      .max_weight = 196.0F,
      .min_depth = app_options.min_depth,
      .max_depth = app_options.max_depth,
      .projective_distance = app_options.projective_tsdf_distance,
      .distance_scaled_truncation = app_options.distance_scaled_truncation,
      .truncation_distance_scale = app_options.truncation_distance_scale,
      .view_angle_weighting = app_options.view_angle_weighting};
}

[[nodiscard]] unsigned int icp_iterations_for_level(
    const AppOptions& app_options,
    unsigned int level) {
  if (app_options.icp_iterations > 0) {
    return static_cast<unsigned int>(app_options.icp_iterations);
  }
  if (level == 0) {
    return 10;
  }
  if (level == 1) {
    return 5;
  }
  return 4;
}

[[nodiscard]] constexpr std::string_view status_name(
    kinectfusion::ProjectiveIcpStatus status) {
  switch (status) {
    case kinectfusion::ProjectiveIcpStatus::converged:
      return "converged";
    case kinectfusion::ProjectiveIcpStatus::too_few_correspondences:
      return "too_few_correspondences";
    case kinectfusion::ProjectiveIcpStatus::unconstrained_system:
      return "unconstrained_system";
    case kinectfusion::ProjectiveIcpStatus::solve_failed:
      return "solve_failed";
    case kinectfusion::ProjectiveIcpStatus::update_too_large:
      return "update_too_large";
  }
  return "unknown";
}

[[nodiscard]] std::string frame_stem(int frame_index) {
  return std::format("frame_{:06d}", frame_index);
}

void write_raycast_image(const kinectfusion::SurfaceMaps& maps,
                         const std::filesystem::path& path) {
  kinectfusion::image_proc::write_png(maps.colors, path.string());
}

void write_raycast_point_cloud(const kinectfusion::SurfaceMaps& maps,
                               const std::filesystem::path& path) {
  const auto& points = maps.points.data();
  const auto& normals = maps.normals.data();
  const auto& colors = maps.colors.data();

  std::size_t vertex_count = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (points.at(i).allFinite() && normals.at(i).allFinite()) {
      ++vertex_count;
    }
  }

  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"Failed to open point cloud output: " +
                             path.string()};
  }

  output << "ply\n"
         << "format ascii 1.0\n"
         << "element vertex " << vertex_count << '\n'
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "property float nx\n"
         << "property float ny\n"
         << "property float nz\n"
         << "property uchar red\n"
         << "property uchar green\n"
         << "property uchar blue\n"
         << "end_header\n";

  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& point = points.at(i);
    const auto& normal = normals.at(i);
    if (!point.allFinite() || !normal.allFinite()) {
      continue;
    }

    const auto color = kinectfusion::rgba_from_pixel(colors.at(i));
    output << point.x() << ' ' << point.y() << ' ' << point.z() << ' '
           << normal.x() << ' ' << normal.y() << ' ' << normal.z() << ' '
           << static_cast<int>(color.x()) << ' '
           << static_cast<int>(color.y()) << ' '
           << static_cast<int>(color.z()) << '\n';
  }
}

void write_outputs(const AppOptions& app_options,
                   const kinectfusion::SurfaceMaps& maps,
                   int frame_index) {
  if (!app_options.write_raycast_images && !app_options.write_point_clouds) {
    return;
  }

  std::filesystem::create_directories(app_options.output_dir);
  const auto stem = frame_stem(frame_index);
  if (app_options.write_raycast_images) {
    write_raycast_image(maps, app_options.output_dir / (stem + "_raycast.png"));
  }
  if (app_options.write_point_clouds) {
    write_raycast_point_cloud(maps,
                              app_options.output_dir / (stem + "_raycast.ply"));
  }
}

}  // namespace

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
    app.add_option("--min-depth", app_options.min_depth,
                   "Minimum usable depth in meters.");
    app.add_option("--max-depth", app_options.max_depth,
                   "Maximum usable depth in meters.");
    app.add_option("--pyramid-levels", app_options.pyramid_levels,
                   "Number of depth pyramid levels to build.");
    app.add_option("--icp-iterations", app_options.icp_iterations,
                   "Fixed ICP iterations per level, or -1 (default) for the "
                   "paper's [10, 5, 4] coarse-to-fine schedule.");
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
                 "Use paper-style lambda-corrected projective TSDF distance.");
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
    app.add_option("--normal-computation", app_options.normal_computation,
                   "Normal stencil: paper or central.");
    app.add_flag("--interactive-relocalization",
                 app_options.interactive_relocalization,
                 "Pause after tracking failure so the user can realign the sensor.");
    app.add_option("--output-dir", app_options.output_dir,
                   "Directory for raycast images and point clouds.");
    app.add_flag("--write-raycast-images,!--no-write-raycast-images",
                 app_options.write_raycast_images,
                 "Write raycast color images as PNG files.");
    app.add_flag("--write-point-clouds,!--no-write-point-clouds",
                 app_options.write_point_clouds,
                 "Write raycast point clouds as ASCII PLY files.");
    app.add_flag("--raycast-tsdf-from-valid-corners,!--strict-raycast-tsdf",
                 app_options.raycast_tsdf_from_valid_corners,
                 "If true, interpolate only using valid corner samples. Otherwise the values are not ");
}

void validate_options(const AppOptions& app_options) {
    if (app_options.volume_resolution <= 0) {
        throw std::invalid_argument("volume resolution must be positive");
    }
    const float extent =
        static_cast<float>(app_options.volume_resolution) * app_options.voxel_size;
    if (app_options.volume_camera_margin < 0.0F ||
        app_options.volume_camera_margin > extent) {
        throw std::invalid_argument(
            "volume camera margin must be within [0, volume extent]");
    }
    if (app_options.normal_computation != "paper" &&
        app_options.normal_computation != "central") {
        throw std::invalid_argument("normal computation must be 'paper' or 'central'");
    }
}

int main(int argc, const char** argv) {
  try {
    AppOptions app_options;

    CLI::App app{"CPU KinectFusion reconstruction"};
    configure_cli(app, app_options);

    CLI11_PARSE(app, argc, argv);

    validate_options(app_options);

    kinectfusion::VirtualSensor sensor;
    if (!sensor.init(app_options.dataset_dir)) {
      spdlog::error("Failed to initialize dataset: {}",
                    app_options.dataset_dir.string());
      return EXIT_FAILURE;
    }

    const Eigen::Vector3i resolution{
        app_options.volume_resolution,
        app_options.volume_resolution,
        app_options.volume_resolution};
    kinectfusion::Volume volume{resolution, app_options.voxel_size,
                                volume_origin(app_options),
                                app_options.truncation_distance};

    kinectfusion::ProjectiveIcpTracker tracker{kinectfusion::ProjectiveIcpOptions{
        .iterations = icp_iterations_for_level(app_options, 0),
        .min_correspondences = 64,
        .max_point_distance = app_options.matching_distance,
        .min_normal_dot = app_options.min_normal_dot,
        .min_update_translation = 1.0e-5F,
        .min_update_rotation = 1.0e-5F,
        .max_update_translation = app_options.max_pose_update_translation,
        .max_update_rotation = app_options.max_pose_update_rotation,
        .min_system_eigenvalue = app_options.min_icp_eigenvalue,
        .max_condition_number = app_options.max_icp_condition_number}};

    const auto depth_options = make_depth_options(app_options);
    const auto tsdf_options = make_tsdf_options(app_options);
    Eigen::Matrix4f camera_to_world = Eigen::Matrix4f::Identity();

    if (!sensor.process_next_frame()) {
      spdlog::error("Dataset contains no frames");
      return EXIT_FAILURE;
    }

    const auto initial_pyramid = kinectfusion::build_surface_pyramid(
        sensor.depth_image(), sensor.depth_intrinsics(),
        Eigen::Matrix4f::Identity(), depth_options);
    const auto* initial_normals =
        initial_pyramid.empty() ? nullptr : &initial_pyramid.front().maps.normals;
    volume.integrate_depth_image(sensor.depth_image(), sensor.depth_intrinsics(),
                                 camera_to_world.inverse(), tsdf_options,
                                 &sensor.color_image(), initial_normals);
    auto model_maps =
        volume.raycast(make_raycast_options(sensor, camera_to_world, app_options));
    write_outputs(app_options, model_maps, 0);

    spdlog::info("Initialized reconstruction from frame 0: observed voxels={}",
                 volume.observed_voxel_count());
    
    int processed_frames = 1;
    bool relocalizing = false;
    int relocalization_frames = 0;

    while (app_options.max_frames < 0 ||
           processed_frames < app_options.max_frames) {
      if (!sensor.process_next_frame()) {
        break;
      }

      const auto live_pyramid = kinectfusion::build_surface_pyramid(
          sensor.depth_image(), sensor.depth_intrinsics(),
          Eigen::Matrix4f::Identity(), depth_options);
      if (live_pyramid.empty()) {
        spdlog::warn("Frame {} produced no depth pyramid",
                     sensor.current_frame_index());
        continue;
      }

      kinectfusion::ProjectiveIcpResult tracking;
      tracking.pose = camera_to_world;
      auto tracked_pose = camera_to_world;
      for (std::size_t level_index = live_pyramid.size(); level_index-- > 0;) {
        const auto level = static_cast<unsigned int>(level_index);
        tracker.set_iterations(icp_iterations_for_level(app_options, level));
        const auto level_model_maps =
            volume.raycast(make_raycast_options(sensor, tracked_pose,
                                                app_options, level));
        const auto level_intrinsics =
            kinectfusion::scale_intrinsics(sensor.depth_intrinsics(), level);
        tracking = tracker.estimate_pose(live_pyramid.at(level_index).maps,
                                         level_model_maps, level_intrinsics,
                                         tracked_pose, tracked_pose);
        tracked_pose = tracking.pose;
        if (!tracking.converged) {
          break;
        }
      }

      if (!tracking.converged) {
        if (!relocalizing) {
          spdlog::warn("Entering relocalization mode at frame {}",
                       sensor.current_frame_index());
        }
        relocalizing = true;
        ++relocalization_frames;
        model_maps = volume.raycast(
            make_raycast_options(sensor, camera_to_world, app_options));
        spdlog::warn(
            "Frame {} tracking rejected: status={} correspondences={} mean_distance={} min_eigenvalue={} condition={} update_translation={} update_rotation={}",
            sensor.current_frame_index(), status_name(tracking.status),
            tracking.correspondences, tracking.mean_point_distance,
            tracking.min_system_eigenvalue, tracking.condition_number,
            tracking.update_translation, tracking.update_rotation);
        if (app_options.interactive_relocalization) {
          spdlog::warn(
              "Relocalization paused. Align the live sensor with the last model prediction and press Enter.");
          std::cin.get();
        }
        ++processed_frames;
        continue;
      }

      if (relocalizing) {
        spdlog::info(
            "Relocalized at frame {} after {} frame(s): correspondences={} mean_distance={}",
            sensor.current_frame_index(), relocalization_frames,
            tracking.correspondences, tracking.mean_point_distance);
        relocalizing = false;
        relocalization_frames = 0;
      }

      camera_to_world = tracked_pose;
      volume.integrate_depth_image(sensor.depth_image(), sensor.depth_intrinsics(),
                                   camera_to_world.inverse(), tsdf_options,
                                   &sensor.color_image(),
                                   &live_pyramid.front().maps.normals);
      model_maps = volume.raycast(
          make_raycast_options(sensor, camera_to_world, app_options));
      write_outputs(app_options, model_maps, sensor.current_frame_index());

      spdlog::info(
          "Frame {} integrated: correspondences={} mean_distance={} observed_voxels={}",
          sensor.current_frame_index(), tracking.correspondences,
          tracking.mean_point_distance, volume.observed_voxel_count());
      ++processed_frames;
    }

    spdlog::info("Finished reconstruction: processed_frames={} observed_voxels={}",
                 processed_frames, volume.observed_voxel_count());
  } catch (const std::exception& e) {
    try {
      spdlog::error("Unhandled exception in main: {}", e.what());
    } catch (...) {
      std::cerr << "Unhandled exception in main: " << e.what() << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
