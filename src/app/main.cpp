#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/image_proc/write_png.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <stdexcept>
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
  std::filesystem::path output_dir{"kinectfusion_output"};
  bool write_raycast_images{true};
  bool write_point_clouds{true};
};

[[nodiscard]] Eigen::Vector3f volume_origin(const AppOptions& options) {
  const auto resolution = static_cast<float>(options.volume_resolution);
  const float half_extent = 0.5F * resolution * options.voxel_size;
  return Eigen::Vector3f{-half_extent, -half_extent,
                         -options.volume_camera_margin};
}

[[nodiscard]] unsigned int icp_iterations_for_level(unsigned int level) {
  if (level == 0) {
    return 10U;
  }
  if (level == 1) {
    return 5U;
  }
  return 4U;
}

// Raycasts the model at a given pyramid level (level 0 is full resolution).
[[nodiscard]] kinectfusion::SurfaceMaps raycast_level(
    const kinectfusion::Volume& volume,
    const kinectfusion::VirtualSensor& sensor,
    const Eigen::Matrix4f& camera_to_world, unsigned int level) {
  const auto scale = 1U << level;
  return volume.raycast(
      kinectfusion::scale_intrinsics(sensor.depth_intrinsics(), level),
      sensor.depth_image().width() / scale,
      sensor.depth_image().height() / scale, camera_to_world);
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
           << static_cast<int>(color.x()) << ' ' << static_cast<int>(color.y())
           << ' ' << static_cast<int>(color.z()) << '\n';
  }
}

void write_outputs(const AppOptions& app_options,
                   const kinectfusion::SurfaceMaps& maps, int frame_index) {
  if (!app_options.write_raycast_images && !app_options.write_point_clouds) {
    return;
  }

  std::filesystem::create_directories(app_options.output_dir);
  const auto stem = frame_stem(frame_index);
  if (app_options.write_raycast_images) {
    write_raycast_image(maps, app_options.output_dir / (stem + "_raycast.png"));
  }
  if (app_options.write_point_clouds) {
    write_raycast_point_cloud(
        maps, app_options.output_dir / (stem + "_raycast.ply"));
  }
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
  app.add_option("--output-dir", app_options.output_dir,
                 "Directory for raycast images and point clouds.");
  app.add_flag("--write-raycast-images,!--no-write-raycast-images",
               app_options.write_raycast_images,
               "Write raycast color images as PNG files.");
  app.add_flag("--write-point-clouds,!--no-write-point-clouds",
               app_options.write_point_clouds,
               "Write raycast point clouds as ASCII PLY files.");
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
}

}  // namespace

int main(int argc, const char** argv) {
  try {
    AppOptions app_options;

    CLI::App app{"CPU KinectFusion reconstruction"};
    configure_cli(app, app_options);

    CLI11_PARSE(app, argc, argv);

    validate_options(app_options);

    // load first frame to sensor
    kinectfusion::VirtualSensor sensor;
    if (!sensor.init(app_options.dataset_dir)) {
      spdlog::error("Failed to initialize dataset: {}",
                    app_options.dataset_dir.string());
      return EXIT_FAILURE;
    }

    const Eigen::Vector3i resolution{
        app_options.volume_resolution,
        app_options.volume_resolution,
        app_options.volume_resolution
    };
    kinectfusion::Volume volume{
        resolution, app_options.voxel_size,
        volume_origin(app_options),
        app_options.truncation_distance
    };

    kinectfusion::ProjectiveIcpTracker tracker;

    // camera position and direction
    Eigen::Matrix4f camera_to_world = Eigen::Matrix4f::Identity();

    if (!sensor.process_next_frame()) {
      spdlog::error("Dataset contains no frames");
      return EXIT_FAILURE;
    }

    // build pyramids
    const auto initial_pyramid = kinectfusion::build_surface_pyramid(
        sensor.depth_image(), sensor.depth_intrinsics());
    const auto* initial_normals =
        initial_pyramid.empty() ? nullptr
                                : &initial_pyramid.front().maps.normals;

    // integrate first frame into volume
    volume.integrate_depth_image(sensor.depth_image(),
                                 sensor.depth_intrinsics(),
                                 camera_to_world.inverse(),
                                 &sensor.color_image(), initial_normals);

    // raycast volume and get model surface maps
    auto model_maps = raycast_level(volume, sensor, camera_to_world, 0);
    write_outputs(app_options, model_maps, 0);

    spdlog::info("Initialized reconstruction from frame 0: observed voxels={}",
                 volume.observed_voxel_count());

    int processed_frames = 1;
    bool relocalizing = false;
    int relocalization_frames = 0;

    while (app_options.max_frames < 0 ||
           processed_frames < app_options.max_frames) {
      // load frame
      if (!sensor.process_next_frame()) {
        break;
      }

      // build pyramid
      const auto live_pyramid = kinectfusion::build_surface_pyramid(
          sensor.depth_image(), sensor.depth_intrinsics());
      if (live_pyramid.empty()) {
        spdlog::warn("Frame {} produced no depth pyramid",
                     sensor.current_frame_index());
        continue;
      }

      kinectfusion::ProjectiveIcpResult tracking;
      tracking.pose = camera_to_world;

      // tracked_pose is current_camera_pose
      auto tracked_pose = camera_to_world;

      // for each level: raycast at current pose, run icp -> new tracked pose
      for (std::size_t level_index = live_pyramid.size(); level_index-- > 0;) {
        const auto level = static_cast<unsigned int>(level_index);
        tracker.set_iterations(icp_iterations_for_level(level));
        const auto level_model_maps =
            raycast_level(volume, sensor, tracked_pose, level);
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
        model_maps = raycast_level(volume, sensor, camera_to_world, 0);
        spdlog::warn(
            "Frame {} tracking rejected: status={} correspondences={} "
            "mean_distance={}",
            sensor.current_frame_index(), status_name(tracking.status),
            tracking.correspondences, tracking.mean_point_distance);
        ++processed_frames;
        continue;
      }

      if (relocalizing) {
        spdlog::info(
            "Relocalized at frame {} after {} frame(s): correspondences={} "
            "mean_distance={}",
            sensor.current_frame_index(), relocalization_frames,
            tracking.correspondences, tracking.mean_point_distance);
        relocalizing = false;
        relocalization_frames = 0;
      }

      // current_camera_pose = tracked_pose
      camera_to_world = tracked_pose;

      // integrate current depth frame into volume using new pose
      volume.integrate_depth_image(
          sensor.depth_image(), sensor.depth_intrinsics(),
          camera_to_world.inverse(), &sensor.color_image(),
          &live_pyramid.front().maps.normals);

      // raycast volume and get updated surface model maps
      model_maps = raycast_level(volume, sensor, camera_to_world, 0);
      write_outputs(app_options, model_maps, sensor.current_frame_index());

      spdlog::info(
          "Frame {} integrated: correspondences={} mean_distance={} "
          "observed_voxels={}",
          sensor.current_frame_index(), tracking.correspondences,
          tracking.mean_point_distance, volume.observed_voxel_count());
      ++processed_frames;
    }

    spdlog::info(
        "Finished reconstruction: processed_frames={} observed_voxels={}",
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
