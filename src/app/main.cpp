#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <kinectfusion/depth_processing.hpp>
#ifdef KINECTFUSION_CUDA_ENABLED
#include <kinectfusion/depth_processing.cuh>
#endif
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

#include "app_options.hpp"
#include "frame_output.hpp"

namespace {

// Raycasts the model at a given pyramid level (level 0 is full resolution).
[[nodiscard]] kinectfusion::SurfaceMaps raycast_level(
    const kinectfusion::Volume& volume,
    const kinectfusion::VirtualSensor& sensor,
    const Eigen::Matrix4f& camera_to_world, const app::AppOptions& app_options,
    unsigned int level) {
  return volume.raycast(
      app_options.raycast_options(sensor, camera_to_world, level));
}

[[nodiscard]] std::vector<kinectfusion::DepthProcessingLevel>
build_live_pyramid(const kinectfusion::VirtualSensor& sensor,
                   const app::AppOptions& app_options,
                   const kinectfusion::DepthProcessingOptions& depth_options) {
#ifdef KINECTFUSION_CUDA_ENABLED
  if (app_options.cuda_depth_processing) {
    return kinectfusion::cuda::build_surface_pyramid(
        sensor.depth_image(), sensor.depth_intrinsics(),
        Eigen::Matrix4f::Identity(), depth_options);
  }
#endif
  return kinectfusion::build_surface_pyramid(
      sensor.depth_image(), sensor.depth_intrinsics(),
      Eigen::Matrix4f::Identity(), depth_options);
}

[[nodiscard]] constexpr std::string_view failure_name(
    kinectfusion::IcpFailure failure) {
  switch (failure) {
    case kinectfusion::IcpFailure::invalid_input:
      return "invalid_input";
    case kinectfusion::IcpFailure::too_few_correspondences:
      return "too_few_correspondences";
    case kinectfusion::IcpFailure::unconstrained_system:
      return "unconstrained_system";
    case kinectfusion::IcpFailure::solve_failed:
      return "solve_failed";
    case kinectfusion::IcpFailure::update_too_large:
      return "update_too_large";
  }
  return "unknown";
}

}  // namespace

int main(int argc, const char** argv) {
  try {
    app::AppOptions app_options;

    CLI::App app{"CPU KinectFusion reconstruction"};
    app::configure_cli(app, app_options);

    CLI11_PARSE(app, argc, argv);

    app::validate_options(app_options);

    // load first frame to sensor
    kinectfusion::VirtualSensor sensor;
    if (!sensor.init(app_options.dataset_dir)) {
      spdlog::error("Failed to initialize dataset: {}",
                    app_options.dataset_dir.string());
      return EXIT_FAILURE;
    }

    const kinectfusion::Vector3s resolution{
        app_options.volume_resolution,
        app_options.volume_resolution,
        app_options.volume_resolution
    };
    kinectfusion::Volume volume{
        resolution, app_options.voxel_size,
        app_options.volume_origin(),
        app_options.truncation_distance
    };

    kinectfusion::ProjectiveIcpTracker tracker{app_options.icp_options()};

    const auto depth_options = app_options.depth_options();
    const auto tsdf_options = app_options.tsdf_options();

    // camera position and direction
    Eigen::Matrix4f camera_to_world = Eigen::Matrix4f::Identity();

    if (!sensor.process_next_frame()) {
      spdlog::error("Dataset contains no frames");
      return EXIT_FAILURE;
    }

    // build pyramids
    const auto initial_pyramid =
        build_live_pyramid(sensor, app_options, depth_options);
    const auto* initial_normals =
        initial_pyramid.empty() ? nullptr
                                : &initial_pyramid.front().maps.normals;

    // integrate first frame into volume
    volume.integrate_depth_image(sensor.depth_image(),
                                 sensor.depth_intrinsics(),
                                 camera_to_world.inverse(),
                                 tsdf_options,
                                 &sensor.color_image(), initial_normals);

    // raycast volume and get model surface maps
    auto model_maps = raycast_level(volume, sensor, camera_to_world,
                                    app_options, 0);
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
      const auto live_pyramid =
          build_live_pyramid(sensor, app_options, depth_options);
      if (live_pyramid.empty()) {
        spdlog::warn("Frame {} produced no depth pyramid",
                     sensor.current_frame_index());
        continue;
      }

      kinectfusion::IcpOutcome tracking;
      tracking.pose = camera_to_world;

      // tracked_pose is current_camera_pose
      auto tracked_pose = camera_to_world;

      // for each level: raycast at current pose, run icp -> new tracked pose
      for (std::size_t level_index = live_pyramid.size(); level_index-- > 0;) {
        const auto level = static_cast<unsigned int>(level_index);
        const auto level_model_maps =
            raycast_level(volume, sensor, tracked_pose, app_options, level);
        const auto level_intrinsics = sensor.depth_intrinsics().scaled(level);
        tracking = tracker.estimate_pose(
            app_options.icp_iterations_for_level(level),
            live_pyramid.at(level_index).maps, level_model_maps,
            level_intrinsics, tracked_pose, tracked_pose);
        tracked_pose = tracking.pose;
        if (!tracking.result) {
          break;
        }
      }

      if (!tracking.result) {
        if (!relocalizing) {
          spdlog::warn("Entering relocalization mode at frame {}",
                       sensor.current_frame_index());
        }
        relocalizing = true;
        ++relocalization_frames;
        model_maps = raycast_level(volume, sensor, camera_to_world,
                                   app_options, 0);
        spdlog::warn(
            "Frame {} tracking rejected: status={} correspondences={} "
            "mean_distance={} min_eigenvalue={} condition={} "
            "update_translation={} update_rotation={}",
            sensor.current_frame_index(), failure_name(tracking.result.error()),
            tracking.diagnostics.correspondences,
            tracking.diagnostics.mean_point_distance,
            tracking.diagnostics.min_system_eigenvalue,
            tracking.diagnostics.condition_number,
            tracking.diagnostics.update_translation,
            tracking.diagnostics.update_rotation);
        if (app_options.interactive_relocalization) {
          spdlog::warn(
              "Relocalization paused. Align the live sensor with the last "
              "model prediction and press Enter.");
          std::cin.get();
        }
        ++processed_frames;
        continue;
      }

      if (relocalizing) {
        spdlog::info(
            "Relocalized at frame {} after {} frame(s): correspondences={} "
            "mean_distance={}",
            sensor.current_frame_index(), relocalization_frames,
            tracking.diagnostics.correspondences,
            tracking.diagnostics.mean_point_distance);
        relocalizing = false;
        relocalization_frames = 0;
      }

      // current_camera_pose = tracked_pose
      camera_to_world = tracked_pose;

      // integrate current depth frame into volume using new pose
      volume.integrate_depth_image(
          sensor.depth_image(), sensor.depth_intrinsics(),
          camera_to_world.inverse(), tsdf_options, &sensor.color_image(),
          &live_pyramid.front().maps.normals);

      // raycast volume and get updated surface model maps
      model_maps = raycast_level(volume, sensor, camera_to_world,
                                 app_options, 0);
      write_outputs(app_options, model_maps, sensor.current_frame_index());

      spdlog::info(
          "Frame {} integrated: correspondences={} mean_distance={} "
          "observed_voxels={}",
          sensor.current_frame_index(), tracking.diagnostics.correspondences,
          tracking.diagnostics.mean_point_distance,
          volume.observed_voxel_count());
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
