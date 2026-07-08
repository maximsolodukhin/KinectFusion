#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <Eigen/LU>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <string>
#include <utility>
#include <vector>

#include "app_options.hpp"
#include "frame_output.hpp"

namespace {

template <typename... Args>
void log_info([[maybe_unused]] spdlog::format_string_t<Args...> fmt,
              [[maybe_unused]] Args&&... args) {
#ifdef KINECTFUSION_ENABLE_LOGGING
  spdlog::info(fmt, std::forward<Args>(args)...);
#endif
}

template <typename... Args>
void log_warn([[maybe_unused]] spdlog::format_string_t<Args...> fmt,
              [[maybe_unused]] Args&&... args) {
#ifdef KINECTFUSION_ENABLE_LOGGING
  spdlog::warn(fmt, std::forward<Args>(args)...);
#endif
}

void configure_logging() {
#ifdef KINECTFUSION_ENABLE_LOGGING
  auto logger = spdlog::stdout_color_mt("kinectfusion");
  logger->set_pattern("[%H:%M:%S] [%^%l%$] %v");
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);
#else
  auto logger = spdlog::stderr_color_mt("kinectfusion");
  logger->set_pattern("[%^%l%$] %v");
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::err);
  spdlog::flush_on(spdlog::level::err);
#endif
}

[[nodiscard]] constexpr std::string_view failure_name(
    kinectfusion::IcpFailure failure) {
  switch (failure) {
    case kinectfusion::IcpFailure::kInvalidInput:
      return "kInvalidInput";
    case kinectfusion::IcpFailure::kTooFewCorrespondences:
      return "kTooFewCorrespondences";
    case kinectfusion::IcpFailure::kUnconstrainedSystem:
      return "kUnconstrainedSystem";
    case kinectfusion::IcpFailure::kSolveFailed:
      return "kSolveFailed";
    case kinectfusion::IcpFailure::kUpdateTooLarge:
      return "kUpdateTooLarge";
  }
  return "unknown";
}

class Reconstruction {
 public:
  explicit Reconstruction(app::AppOptions options)
      : options_(std::move(options)),
        volume_(kinectfusion::Vector3s{options_.volume_resolution,
                                       options_.volume_resolution,
                                       options_.volume_resolution},
                options_.voxel_size, options_.volume_origin(),
                options_.truncation_distance),
        tracker_{options_.icp_options()},
        depth_processor_{options_.depth_options()},
        tsdf_options_(options_.tsdf_options()) {}

  [[nodiscard]] int run() {
    if (!initialize()) {
      return EXIT_FAILURE;
    }
    while (options_.max_frames < 0 || processed_frames_ < options_.max_frames) {
      log_info("Loading next frame");
      if (!sensor_.process_next_frame()) {
        break;
      }
      process_frame();
    }
    log_info("Finished reconstruction: processed_frames={} observed_voxels={}",
             processed_frames_, volume_.observed_voxel_count());
    return EXIT_SUCCESS;
  }

 private:
  using SurfacePyramid = std::vector<kinectfusion::DepthProcessingLevel>;

  [[nodiscard]] bool initialize() {
    log_info("Opening dataset: {}", options_.dataset_dir.string());
    if (!sensor_.init(options_.dataset_dir)) {
      spdlog::error("Failed to initialize dataset: {}",
                    options_.dataset_dir.string());
      return false;
    }
    log_info("Loading initial frame");
    if (!sensor_.process_next_frame()) {
      spdlog::error("Dataset contains no frames");
      return false;
    }
    log_frame_loaded();

    log_info("Frame {}: building surface pyramid (requested levels={})",
             sensor_.current_frame_index(), options_.depth_options().levels);
    const auto initial_pyramid = build_pyramid();
    log_info("Frame {}: built {} pyramid level(s)",
             sensor_.current_frame_index(), initial_pyramid.size());
    const auto* initial_normals = initial_pyramid.empty()
                                      ? nullptr
                                      : &initial_pyramid.front().maps.normals;

    log_info(
        "Frame {}: integrating first depth frame into TSDF volume ({}^3 "
        "voxels)",
        sensor_.current_frame_index(), options_.volume_resolution);
    volume_.integrate_depth_image(
        {.depth = &sensor_.depth_image(),
         .color = &sensor_.color_image(),
         .normals = initial_normals,
         .intrinsics = sensor_.depth_intrinsics(),
         .world_to_camera = camera_to_world_.inverse()},
        tsdf_options_);

    log_info("Frame {}: raycasting initialized model",
             sensor_.current_frame_index());
    model_maps_ = raycast_model(camera_to_world_, 0);
    log_info("Frame {}: writing outputs to {}", sensor_.current_frame_index(),
             options_.output_dir.string());
    write_outputs(options_, model_maps_, 0);

    log_info("Initialized reconstruction from frame 0: observed voxels={}",
             volume_.observed_voxel_count());
    return true;
  }

  void process_frame() {
    log_frame_loaded();
    log_info("Frame {}: building live surface pyramid",
             sensor_.current_frame_index());
    const auto live_pyramid = build_pyramid();
    if (live_pyramid.empty()) {
      log_warn("Frame {} produced no depth pyramid",
               sensor_.current_frame_index());
      return;
    }
    log_info("Frame {}: built {} live pyramid level(s)",
             sensor_.current_frame_index(), live_pyramid.size());

    const auto tracking = track_pose(live_pyramid);
    if (tracking.result) {
      integrate_tracked_frame(live_pyramid, tracking);
    } else {
      relocalize(tracking);
    }
    ++processed_frames_;
  }

  [[nodiscard]] kinectfusion::IcpOutcome track_pose(
      const SurfacePyramid& live_pyramid) const {
    kinectfusion::IcpOutcome tracking;
    tracking.pose = camera_to_world_;
    auto tracked_pose = camera_to_world_;

    log_info("Frame {}: tracking pose with projective ICP",
             sensor_.current_frame_index());
    for (std::size_t level_index = live_pyramid.size(); level_index-- > 0;) {
      const auto level = static_cast<unsigned int>(level_index);
      log_info("Frame {} level {}: raycasting model prediction",
               sensor_.current_frame_index(), level);
      const auto level_model_maps = raycast_model(tracked_pose, level);
      const auto level_intrinsics = sensor_.depth_intrinsics().scaled(level);
      log_info("Frame {} level {}: running ICP (iterations={})",
               sensor_.current_frame_index(), level,
               options_.icp_iterations_for_level(level));
      tracking = tracker_.estimate_pose(
          {.live = view(live_pyramid.at(level_index).maps),
           .model = {.vertices = level_model_maps.points.view(),
                     .normals = level_model_maps.normals.view()},
           .model_intrinsics = level_intrinsics,
           .model_camera_to_world = tracked_pose,
           .initial_camera_to_world = tracked_pose,
           .iterations = options_.icp_iterations_for_level(level)});
      log_info(
          "Frame {} level {}: ICP result={} correspondences={} "
          "mean_distance={}",
          sensor_.current_frame_index(), level,
          tracking.result ? "accepted" : "rejected",
          tracking.diagnostics.correspondences,
          tracking.diagnostics.mean_point_distance);
      tracked_pose = tracking.pose;
      if (!tracking.result) {
        break;
      }
    }
    return tracking;
  }

  void integrate_tracked_frame(const SurfacePyramid& live_pyramid,
                               const kinectfusion::IcpOutcome& tracking) {
    if (relocalizing_) {
      log_info(
          "Relocalized at frame {} after {} frame(s): correspondences={} "
          "mean_distance={}",
          sensor_.current_frame_index(), relocalization_frames_,
          tracking.diagnostics.correspondences,
          tracking.diagnostics.mean_point_distance);
      relocalizing_ = false;
      relocalization_frames_ = 0;
    }

    camera_to_world_ = tracking.pose;

    log_info("Frame {}: integrating tracked depth frame into TSDF volume",
             sensor_.current_frame_index());
    volume_.integrate_depth_image(
        {.depth = &sensor_.depth_image(),
         .color = &sensor_.color_image(),
         .normals = &live_pyramid.front().maps.normals,
         .intrinsics = sensor_.depth_intrinsics(),
         .world_to_camera = camera_to_world_.inverse()},
        tsdf_options_);

    log_info("Frame {}: raycasting updated model",
             sensor_.current_frame_index());
    model_maps_ = raycast_model(camera_to_world_, 0);
    log_info("Frame {}: writing outputs to {}", sensor_.current_frame_index(),
             options_.output_dir.string());
    write_outputs(options_, model_maps_, sensor_.current_frame_index());

    log_info(
        "Frame {} integrated: correspondences={} mean_distance={} "
        "observed_voxels={}",
        sensor_.current_frame_index(), tracking.diagnostics.correspondences,
        tracking.diagnostics.mean_point_distance,
        volume_.observed_voxel_count());
  }

  void relocalize(const kinectfusion::IcpOutcome& tracking) {
    if (!relocalizing_) {
      log_warn("Entering relocalization mode at frame {}",
               sensor_.current_frame_index());
    }
    relocalizing_ = true;
    ++relocalization_frames_;
    log_info("Frame {}: raycasting last accepted pose for relocalization",
             sensor_.current_frame_index());
    model_maps_ = raycast_model(camera_to_world_, 0);
    log_warn(
        "Frame {} tracking rejected: status={} correspondences={} "
        "mean_distance={} min_eigenvalue={} condition={} "
        "update_translation={} update_rotation={}",
        sensor_.current_frame_index(), failure_name(tracking.result.error()),
        tracking.diagnostics.correspondences,
        tracking.diagnostics.mean_point_distance,
        tracking.diagnostics.min_system_eigenvalue,
        tracking.diagnostics.condition_number,
        tracking.diagnostics.update_translation,
        tracking.diagnostics.update_rotation);
    if (options_.interactive_relocalization) {
      std::cerr << "Relocalization paused. Align the live sensor with the "
                   "last model prediction and press Enter.\n";
      std::cin.get();
    }
  }

  [[nodiscard]] SurfacePyramid build_pyramid() const {
    return depth_processor_.build_surface_pyramid(sensor_.depth_image(),
                                                  sensor_.depth_intrinsics());
  }

  [[nodiscard]] kinectfusion::SurfaceMaps raycast_model(
      const Eigen::Matrix4f& camera_to_world, unsigned int level) const {
    return volume_.raycast(
        options_.raycast_options(sensor_, camera_to_world, level));
  }

  void log_frame_loaded() const {
    log_info("Frame {} loaded: depth={}x{} color={}x{}",
             sensor_.current_frame_index(), sensor_.depth_image().width(),
             sensor_.depth_image().height(), sensor_.color_image().width(),
             sensor_.color_image().height());
  }

  app::AppOptions options_;
  kinectfusion::VirtualSensor sensor_;
  kinectfusion::Volume volume_;
  kinectfusion::ProjectiveIcpTracker tracker_;
  kinectfusion::DepthProcessor depth_processor_;
  kinectfusion::TsdfIntegrationOptions tsdf_options_;
  Eigen::Matrix4f camera_to_world_{Eigen::Matrix4f::Identity()};
  kinectfusion::SurfaceMaps model_maps_;
  int processed_frames_{1};
  bool relocalizing_{false};
  int relocalization_frames_{0};
};

}  // namespace

int main(int argc, const char** argv) {
  try {
    configure_logging();

    app::AppOptions app_options;

    CLI::App app{"CPU KinectFusion reconstruction"};
    app::configure_cli(app, app_options);

    CLI11_PARSE(app, argc, argv);

    app::validate_options(app_options);

    log_info(
        "Starting KinectFusion: dataset={} max_frames={} volume={}^3 "
        "voxel_size={} truncation_distance={}",
        app_options.dataset_dir.string(), app_options.max_frames,
        app_options.volume_resolution, app_options.voxel_size,
        app_options.truncation_distance);

    Reconstruction reconstruction{std::move(app_options)};
    return reconstruction.run();
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
