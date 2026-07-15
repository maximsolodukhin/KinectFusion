#include "reconstruction.hpp"

#include <spdlog/spdlog.h>

// Eigen/Core declares MatrixBase::inverse(); Eigen/LU defines it.
#include <Eigen/LU>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/volume.hpp>
#include <utility>
#include <variant>

#include "app_options.hpp"
#include "frame_output.hpp"
#include "logging.hpp"

namespace app {

Reconstruction::Reconstruction(AppOptions options)
    : options_(std::move(options)),
      frame_output_(options_),
      tracker_(options_.icp_options()),
      depth_processor_(options_.depth_options()),
      pipelines_(
          kinectfusion::PipelineSet::create(options_.pipeline_set_config())) {}

int Reconstruction::run() {
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
           processed_frames_, pipelines_.reference().observed_voxel_count());

  return EXIT_SUCCESS;
}

bool Reconstruction::initialize() {
  log_pipelines();
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
  log_info("Frame {}: built {} pyramid level(s)", sensor_.current_frame_index(),
           initial_pyramid.size());
  const auto* initial_normals =
      initial_pyramid.empty() ? nullptr : &initial_pyramid.front().maps.normals;

  log_info(
      "Frame {}: integrating first depth frame into TSDF volume ({}^3 "
      "voxels)",
      sensor_.current_frame_index(), options_.volume_resolution);
  integrate_frame(initial_normals);

  log_info("Frame {}: raycasting initialized model",
           sensor_.current_frame_index());
  render_model_outputs();

  log_info("Initialized reconstruction from frame 0: observed voxels={}",
           pipelines_.reference().observed_voxel_count());

  return true;
}

void Reconstruction::process_frame() {
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

kinectfusion::IcpOutcome Reconstruction::track_pose(
    const SurfacePyramid& live_pyramid) {
  kinectfusion::IcpOutcome tracking;
  tracking.pose = camera_to_world_;
  auto tracked_pose = camera_to_world_;

  log_info("Frame {}: tracking pose with projective ICP",
           sensor_.current_frame_index());

  for (std::size_t level_index = live_pyramid.size(); level_index-- > 0;) {
    const auto level = static_cast<unsigned int>(level_index);
    log_info("Frame {} level {}: raycasting model prediction",
             sensor_.current_frame_index(), level);

    const auto camera =
        AppOptions::raycast_camera(sensor_, tracked_pose, level);
    const auto level_intrinsics = sensor_.depth_intrinsics().scaled(level);
    const unsigned int iterations = options_.icp_iterations_for_level(level);

    log_info("Frame {} level {}: running ICP (iterations={})",
             sensor_.current_frame_index(), level, iterations);

    const auto live_views = view(live_pyramid.at(level_index).maps);
    tracking = std::visit(
        [&](const auto& surfaces) {
          return tracker_.estimate_pose(surfaces, level_intrinsics,
                                        tracked_pose, iterations);
        },
        pipelines_.tracking_surfaces(camera, live_views));

    log_info("Frame {} level {}: ICP result={} {}",
             sensor_.current_frame_index(), level,
             tracking.result ? "accepted" : "rejected", tracking.diagnostics);

    tracked_pose = tracking.pose;
    if (!tracking.result) {
      break;
    }
  }
  return tracking;
}

void Reconstruction::integrate_tracked_frame(
    const SurfacePyramid& live_pyramid,
    const kinectfusion::IcpOutcome& tracking) {
  if (relocalizing_) {
    log_info("Relocalized at frame {} after {} frame(s): {}",
             sensor_.current_frame_index(), relocalization_frames_,
             tracking.diagnostics);
    relocalizing_ = false;
    relocalization_frames_ = 0;
  }

  camera_to_world_ = tracking.pose;

  log_info("Frame {}: integrating tracked depth frame into TSDF volume",
           sensor_.current_frame_index());
  integrate_frame(&live_pyramid.front().maps.normals);

  log_info("Frame {}: raycasting updated model", sensor_.current_frame_index());
  render_model_outputs();

  log_info("Frame {} integrated: {} observed_voxels={}",
           sensor_.current_frame_index(), tracking.diagnostics,
           pipelines_.reference().observed_voxel_count());
}

void Reconstruction::integrate_frame(
    const kinectfusion::image_proc::Vector3fImage* normals) {
  pipelines_.integrate({.depth = &sensor_.depth_image(),
                        .color = &sensor_.color_image(),
                        .normals = normals,
                        .intrinsics = sensor_.depth_intrinsics(),
                        .world_to_camera = camera_to_world_.inverse()});
}

void Reconstruction::relocalize(const kinectfusion::IcpOutcome& tracking) {
  if (!relocalizing_) {
    log_warn("Entering relocalization mode at frame {}",
             sensor_.current_frame_index());
  }
  relocalizing_ = true;
  ++relocalization_frames_;
  log_warn("Frame {} tracking rejected: status={} {}",
           sensor_.current_frame_index(), tracking.result.error(),
           tracking.diagnostics);
  if (options_.interactive_relocalization) {
    std::cerr << "Relocalization paused. Align the live sensor with the "
                 "last model prediction and press Enter.\n";
    std::cin.get();
  }
}

Reconstruction::SurfacePyramid Reconstruction::build_pyramid() const {
  return depth_processor_.build_surface_pyramid(sensor_.depth_image(),
                                                sensor_.depth_intrinsics());
}

void Reconstruction::render_model_outputs() {
  const int frame_index = sensor_.current_frame_index();
  const auto camera = AppOptions::raycast_camera(sensor_, camera_to_world_, 0);

  log_info("Frame {}: writing outputs to {}", frame_index,
           options_.output_dir.string());
  if (!pipelines_.should_compare(frame_index)) {
    frame_output_.write_frame(pipelines_.raycast_reference(camera),
                              frame_index);
    return;
  }

  auto outputs = pipelines_.raycast_all(camera);
  const auto comparisons = pipelines_.compare(outputs);
  for (const auto& comparison : comparisons) {
    log_info("Frame {} {}", frame_index, comparison);
  }
  frame_output_.append_ablation_stats(frame_index, comparisons);

  const std::string reference_name = pipelines_.reference().name();
  for (const auto& output : outputs) {
    const bool is_reference = output.name == reference_name;
    frame_output_.write_frame(output.maps, frame_index,
                              is_reference ? std::string{} : output.name);
  }
}

void Reconstruction::log_pipelines() const {
  const kinectfusion::Pipeline& reference = pipelines_.reference();
  for (const auto& member : pipelines_.members()) {
    const bool is_reference =
        pipelines_.size() > 1 && member.pipeline.get() == &reference;
    log_info("Pipeline '{}' ready{}", member.pipeline->name(),
             is_reference ? " (reference)" : "");
    if (!member.fallback_reason.empty()) {
      log_warn("Pipeline '{}': {}", member.pipeline->name(),
               member.fallback_reason);
    }
  }
}

void Reconstruction::log_frame_loaded() const {
  log_info("Frame {} loaded: depth={}x{} color={}x{}",
           sensor_.current_frame_index(), sensor_.depth_image().width(),
           sensor_.depth_image().height(), sensor_.color_image().width(),
           sensor_.color_image().height());
}

}  // namespace app
