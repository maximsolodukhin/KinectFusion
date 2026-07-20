#include "reconstruction.hpp"

#include <spdlog/spdlog.h>

// Eigen/Core declares MatrixBase::inverse(); Eigen/LU defines it.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>  // NOLINT(misc-include-cleaner)
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/marching_cubes.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <string>
#include <system_error>
#include <utility>

#include "app_options.hpp"
#include "frame_output.hpp"
#include "logging.hpp"

namespace app {
namespace {

class IcpConsumer final : public kinectfusion::TrackingSurfaceConsumer {
 public:
  IcpConsumer(const kinectfusion::ProjectiveIcpTracker& tracker,
              const kinectfusion::CameraIntrinsics& model_intrinsics,
              Eigen::Matrix4f model_pose, unsigned int iterations)
      : tracker_(tracker),
        model_intrinsics_(model_intrinsics),
        model_pose_(std::move(model_pose)),
        iterations_(iterations) {}

  void consume(const kinectfusion::HostTrackingSurfaces& surfaces) override {
    estimate(surfaces);
  }
  void consume(const kinectfusion::DeviceTrackingSurfaces& surfaces) override {
    estimate(surfaces);
  }

  [[nodiscard]] const kinectfusion::IcpOutcome& outcome() const {
    return outcome_;
  }

 private:
  template <kinectfusion::MemorySpace Space>
  void estimate(const kinectfusion::TrackingSurfaces<Space>& surfaces) {
    outcome_ = tracker_.estimate_pose(surfaces, model_intrinsics_, model_pose_,
                                      iterations_);
  }

  const kinectfusion::ProjectiveIcpTracker& tracker_;
  kinectfusion::CameraIntrinsics model_intrinsics_;
  Eigen::Matrix4f model_pose_;
  unsigned int iterations_;
  kinectfusion::IcpOutcome outcome_;
};

}  // namespace

Reconstruction::Reconstruction(AppOptions options)
    : options_(std::move(options)),
      frame_output_(options_),
      tracker_(options_.icp_options()),
      pipelines_(
          kinectfusion::PipelineSet::create(options_.pipeline_set_config())) {
  auto creation = kinectfusion::PyramidSource::create(pipelines_.common_space(),
                                                      options_.depth_options());
  if (!creation.fallback_reason.empty()) {
    log_warn("Depth pyramid: {}", creation.fallback_reason);
  }
  pyramid_source_ = std::move(creation.source);
}

int Reconstruction::run() {
  if (!initialize()) {
    return EXIT_FAILURE;
  }

  const auto loop_start = std::chrono::steady_clock::now();
  while (options_.max_frames < 0 || processed_frames_ < options_.max_frames) {
    log_info("Loading next frame");
    if (!sensor_.process_next_frame()) {
      break;
    }
    process_frame();
  }
  const std::chrono::duration<double> loop_seconds =
      std::chrono::steady_clock::now() - loop_start;

  write_trajectory();
  if (options_.write_mesh) {
    write_meshes();
  }
  log_info("Finished reconstruction: processed_frames={} observed_voxels={}",
           processed_frames_, pipelines_.reference().observed_voxel_count());
  if (processed_frames_ > 0 && loop_seconds.count() > 0.0) {
    // Kept as cout + format because it's a single most important summary line
    // and I want it even with disabled logging.
    std::cout << std::format(
        "Frame loop: {} frames in {:.2f} s ({:.1f} fps)\n", processed_frames_,
        loop_seconds.count(),
        static_cast<double>(processed_frames_) / loop_seconds.count());
  }

  return EXIT_SUCCESS;
}

bool Reconstruction::initialize() {
  log_pipelines();
  log_info("Opening dataset: {}", options_.dataset_dir.string());
  if (!sensor_.init(options_.dataset_dir, options_.preload_frames)) {
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
  const auto initial_levels = build_pyramid();
  log_info("Frame {}: built {} pyramid level(s)", sensor_.current_frame_index(),
           initial_levels);

  log_info(
      "Frame {}: integrating first depth frame into TSDF volume ({}^3 "
      "voxels)",
      sensor_.current_frame_index(), options_.volume_resolution);
  integrate_frame();

  log_info("Frame {}: raycasting initialized model",
           sensor_.current_frame_index());
  render_model_outputs();

  log_info("Initialized reconstruction from frame 0: observed voxels={}",
           pipelines_.reference().observed_voxel_count());
  trajectory_.emplace_back(sensor_.current_timestamp(), camera_to_world_);

  return true;
}

void Reconstruction::process_frame() {
  log_frame_loaded();
  log_info("Frame {}: building live surface pyramid",
           sensor_.current_frame_index());
  const auto levels = build_pyramid();
  if (levels == 0) {
    log_warn("Frame {} produced no depth pyramid",
             sensor_.current_frame_index());
    return;
  }
  log_info("Frame {}: built {} live pyramid level(s)",
           sensor_.current_frame_index(), levels);

  const auto tracking = track_pose(levels);
  if (tracking.result) {
    integrate_tracked_frame(tracking);
  } else {
    relocalize(tracking);
  }
  trajectory_.emplace_back(sensor_.current_timestamp(), camera_to_world_);
  ++processed_frames_;
}

kinectfusion::IcpOutcome Reconstruction::track_pose(std::size_t levels) {
  kinectfusion::IcpOutcome tracking;
  tracking.pose = camera_to_world_;
  auto tracked_pose = camera_to_world_;

  log_info("Frame {}: tracking pose with projective ICP",
           sensor_.current_frame_index());

  for (std::size_t level_index = levels; level_index-- > 0;) {
    const auto level = static_cast<unsigned int>(level_index);
    log_info("Frame {} level {}: raycasting model prediction",
             sensor_.current_frame_index(), level);

    const auto camera =
        AppOptions::raycast_camera(sensor_, tracked_pose, level);
    const auto pyramid_level = pyramid_source_->level(level_index);
    const unsigned int iterations = options_.icp_iterations_for_level(level);

    log_info("Frame {} level {}: running ICP (iterations={})",
             sensor_.current_frame_index(), level, iterations);

    IcpConsumer consumer{tracker_, pyramid_level.intrinsics, tracked_pose,
                         iterations};
    pipelines_.track(camera, pyramid_level, consumer);
    tracking = consumer.outcome();

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
  integrate_frame();

  log_info("Frame {}: raycasting updated model", sensor_.current_frame_index());
  render_model_outputs();

  log_info("Frame {} integrated: {} observed_voxels={}",
           sensor_.current_frame_index(), tracking.diagnostics,
           pipelines_.reference().observed_voxel_count());
}

void Reconstruction::integrate_frame() {
  const kinectfusion::DepthFrame frame{
      .depth = &sensor_.depth_image(),
      .color = &sensor_.color_image(),
      .normals = pyramid_source_->host_normals(),
      .intrinsics = sensor_.depth_intrinsics(),
      .world_to_camera = camera_to_world_.inverse()};
  pipelines_.integrate(frame, pyramid_source_->device_frame(frame));
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

std::size_t Reconstruction::build_pyramid() {
  return pyramid_source_->build(sensor_.depth_image(),
                                sensor_.depth_intrinsics());
}

void Reconstruction::render_model_outputs() {
  const int frame_index = sensor_.current_frame_index();
  if (!frame_output_.writes_frames() &&
      !pipelines_.should_compare(frame_index)) {
    return;
  }
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
  for (auto& output : outputs) {
    const bool is_reference = output.name == reference_name;
    frame_output_.write_frame(std::move(output.maps), frame_index,
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

// TUM trajectory format (timestamp tx ty tz qx qy qz qw), evaluated against
// the dataset groundtruth by scripts/evaluate_ate.py.
void Reconstruction::write_trajectory() const {
  const auto path = options_.output_dir / "trajectory.txt";
  std::error_code create_error;
  std::filesystem::create_directories(options_.output_dir, create_error);
  std::ofstream file{path};
  if (!file) {
    log_warn("Could not write trajectory to {}", path.string());
    return;
  }
  constexpr int kTimestampPrecision = 9;
  file << std::setprecision(kTimestampPrecision);
  for (const auto& [timestamp, pose] : trajectory_) {
    const Eigen::Quaternionf rotation{Eigen::Matrix3f{pose.block<3, 3>(0, 0)}};
    file << std::fixed << timestamp << ' ' << pose(0, 3) << ' ' << pose(1, 3)
         << ' ' << pose(2, 3) << ' ' << rotation.x() << ' ' << rotation.y()
         << ' ' << rotation.z() << ' ' << rotation.w() << '\n';
  }
  log_info("Wrote {} trajectory poses to {}", trajectory_.size(),
           path.string());
}

void Reconstruction::write_meshes() const {
  const std::string reference_name = pipelines_.reference().name();
  for (const auto& member : pipelines_.members()) {
    const auto mesh = member.pipeline->extract_mesh(options_.mesh_min_weight);
    const bool is_reference = member.pipeline->name() == reference_name;
    frame_output_.write_mesh(mesh, is_reference ? "" : member.pipeline->name());
    log_info("Pipeline '{}' mesh: {} vertices, {} triangles",
             member.pipeline->name(), mesh.positions.size(),
             mesh.triangles.size());
  }
}

void Reconstruction::log_frame_loaded() const {
  log_info("Frame {} loaded: depth={}x{} color={}x{}",
           sensor_.current_frame_index(), sensor_.depth_image().width(),
           sensor_.depth_image().height(), sensor_.color_image().width(),
           sensor_.color_image().height());
}

}  // namespace app
