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
#include <numeric>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "app_options.hpp"
#include "frame_output.hpp"
#include "logging.hpp"
#include "trajectory_eval.hpp"

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
      set_config_(options_.pipeline_set_config()),
      pipelines_(kinectfusion::PipelineSet::create(set_config_)),
      tracks_(make_tracks(set_config_)) {
  auto creation = kinectfusion::PyramidSource::create(pipelines_.common_space(),
                                                      options_.depth_options());
  if (!creation.fallback_reason.empty()) {
    log_warn("Depth pyramid: {}", creation.fallback_reason);
  }
  pyramid_source_ = std::move(creation.source);
}

std::vector<Reconstruction::Track> Reconstruction::make_tracks(
    const kinectfusion::PipelineSetConfig& config) const {
  std::vector<Track> tracks;
  tracks.reserve(config.pipelines.size());
  for (const kinectfusion::PipelineConfig& pipeline : config.pipelines) {
    kinectfusion::ProjectiveIcpOptions icp = options_.icp_options();
    icp.damping = pipeline.icp_damping;
    icp.schedule = pipeline.icp_schedule;
    icp.adaptive_damping = pipeline.icp_adaptive_damping;
    Track track;
    track.tracker = kinectfusion::ProjectiveIcpTracker{icp};
    tracks.push_back(std::move(track));
  }
  return tracks;
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

  write_trajectories();
  report_ate();
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
  std::vector<std::size_t> all_members(tracks_.size());
  std::ranges::iota(all_members, std::size_t{0});
  integrate_members(all_members);

  log_info("Frame {}: raycasting initialized model",
           sensor_.current_frame_index());
  render_model_outputs();

  log_info("Initialized reconstruction from frame 0: observed voxels={}",
           pipelines_.reference().observed_voxel_count());
  record_trajectory();

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

  // Each pipeline tracks against its own model, then only the pipelines that
  // tracked integrate this frame (a rejected one keeps its last good pose).
  std::vector<std::size_t> tracked;
  tracked.reserve(tracks_.size());
  for (std::size_t member = 0; member < tracks_.size(); ++member) {
    const auto tracking = track_pose(member, levels);
    tracks_.at(member).camera_to_world = tracking.pose;
    if (tracking.result) {
      finish_relocalization(member, tracking);
      tracked.push_back(member);
    } else {
      relocalize(member, tracking);
    }
  }

  integrate_members(tracked);
  render_model_outputs();
  record_trajectory();
  ++processed_frames_;
}

kinectfusion::IcpOutcome Reconstruction::track_pose(std::size_t member,
                                                    std::size_t levels) {
  const Track& track = tracks_.at(member);
  kinectfusion::IcpOutcome tracking;
  tracking.pose = track.camera_to_world;
  auto tracked_pose = track.camera_to_world;

  log_info("Frame {} pipeline {}: tracking pose with projective ICP",
           sensor_.current_frame_index(), member);

  for (std::size_t level_index = levels; level_index-- > 0;) {
    const auto level = static_cast<unsigned int>(level_index);
    const auto camera =
        AppOptions::raycast_camera(sensor_, tracked_pose, level);
    const auto pyramid_level = pyramid_source_->level(level_index);
    const unsigned int iterations = options_.icp_iterations_for_level(level);

    IcpConsumer consumer{track.tracker, pyramid_level.intrinsics, tracked_pose,
                         iterations};
    pipelines_.track_member(member, camera, pyramid_level, consumer);
    tracking = consumer.outcome();

    log_info("Frame {} pipeline {} level {}: ICP result={} {}",
             sensor_.current_frame_index(), member, level,
             tracking.result ? "accepted" : "rejected", tracking.diagnostics);

    tracked_pose = tracking.pose;
    if (!tracking.result) {
      break;
    }
  }
  return tracking;
}

void Reconstruction::finish_relocalization(
    std::size_t member, const kinectfusion::IcpOutcome& tracking) {
  Track& track = tracks_.at(member);
  if (!track.relocalizing) {
    return;
  }
  log_info("Pipeline {} relocalized at frame {} after {} frame(s): {}", member,
           sensor_.current_frame_index(), track.relocalization_frames,
           tracking.diagnostics);
  track.relocalizing = false;
  track.relocalization_frames = 0;
}

void Reconstruction::integrate_members(
    const std::vector<std::size_t>& members) {
  kinectfusion::DepthFrame frame{.depth = &sensor_.depth_image(),
                                 .color = &sensor_.color_image(),
                                 .normals = pyramid_source_->host_normals(),
                                 .intrinsics = sensor_.depth_intrinsics()};
  const kinectfusion::DeviceDepthFrame* upload =
      pyramid_source_->device_frame(frame);
  for (const std::size_t member : members) {
    frame.world_to_camera = tracks_.at(member).camera_to_world.inverse();
    pipelines_.integrate_member(member, frame, upload);
  }
}

void Reconstruction::relocalize(std::size_t member,
                                const kinectfusion::IcpOutcome& tracking) {
  Track& track = tracks_.at(member);
  if (!track.relocalizing) {
    log_warn("Pipeline {} entering relocalization mode at frame {}", member,
             sensor_.current_frame_index());
  }
  track.relocalizing = true;
  ++track.relocalization_frames;
  log_warn("Frame {} pipeline {} tracking rejected: status={} {}",
           sensor_.current_frame_index(), member, tracking.result.error(),
           tracking.diagnostics);
  if (options_.interactive_relocalization) {
    std::cerr << "Relocalization paused. Align the live sensor with the "
                 "last model prediction and press Enter.\n";
    std::cin.get();
  }
}

void Reconstruction::record_trajectory() {
  const double timestamp = sensor_.current_timestamp();
  for (Track& track : tracks_) {
    track.trajectory.emplace_back(timestamp, track.camera_to_world);
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
  // Every pipeline is rendered from the reference pose so the surface
  // comparison and the written frames share one viewpoint.
  const auto camera =
      AppOptions::raycast_camera(sensor_, reference_track().camera_to_world, 0);

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
// the dataset groundtruth by scripts/evaluate_ate.py. The reference pipeline
// writes output_dir/trajectory.txt; the others write it under their own
// subdirectory, mirroring how the raycasts are laid out.
void Reconstruction::write_trajectories() const {
  const auto& members = pipelines_.members();
  for (std::size_t member = 0; member < tracks_.size(); ++member) {
    const bool is_reference = member == reference_index();
    const auto directory =
        is_reference
            ? options_.output_dir
            : options_.output_dir / members.at(member).pipeline->name();
    const auto path = directory / "trajectory.txt";

    std::error_code create_error;
    std::filesystem::create_directories(directory, create_error);
    std::ofstream file{path};
    if (!file) {
      log_warn("Could not write trajectory to {}", path.string());
      continue;
    }
    constexpr int kTimestampPrecision = 9;
    file << std::setprecision(kTimestampPrecision);
    for (const auto& [timestamp, pose] : tracks_.at(member).trajectory) {
      const Eigen::Quaternionf rotation{
          Eigen::Matrix3f{pose.block<3, 3>(0, 0)}};
      file << std::fixed << timestamp << ' ' << pose(0, 3) << ' ' << pose(1, 3)
           << ' ' << pose(2, 3) << ' ' << rotation.x() << ' ' << rotation.y()
           << ' ' << rotation.z() << ' ' << rotation.w() << '\n';
    }
    log_info("Wrote {} trajectory poses to {}",
             tracks_.at(member).trajectory.size(), path.string());
  }
}

void Reconstruction::report_ate() const {
  const AteEvaluator evaluator{options_.dataset_dir / "groundtruth.txt"};
  if (!evaluator.has_groundtruth()) {
    log_info("No groundtruth.txt in {}; skipping ATE",
             options_.dataset_dir.string());
    return;
  }

  const auto& members = pipelines_.members();
  constexpr double kMetersToCm = 100.0;
  for (std::size_t member = 0; member < tracks_.size(); ++member) {
    const std::string& name = members.at(member).pipeline->name();
    const auto stats = evaluator.evaluate(tracks_.at(member).trajectory);
    if (!stats) {
      std::cout << std::format("ATE {}: too few groundtruth associations\n",
                               name);
      continue;
    }
    // cout, not the logger, so the ablation summary shows without logging.
    std::cout << std::format(
        "ATE {}: pairs={} rmse={:.3f} cm mean={:.3f} cm median={:.3f} cm "
        "max={:.3f} cm\n",
        name, stats->pairs, stats->rmse * kMetersToCm,
        stats->mean * kMetersToCm, stats->median * kMetersToCm,
        stats->max_error * kMetersToCm);
  }
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
