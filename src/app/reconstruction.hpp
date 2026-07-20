#ifndef KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP
#define KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP

#include <Eigen/Core>
#include <cstddef>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "app_options.hpp"
#include "frame_output.hpp"

namespace app {

class Reconstruction {
 public:
  explicit Reconstruction(AppOptions options);

  // Runs the reconstruction to completion. Returns an EXIT_SUCCESS/EXIT_FAILURE
  // status suitable for returning from main().
  [[nodiscard]] int run();

 private:
  // One independently tracked pipeline: its own tracker, pose, and trajectory.
  // The set compares these against each other, so each ablates its own ICP.
  struct Track {
    kinectfusion::ProjectiveIcpTracker tracker;
    Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};
    // (TUM timestamp, camera_to_world) per processed frame, written for
    // groundtruth ATE/RPE evaluation.
    std::vector<std::pair<double, Eigen::Matrix4f>> trajectory;
    bool relocalizing{false};
    int relocalization_frames{0};
  };

  [[nodiscard]] std::vector<Track> make_tracks(
      const kinectfusion::PipelineSetConfig& config) const;

  [[nodiscard]] bool initialize();
  void process_frame();
  [[nodiscard]] kinectfusion::IcpOutcome track_pose(std::size_t member,
                                                    std::size_t levels);
  void finish_relocalization(std::size_t member,
                             const kinectfusion::IcpOutcome& tracking);
  void relocalize(std::size_t member, const kinectfusion::IcpOutcome& tracking);
  // Rebuilds the pyramid for the current sensor frame; returns level count.
  [[nodiscard]] std::size_t build_pyramid();
  // Fuses the current sensor frame into each listed member at its own tracked
  // pose, sharing one device upload of the pose-independent frame data.
  void integrate_members(const std::vector<std::size_t>& members);
  void record_trajectory();
  void render_model_outputs();
  void log_pipelines() const;
  void write_trajectories() const;
  // Prints per-pipeline ATE against the dataset groundtruth, if present.
  void report_ate() const;
  // Extracts and writes one mesh.ply per pipeline (marching cubes).
  void write_meshes() const;
  void log_frame_loaded() const;

  [[nodiscard]] std::size_t reference_index() const {
    return pipelines_.reference_index();
  }
  [[nodiscard]] Track& reference_track() {
    return tracks_.at(reference_index());
  }
  [[nodiscard]] const Track& reference_track() const {
    return tracks_.at(reference_index());
  }

  AppOptions options_;
  FrameOutput frame_output_;
  kinectfusion::VirtualSensor sensor_;
  kinectfusion::PipelineSetConfig set_config_;
  std::unique_ptr<kinectfusion::PyramidSource> pyramid_source_;
  kinectfusion::PipelineSet pipelines_;
  // Parallel to pipelines_.members().
  std::vector<Track> tracks_;
  int processed_frames_{1};
};

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP */
