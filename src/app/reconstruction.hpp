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
  [[nodiscard]] bool initialize();
  void process_frame();
  [[nodiscard]] kinectfusion::IcpOutcome track_pose(std::size_t levels);
  void integrate_tracked_frame(const kinectfusion::IcpOutcome& tracking);
  // Fuses the current sensor frame into every pipeline at the tracked pose.
  void integrate_frame();
  void relocalize(const kinectfusion::IcpOutcome& tracking);
  // Rebuilds the pyramid for the current sensor frame; returns level count.
  [[nodiscard]] std::size_t build_pyramid();
  void render_model_outputs();
  void log_pipelines() const;
  void write_trajectory() const;
  // Extracts and writes one mesh.ply per pipeline (marching cubes).
  void write_meshes() const;
  void log_frame_loaded() const;

  AppOptions options_;
  FrameOutput frame_output_;
  kinectfusion::VirtualSensor sensor_;
  kinectfusion::ProjectiveIcpTracker tracker_;
  std::unique_ptr<kinectfusion::PyramidSource> pyramid_source_;
  kinectfusion::PipelineSet pipelines_;
  Eigen::Matrix4f camera_to_world_{Eigen::Matrix4f::Identity()};
  // (TUM timestamp, camera_to_world) per processed frame, written as
  // trajectory.txt for groundtruth ATE/RPE evaluation.
  std::vector<std::pair<double, Eigen::Matrix4f>> trajectory_;
  int processed_frames_{1};
  bool relocalizing_{false};
  int relocalization_frames_{0};
};

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP */
