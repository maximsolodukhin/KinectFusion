#ifndef KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP
#define KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP

#include <Eigen/Core>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <kinectfusion/volume.hpp>

#include "app_options.hpp"

namespace app {

class Reconstruction {
 public:
  explicit Reconstruction(AppOptions options);

  // Runs the reconstruction to completion. Returns an EXIT_SUCCESS/EXIT_FAILURE
  // status suitable for returning from main().
  [[nodiscard]] int run();

 private:
  using SurfacePyramid = kinectfusion::SurfacePyramid;

  [[nodiscard]] bool initialize();
  void process_frame();
  [[nodiscard]] kinectfusion::IcpOutcome track_pose(
      const SurfacePyramid& live_pyramid) const;
  void integrate_tracked_frame(const SurfacePyramid& live_pyramid,
                               const kinectfusion::IcpOutcome& tracking);
  // Fuses the current sensor frame into the volume at the tracked pose.
  void integrate_frame(const kinectfusion::image_proc::Vector3fImage* normals);
  void relocalize(const kinectfusion::IcpOutcome& tracking);
  [[nodiscard]] SurfacePyramid build_pyramid() const;
  [[nodiscard]] kinectfusion::SurfaceMaps raycast_model(
      const Eigen::Matrix4f& camera_to_world, unsigned int level) const;
  void log_frame_loaded() const;

  AppOptions options_;
  kinectfusion::VirtualSensor sensor_;
  kinectfusion::Volume volume_;
  kinectfusion::ProjectiveIcpTracker tracker_;
  kinectfusion::DepthProcessor<kinectfusion::MemorySpace::kHost>
      depth_processor_;
  kinectfusion::TsdfIntegrationOptions tsdf_options_;
  Eigen::Matrix4f camera_to_world_{Eigen::Matrix4f::Identity()};
  kinectfusion::SurfaceMaps model_maps_;
  int processed_frames_{1};
  bool relocalizing_{false};
  int relocalization_frames_{0};
};

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_RECONSTRUCTION_HPP */
