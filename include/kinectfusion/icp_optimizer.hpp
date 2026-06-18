#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>

#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {

enum class ProjectiveIcpStatus {
  converged,
  too_few_correspondences,
  unconstrained_system,
  solve_failed,
  update_too_large,
};

struct ProjectiveIcpResult {
  Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
  bool converged{false};
  ProjectiveIcpStatus status{ProjectiveIcpStatus::too_few_correspondences};
  std::size_t correspondences{};
  float mean_point_distance{};
  float min_system_eigenvalue{};
  float condition_number{};
  float update_translation{};
  float update_rotation{};
};

class ProjectiveIcpTracker {
 public:
  void set_iterations(unsigned int v) { iterations_ = v; }
  void set_min_correspondences(std::size_t v) { min_correspondences_ = v; }
  void set_max_point_distance(float v) { max_point_distance_ = v; }
  void set_min_normal_dot(float v) { min_normal_dot_ = v; }
  void set_min_update_translation(float v) { min_update_translation_ = v; }
  void set_min_update_rotation(float v) { min_update_rotation_ = v; }
  void set_max_update_translation(float v) { max_update_translation_ = v; }
  void set_max_update_rotation(float v) { max_update_rotation_ = v; }
  void set_min_system_eigenvalue(float v) { min_system_eigenvalue_ = v; }
  void set_max_condition_number(float v) { max_condition_number_ = v; }

  [[nodiscard]] ProjectiveIcpResult estimate_pose(
      const VertexNormalMaps& live_maps,
      const SurfaceMaps& model_maps,
      const CameraIntrinsics& model_intrinsics,
      const Eigen::Matrix4f& model_camera_to_world,
      Eigen::Matrix4f initial_camera_to_world) const;

 private:
  struct CorrespondenceSet {
    std::size_t count{};
    Eigen::Matrix<float, 6, 6> normal_matrix{Eigen::Matrix<float, 6, 6>::Zero()};
    Eigen::Matrix<float, 6, 1> normal_rhs{Eigen::Matrix<float, 6, 1>::Zero()};
    float distance_sum{};
  };

  struct SystemStability {
    bool stable{};
    float min_eigenvalue{};
    float condition_number{};
  };

  struct Increment {
    Eigen::Matrix4f transform{Eigen::Matrix4f::Identity()};
    float update_translation{};
    float update_rotation{};
    bool solved{};
  };

  [[nodiscard]] CorrespondenceSet find_correspondences(
      const VertexNormalMaps& live_maps,
      const SurfaceMaps& model_maps,
      const CameraIntrinsics& model_intrinsics,
      const Eigen::Matrix4f& model_world_to_camera,
      const Eigen::Matrix4f& camera_to_world) const;

  [[nodiscard]] SystemStability check_system_stability(
      const CorrespondenceSet& correspondences) const;

  [[nodiscard]] Increment solve_increment(
      const CorrespondenceSet& correspondences) const;

  unsigned int iterations_{10};
  std::size_t min_correspondences_{64};
  float max_point_distance_{0.05F};
  // cos(15 degrees); reject correspondences whose normals disagree by more.
  float min_normal_dot_{0.9659258F};
  float min_update_translation_{1.0e-5F};
  float min_update_rotation_{1.0e-5F};
  float max_update_translation_{0.15F};
  float max_update_rotation_{0.35F};
  float min_system_eigenvalue_{1.0e-6F};
  float max_condition_number_{1.0e6F};
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP */
