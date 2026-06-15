#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_PROJECTIVE_ICP_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_PROJECTIVE_ICP_HPP

#include <cstddef>
#include <vector>

#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/volume.hpp>

namespace kinectfusion {

struct ProjectiveIcpOptions {
  unsigned int iterations{10};
  std::size_t min_correspondences{64};
  float max_point_distance{0.05F};
  float min_normal_dot{0.8F};
  float min_update_translation{1.0e-5F};
  float min_update_rotation{1.0e-5F};
  float max_update_translation{0.15F};
  float max_update_rotation{0.35F};
  float min_system_eigenvalue{1.0e-6F};
  float max_condition_number{1.0e6F};
};

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
  explicit ProjectiveIcpTracker(ProjectiveIcpOptions options = {});

  void set_iterations(unsigned int iterations);
  void set_min_correspondences(std::size_t min_correspondences);
  void set_max_point_distance(float max_point_distance);
  void set_min_normal_dot(float min_normal_dot);

  [[nodiscard]] ProjectiveIcpResult estimate_pose(
      const VertexNormalMaps& live_maps,
      const SurfaceMaps& model_maps,
      const CameraIntrinsics& model_intrinsics,
      const Eigen::Matrix4f& model_camera_to_world,
      Eigen::Matrix4f initial_camera_to_world) const;

 private:
  // Accumulated linearized point-to-plane normal equations for one ICP
  // iteration. Each accepted projective correspondence contributes a row
  // a = [source x n; n] (source = live vertex after the current pose estimate,
  // n = model normal) with right-hand side a * (n . (target - source)). The
  // 6x6 normal_matrix is solved directly for the increment and reused for the
  // degeneracy / conditioning check.
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

  ProjectiveIcpOptions options_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_PROJECTIVE_ICP_HPP */
