#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/volume.hpp>
#include <variant>

namespace kinectfusion {

// Degrees of freedom of the linearised point-plane ICP system (three rotation
// + three translation parameters).
inline constexpr int icp_dof = 6;

inline constexpr std::size_t default_min_icp_correspondences = 64;
inline constexpr float default_max_icp_point_distance_meters = 0.05F;
// cos(15 degrees); reject correspondences whose normals disagree by more.
inline constexpr float default_min_icp_normal_dot = 0.9659258F;
inline constexpr float default_min_icp_update_translation_meters = 1.0e-5F;
inline constexpr float default_min_icp_update_rotation_radians = 1.0e-5F;
inline constexpr float default_max_icp_update_translation_meters = 0.15F;
inline constexpr float default_max_icp_update_rotation_radians = 0.35F;
inline constexpr float default_min_icp_system_eigenvalue = 1.0e-6F;
inline constexpr float default_max_icp_condition_number = 1.0e6F;

enum class IcpFailure : std::uint8_t {
  invalid_input,
  too_few_correspondences,
  unconstrained_system,
  solve_failed,
  update_too_large,
};

struct Converged {
  float update_translation{};
  float update_rotation{};
};

struct MaxIterations {};

using IcpSuccess = std::variant<Converged, MaxIterations>;

struct IcpDiagnostics {
  std::size_t correspondences{};
  float mean_point_distance{};
  float min_system_eigenvalue{};
  float condition_number{};
  float update_translation{};
  float update_rotation{};
};

// The pose is always present (on failure it is the last good pose for the
// caller to fall back on this frame), so it sits outside the success/error
// channel, as do the always-available diagnostics.
struct IcpOutcome {
  Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
  IcpDiagnostics diagnostics{};
  std::expected<IcpSuccess, IcpFailure> result{};
};

// Run-wide ICP tuning. These are fixed for the whole reconstruction, so they
// are passed to the tracker once at construction. The per-call iteration budget
// is not here: it varies by pyramid level and is a parameter of estimate_pose.
struct ProjectiveIcpOptions {
  std::size_t min_correspondences{default_min_icp_correspondences};
  float max_point_distance{default_max_icp_point_distance_meters};
  float min_normal_dot{default_min_icp_normal_dot};
  float min_update_translation{default_min_icp_update_translation_meters};
  float min_update_rotation{default_min_icp_update_rotation_radians};
  float max_update_translation{default_max_icp_update_translation_meters};
  float max_update_rotation{default_max_icp_update_rotation_radians};
  float min_system_eigenvalue{default_min_icp_system_eigenvalue};
  float max_condition_number{default_max_icp_condition_number};
};

class ProjectiveIcpTracker {
 public:
  explicit ProjectiveIcpTracker(ProjectiveIcpOptions options = {})
      : options_(options) {}

  [[nodiscard]] IcpOutcome estimate_pose(
      unsigned int iterations, const VertexNormalMaps& live_maps,
      const SurfaceMaps& model_maps, const CameraIntrinsics& model_intrinsics,
      const Eigen::Matrix4f& model_camera_to_world,
      Eigen::Matrix4f initial_camera_to_world) const;

 private:
  struct CorrespondenceSet {
    std::size_t count{};
    Eigen::Matrix<float, icp_dof, icp_dof> normal_matrix{
        Eigen::Matrix<float, icp_dof, icp_dof>::Zero()};
    Eigen::Matrix<float, icp_dof, 1> normal_rhs{
        Eigen::Matrix<float, icp_dof, 1>::Zero()};
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
      const VertexNormalMaps& live_maps, const SurfaceMaps& model_maps,
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

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP */
