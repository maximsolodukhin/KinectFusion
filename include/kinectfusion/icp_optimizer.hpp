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
  std::size_t min_correspondences{64};
  float max_point_distance{0.05F};
  // cos(15 degrees); reject correspondences whose normals disagree by more.
  float min_normal_dot{0.9659258F};
  float min_update_translation{1.0e-5F};
  float min_update_rotation{1.0e-5F};
  float max_update_translation{0.15F};
  float max_update_rotation{0.35F};
  float min_system_eigenvalue{1.0e-6F};
  float max_condition_number{1.0e6F};
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
    Eigen::Matrix<float, 6, 6> normal_matrix{
        Eigen::Matrix<float, 6, 6>::Zero()};
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
