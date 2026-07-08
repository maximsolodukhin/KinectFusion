#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <kinectfusion/depth_processing.hpp>
#include <optional>
#include <variant>

namespace kinectfusion {

// Degrees of freedom of the linearised point-plane ICP system (three rotation
// + three translation parameters).
inline constexpr int kIcpDof = 6;

inline constexpr std::size_t kDefaultMinIcpCorrespondences = 64;
inline constexpr float kDefaultMaxIcpPointDistanceMeters = 0.05F;
// cos(15 degrees); reject correspondences whose normals disagree by more.
inline constexpr float kDefaultMinIcpNormalDot = 0.9659258F;
inline constexpr float kDefaultMinIcpUpdateTranslationMeters = 1.0e-5F;
inline constexpr float kDefaultMinIcpUpdateRotationRadians = 1.0e-5F;
inline constexpr float kDefaultMaxIcpUpdateTranslationMeters = 0.15F;
inline constexpr float kDefaultMaxIcpUpdateRotationRadians = 0.35F;
inline constexpr float kDefaultMinIcpSystemEigenvalue = 1.0e-6F;
inline constexpr float kDefaultMaxIcpConditionNumber = 1.0e6F;

enum class IcpFailure : std::uint8_t {
  kInvalidInput,
  kTooFewCorrespondences,
  kUnconstrainedSystem,
  kSolveFailed,
  kUpdateTooLarge,
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
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::expected<IcpSuccess, IcpFailure> result{};
};

// One pose-tracking request: image-aligned live and model vertex/normal maps
// (the model rendered at model_camera_to_world with model_intrinsics), the
// pose to start optimising from, and the iteration budget for this pyramid
// level.
struct TrackingRequest {
  ConstHostVertexNormalMapsView live{};
  ConstHostVertexNormalMapsView model{};
  CameraIntrinsics model_intrinsics{};
  Eigen::Matrix4f model_camera_to_world{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f initial_camera_to_world{Eigen::Matrix4f::Identity()};
  unsigned int iterations{1};
};

// Run-wide ICP tuning. These are fixed for the whole reconstruction, so they
// are passed to the tracker once at construction. The per-call iteration budget
// is not here: it varies by pyramid level and lives in TrackingRequest.
struct ProjectiveIcpOptions {
  std::size_t min_correspondences{kDefaultMinIcpCorrespondences};
  float max_point_distance{kDefaultMaxIcpPointDistanceMeters};
  float min_normal_dot{kDefaultMinIcpNormalDot};
  float min_update_translation{kDefaultMinIcpUpdateTranslationMeters};
  float min_update_rotation{kDefaultMinIcpUpdateRotationRadians};
  float max_update_translation{kDefaultMaxIcpUpdateTranslationMeters};
  float max_update_rotation{kDefaultMaxIcpUpdateRotationRadians};
  float min_system_eigenvalue{kDefaultMinIcpSystemEigenvalue};
  float max_condition_number{kDefaultMaxIcpConditionNumber};
};

class ProjectiveIcpTracker {
 public:
  explicit ProjectiveIcpTracker(ProjectiveIcpOptions options = {})
      : options_(options) {}

  [[nodiscard]] IcpOutcome estimate_pose(const TrackingRequest& request) const;

 private:
  // Accumulated point-plane normal equations (J^T J and J^T r) plus the
  // correspondence statistics the diagnostics report.
  struct NormalEquations {
    std::size_t count{};
    Eigen::Matrix<float, kIcpDof, kIcpDof> matrix{
        Eigen::Matrix<float, kIcpDof, kIcpDof>::Zero()};
    Eigen::Matrix<float, kIcpDof, 1> rhs{
        Eigen::Matrix<float, kIcpDof, 1>::Zero()};
    float distance_sum{};
  };

  // One accepted live-to-model correspondence, linearised for the point-plane
  // system: `jacobian` is its 6-DOF row, `residual` the point-plane error.
  struct Correspondence {
    Eigen::Matrix<float, kIcpDof, 1> jacobian;
    float residual;
    float distance;
  };

  // Transforms shared by every pixel within one Gauss-Newton iteration.
  struct IterationTransforms {
    Eigen::Matrix4f model_world_to_camera{Eigen::Matrix4f::Identity()};
    Eigen::Matrix3f rotation{Eigen::Matrix3f::Identity()};
    Eigen::Vector3f translation{Eigen::Vector3f::Zero()};
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

  [[nodiscard]] NormalEquations find_correspondences(
      const TrackingRequest& request,
      const IterationTransforms& transforms) const;

  // Projects one live vertex into the model frame and gates the surface pair
  // it lands on; nullopt when the projection misses the model image, the
  // model pixel is unmeasured, or the distance/normal checks reject the pair.
  [[nodiscard]] std::optional<Correspondence> match_point(
      const Vec3f& live_vertex, const Vec3f& live_normal,
      const TrackingRequest& request,
      const IterationTransforms& transforms) const;

  [[nodiscard]] SystemStability check_system_stability(
      const NormalEquations& equations) const;

  [[nodiscard]] Increment solve_increment(
      const NormalEquations& equations) const;

  ProjectiveIcpOptions options_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP */
