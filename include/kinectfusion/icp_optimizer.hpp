#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <variant>

namespace kinectfusion {

inline constexpr std::size_t kDefaultMinIcpCorrespondences = 64;
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
  using TransformMat = Eigen::Matrix4f;

  ConstHostVertexNormalMapsView live{};
  ConstHostVertexNormalMapsView model{};
  CameraIntrinsics model_intrinsics{};
  TransformMat model_camera_to_world{TransformMat::Identity()};
  TransformMat initial_camera_to_world{TransformMat::Identity()};
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

// Host driver
// Per-iteration correspondence sweep via CorrespondenceSearch<kHost>, then the
// 6x6 stability check and solve.
class ProjectiveIcpTracker {
 public:
  explicit ProjectiveIcpTracker(ProjectiveIcpOptions options = {})
      : options_(options) {}

  [[nodiscard]] IcpOutcome estimate_pose(const TrackingRequest& request) const;

 private:
  struct SystemStability {
    bool stable{};
    float min_eigenvalue{};
    float condition_number{};
  };

  struct Increment {
    // rigid body trasnform
    using TransformMat = Eigen::Matrix4f;
    TransformMat transform{TransformMat::Identity()};
    float update_translation{};
    float update_rotation{};
    bool solved{};
  };

  [[nodiscard]] IcpNormalEquations find_correspondences(
      const TrackingRequest& request,
      const IcpIterationTransforms& transforms) const;

  [[nodiscard]] SystemStability check_system_stability(
      const IcpNormalEquations& equations) const;

  [[nodiscard]] Increment solve_increment(
      const IcpNormalEquations& equations) const;

  ProjectiveIcpOptions options_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP */
