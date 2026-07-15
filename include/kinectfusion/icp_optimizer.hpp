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

// One pose-tracking request: the image-aligned tracking surfaces (the model
// rendered at model_camera_to_world with model_intrinsics), the pose to start
// optimising from, and the iteration budget for this pyramid level.
template <MemorySpace Space = MemorySpace::kHost>
struct BasicTrackingRequest {
  using TransformMat = Eigen::Matrix4f;

  TrackingSurfaces<Space> surfaces{};
  CameraIntrinsics model_intrinsics{};
  TransformMat model_camera_to_world{TransformMat::Identity()};
  TransformMat initial_camera_to_world{TransformMat::Identity()};
  unsigned int iterations{1};
};

using TrackingRequest = BasicTrackingRequest<MemorySpace::kHost>;
using DeviceTrackingRequest = BasicTrackingRequest<MemorySpace::kDevice>;

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

// Gauss-Newton driver: a per-iteration correspondence sweep in the request's
// memory space builds the normal equations; the 6x6 stability check and solve
// stay on the host.
class ProjectiveIcpTracker {
 public:
  explicit ProjectiveIcpTracker(ProjectiveIcpOptions options = {})
      : options_(options) {}

  [[nodiscard]] IcpOutcome estimate_pose(const TrackingRequest& request) const;
  // Throws std::logic_error when built without CUDA support.
  [[nodiscard]] IcpOutcome estimate_pose(
      const DeviceTrackingRequest& request) const;

  // Tracks surfaces whose model was rendered at `pose`, starting the
  // optimisation from that same pose.
  template <MemorySpace Space>
  [[nodiscard]] IcpOutcome estimate_pose(
      const TrackingSurfaces<Space>& surfaces,
      const CameraIntrinsics& model_intrinsics, const Eigen::Matrix4f& pose,
      unsigned int iterations) const {
    return estimate_pose(
        BasicTrackingRequest<Space>{.surfaces = surfaces,
                                    .model_intrinsics = model_intrinsics,
                                    .model_camera_to_world = pose,
                                    .initial_camera_to_world = pose,
                                    .iterations = iterations});
  }

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

  // `find_equations` produces the normal equations for one iteration's
  // transforms in the request's memory space.
  template <MemorySpace Space, typename FindEquations>
  [[nodiscard]] IcpOutcome estimate_with(
      const BasicTrackingRequest<Space>& request,
      const FindEquations& find_equations) const;

  [[nodiscard]] CorrespondenceGates gates() const {
    return {.max_point_distance = options_.max_point_distance,
            .min_normal_dot = options_.min_normal_dot};
  }

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
