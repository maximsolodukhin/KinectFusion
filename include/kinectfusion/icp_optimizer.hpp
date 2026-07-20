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

// Which CUDA-graph construction the device correspondence sweep uses. An
// ablation axis only: both build the same graph and produce the same result.
enum class IcpGraphBuild : std::uint8_t { kExplicit, kCaptured };

struct Converged {
  float update_translation{};
  float update_rotation{};
};

struct MaxIterations {};

using IcpSuccess = std::variant<Converged, MaxIterations>;

struct IcpDiagnostics {
  std::size_t correspondences{};
  float mean_point_distance{};
  float mean_squared_residual{};
  float min_system_eigenvalue{};
  float condition_number{};
  float update_translation{};
  float update_rotation{};
  float damping_lambda{};
  std::size_t rejected_steps{};
};

// The pose is always present, so it sits outside the success/error channel.
// On failure it is the last good pose.
struct IcpOutcome {
  Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
  IcpDiagnostics diagnostics{};
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::expected<IcpSuccess, IcpFailure> result{};
};

// One pose-tracking request for one pyramid level. The model surfaces were
// rendered at model_camera_to_world with model_intrinsics.
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

// Run-wide ICP tuning, fixed for the whole reconstruction. The iteration
// budget varies by pyramid level and lives in TrackingRequest.
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
  IcpGraphBuild device_graph_build{IcpGraphBuild::kExplicit};
  // Ablation: run the whole GN loop on the device with one sync per level.
  // The stability check runs once on the final system.
  bool device_solve{false};
  // Damping replaces the eigenvalue veto, so kUnconstrainedSystem is raised
  // only when mode is kNone.
  IcpDamping damping{.mode = IcpDampingMode::kNone};
  // Adapts lambda per trial and rolls back trials that raise the cost.
  bool adaptive_damping{false};
  IcpDampingSchedule schedule{};
};

// Damped Gauss-Newton / Levenberg-Marquardt driver. A per-iteration
// correspondence sweep builds the normal equations in the request's memory
// space. The stability check and the 6x6 solve stay on the host.
class ProjectiveIcpTracker {
 public:
  explicit ProjectiveIcpTracker(ProjectiveIcpOptions options = {})
      : options_(options),
        device_sweep_(make_device_sweep(options.device_graph_build)) {}

  [[nodiscard]] IcpOutcome estimate_pose(const TrackingRequest& request) const;
  // Throws std::logic_error when built without CUDA support.
  [[nodiscard]] IcpOutcome estimate_pose(
      const DeviceTrackingRequest& request) const;

  // `pose` is the model render pose and also the initial pose.
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
    float lambda{};
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

  [[nodiscard]] IcpOutcome estimate_device_loop(
      const DeviceTrackingRequest& request) const;

  [[nodiscard]] IcpNormalEquations find_correspondences(
      const TrackingRequest& request,
      const IcpIterationTransforms& transforms) const;

  [[nodiscard]] SystemStability check_system_stability(
      const IcpNormalEquations& equations) const;

  [[nodiscard]] Increment solve_increment(const IcpNormalEquations& equations,
                                          float lambda) const;

  [[nodiscard]] static Increment increment_from_solution(
      const Eigen::Matrix<float, kIcpDof, 1>& solution);

  [[nodiscard]] bool within_update_caps(const Increment& increment) const {
    return increment.update_translation <= options_.max_update_translation &&
           increment.update_rotation <= options_.max_update_rotation;
  }

  // Either graph-build variant of the device sweep, chosen by options.
  using DeviceSweep =
      std::variant<DeviceCorrespondenceSweep, CapturedCorrespondenceSweep>;

  [[nodiscard]] static DeviceSweep make_device_sweep(IcpGraphBuild build) {
    if (build == IcpGraphBuild::kCaptured) {
      return DeviceSweep{std::in_place_type<CapturedCorrespondenceSweep>};
    }
    return DeviceSweep{std::in_place_type<DeviceCorrespondenceSweep>};
  }

  ProjectiveIcpOptions options_;
  // Reduction scratch aka cache
  mutable DeviceSweep device_sweep_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_OPTIMIZER_HPP */
