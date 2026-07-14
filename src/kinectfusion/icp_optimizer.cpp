#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <expected>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>

namespace kinectfusion {
namespace {

// Below this rotation magnitude the incremental rotation is treated as the
// identity to keep the axis direction well-defined when normalising by angle.
constexpr float kMinimumRotationAngle = 1.0e-12F;

[[nodiscard]] bool sizes_match(const ConstHostVertexNormalMapsView& maps) {
  return maps.vertices.width == maps.normals.width &&
         maps.vertices.height == maps.normals.height;
}

}  // namespace

IcpOutcome ProjectiveIcpTracker::estimate_pose(
    const TrackingRequest& request) const {
  IcpOutcome outcome{.pose = request.initial_camera_to_world};

  // Vertex and normal maps are produced together (by the depth pipeline and
  // the raycaster), so a size mismatch is a caller error rather than a
  // tracking failure. The expected channel makes it impossible to ignore.
  if (!sizes_match(request.live) || !sizes_match(request.model)) {
    outcome.result = std::unexpected(IcpFailure::kInvalidInput);
    return outcome;
  }

  const Eigen::Matrix4f model_world_to_camera =
      request.model_camera_to_world.inverse();
  for (unsigned int iteration = 0; iteration < request.iterations;
       ++iteration) {
    const IcpIterationTransforms transforms =
        IcpIterationTransforms::from_poses(outcome.pose, model_world_to_camera);
    const auto equations = find_correspondences(request, transforms);
    outcome.diagnostics.correspondences = equations.count;
    if (equations.count > 0) {
      outcome.diagnostics.mean_point_distance =
          equations.distance_sum / static_cast<float>(equations.count);
    }
    if (equations.count < options_.min_correspondences) {
      outcome.result = std::unexpected(IcpFailure::kTooFewCorrespondences);
      return outcome;
    }

    const auto stability = check_system_stability(equations);
    outcome.diagnostics.min_system_eigenvalue = stability.min_eigenvalue;
    outcome.diagnostics.condition_number = stability.condition_number;
    if (!stability.stable) {
      outcome.result = std::unexpected(IcpFailure::kUnconstrainedSystem);
      return outcome;
    }

    const auto increment = solve_increment(equations);
    if (!increment.solved) {
      outcome.result = std::unexpected(IcpFailure::kSolveFailed);
      return outcome;
    }

    outcome.diagnostics.update_translation = increment.update_translation;
    outcome.diagnostics.update_rotation = increment.update_rotation;
    if (increment.update_translation > options_.max_update_translation ||
        increment.update_rotation > options_.max_update_rotation) {
      outcome.result = std::unexpected(IcpFailure::kUpdateTooLarge);
      return outcome;
    }

    outcome.pose = increment.transform * outcome.pose;

    if (increment.update_translation < options_.min_update_translation &&
        increment.update_rotation < options_.min_update_rotation) {
      outcome.result = IcpSuccess{
          Converged{.update_translation = increment.update_translation,
                    .update_rotation = increment.update_rotation}};
      return outcome;
    }
  }

  // Every completed iteration was stable and well-constrained (otherwise we'd
  // have returned early), but the budget ran out before reaching tolerance.
  outcome.result = IcpSuccess{MaxIterations{}};
  return outcome;
}

IcpNormalEquations ProjectiveIcpTracker::find_correspondences(
    const TrackingRequest& request,
    const IcpIterationTransforms& transforms) const {
  IcpNormalEquations equations;
  const CorrespondenceSearch<MemorySpace::kHost> search{
      request.live, request.model, request.model_intrinsics, transforms,
      CorrespondenceGates{.max_point_distance = options_.max_point_distance,
                          .min_normal_dot = options_.min_normal_dot}};
  const auto& live = request.live.vertices;
  for (const auto [x, y] : PixelIndices{live.width, live.height}) {
    if (const auto match = search.match(x, y)) {
      equations.accumulate(*match);
    }
  }
  return equations;
}

ProjectiveIcpTracker::SystemStability
ProjectiveIcpTracker::check_system_stability(
    const IcpNormalEquations& equations) const {
  if (equations.count < options_.min_correspondences) {
    return {};
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, kIcpDof, kIcpDof>>
      solver{equations.matrix()};
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return {};
  }

  const float min_eigenvalue = solver.eigenvalues().minCoeff();
  const float max_eigenvalue = solver.eigenvalues().maxCoeff();
  if (min_eigenvalue <= 0.0F || max_eigenvalue <= 0.0F) {
    return SystemStability{.stable = false,
                           .min_eigenvalue = min_eigenvalue,
                           .condition_number = 0.0F};
  }

  const float condition_number = max_eigenvalue / min_eigenvalue;
  return SystemStability{
      .stable = min_eigenvalue >= options_.min_system_eigenvalue &&
                condition_number <= options_.max_condition_number,
      .min_eigenvalue = min_eigenvalue,
      .condition_number = condition_number};
}

ProjectiveIcpTracker::Increment ProjectiveIcpTracker::solve_increment(
    const IcpNormalEquations& equations) const {
  if (equations.count < options_.min_correspondences) {
    return {};
  }

  const Eigen::Matrix<float, kIcpDof, 1> solution =
      equations.matrix().ldlt().solve(equations.rhs());
  if (!solution.allFinite()) {
    return {};
  }

  const Eigen::Vector3f angle_axis = solution.head<3>();
  const Eigen::Vector3f translation = solution.tail<3>();
  const float angle = angle_axis.norm();
  Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
  if (angle > kMinimumRotationAngle) {
    rotation = Eigen::AngleAxisf(angle, angle_axis / angle).toRotationMatrix();
  }

  return Increment{.transform = make_transform_matrix(rotation, translation),
                   .update_translation = translation.norm(),
                   .update_rotation = angle,
                   .solved = true};
}

}  // namespace kinectfusion
