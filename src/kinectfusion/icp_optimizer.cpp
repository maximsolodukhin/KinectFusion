#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <expected>
#include <kinectfusion/icp_optimizer.hpp>
#include <optional>

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
    const IterationTransforms transforms{
        .model_world_to_camera = model_world_to_camera,
        .rotation = outcome.pose.block<3, 3>(0, 0),
        .translation = outcome.pose.block<3, 1>(0, 3)};
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

ProjectiveIcpTracker::NormalEquations
ProjectiveIcpTracker::find_correspondences(
    const TrackingRequest& request,
    const IterationTransforms& transforms) const {
  NormalEquations equations;
  const auto& live = request.live;
  for (std::size_t y = 0; y < live.vertices.height; ++y) {
    for (std::size_t x = 0; x < live.vertices.width; ++x) {
      const Vec3f& live_vertex = live.vertices.at(x, y);
      const Vec3f& live_normal = live.normals.at(x, y);
      if (!all_finite(live_vertex) || !all_finite(live_normal)) {
        continue;
      }
      const auto match =
          match_point(live_vertex, live_normal, request, transforms);
      if (!match) {
        continue;
      }
      equations.matrix += match->jacobian * match->jacobian.transpose();
      equations.rhs += match->jacobian * match->residual;
      equations.distance_sum += match->distance;
      ++equations.count;
    }
  }
  return equations;
}

std::optional<ProjectiveIcpTracker::Correspondence>
ProjectiveIcpTracker::match_point(const Vec3f& live_vertex,
                                  const Vec3f& live_normal,
                                  const TrackingRequest& request,
                                  const IterationTransforms& transforms) const {
  const Eigen::Vector3f source =
      (transforms.rotation * to_eigen(live_vertex)) + transforms.translation;
  const Eigen::Vector4f source_in_model_camera =
      transforms.model_world_to_camera * source.homogeneous();
  if (!source_in_model_camera.allFinite() ||
      source_in_model_camera.z() <= 0.0F) {
    return std::nullopt;
  }

  const Eigen::Vector2f pixel =
      request.model_intrinsics.project(source_in_model_camera.head<3>());
  const auto model_x = std::lround(pixel.x());
  const auto model_y = std::lround(pixel.y());
  if (model_x < 0 || model_y < 0) {
    return std::nullopt;
  }
  const auto col = static_cast<std::size_t>(model_x);
  const auto row = static_cast<std::size_t>(model_y);
  if (col >= request.model.vertices.width ||
      row >= request.model.vertices.height) {
    return std::nullopt;
  }

  const Vec3f& model_vertex_sample = request.model.vertices.at(col, row);
  const Vec3f& model_normal_sample = request.model.normals.at(col, row);
  if (!all_finite(model_vertex_sample) || !all_finite(model_normal_sample)) {
    return std::nullopt;
  }

  const Eigen::Vector3f model_vertex = to_eigen(model_vertex_sample);
  const Eigen::Vector3f model_normal = to_eigen(model_normal_sample);
  const Eigen::Vector3f source_normal =
      (transforms.rotation * to_eigen(live_normal)).normalized();
  const float distance = (source - model_vertex).norm();
  if (distance > options_.max_point_distance) {
    return std::nullopt;
  }
  if (source_normal.dot(model_normal) < options_.min_normal_dot) {
    return std::nullopt;
  }

  Eigen::Matrix<float, kIcpDof, 1> jacobian;
  jacobian << source.cross(model_normal), model_normal;
  return Correspondence{.jacobian = jacobian,
                        .residual = model_normal.dot(model_vertex - source),
                        .distance = distance};
}

ProjectiveIcpTracker::SystemStability
ProjectiveIcpTracker::check_system_stability(
    const NormalEquations& equations) const {
  if (equations.count < options_.min_correspondences) {
    return {};
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, kIcpDof, kIcpDof>>
      solver{equations.matrix};
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
    const NormalEquations& equations) const {
  if (equations.count < options_.min_correspondences) {
    return {};
  }

  const Eigen::Matrix<float, kIcpDof, 1> solution =
      equations.matrix.ldlt().solve(equations.rhs);
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
