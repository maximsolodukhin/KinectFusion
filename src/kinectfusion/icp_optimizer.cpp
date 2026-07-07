#include <Eigen/Eigenvalues>
#include <cmath>
#include <expected>
#include <kinectfusion/icp_optimizer.hpp>
#include <utility>

namespace kinectfusion {

IcpOutcome ProjectiveIcpTracker::estimate_pose(
    unsigned int iterations, const VertexNormalMaps& live_maps,
    const SurfaceMaps& model_maps, const CameraIntrinsics& model_intrinsics,
    const Eigen::Matrix4f& model_camera_to_world,
    Eigen::Matrix4f initial_camera_to_world) const {
  IcpOutcome outcome{.pose = initial_camera_to_world};

  // Point and normal maps are produced together by the raycaster, so a size
  // mismatch is a caller error rather than a tracking failure. The expected
  // channel makes it impossible for the caller to ignore.
  if (model_maps.points.width() != model_maps.normals.width() ||
      model_maps.points.height() != model_maps.normals.height()) {
    outcome.result = std::unexpected(IcpFailure::invalid_input);
    return outcome;
  }

  const Eigen::Matrix4f model_world_to_camera = model_camera_to_world.inverse();
  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    const auto correspondences =
        find_correspondences(live_maps, model_maps, model_intrinsics,
                             model_world_to_camera, outcome.pose);
    outcome.diagnostics.correspondences = correspondences.count;
    if (correspondences.count > 0) {
      outcome.diagnostics.mean_point_distance =
          correspondences.distance_sum /
          static_cast<float>(correspondences.count);
    }
    if (correspondences.count < options_.min_correspondences) {
      outcome.result = std::unexpected(IcpFailure::too_few_correspondences);
      return outcome;
    }

    const auto stability = check_system_stability(correspondences);
    outcome.diagnostics.min_system_eigenvalue = stability.min_eigenvalue;
    outcome.diagnostics.condition_number = stability.condition_number;
    if (!stability.stable) {
      outcome.result = std::unexpected(IcpFailure::unconstrained_system);
      return outcome;
    }

    const auto increment = solve_increment(correspondences);
    if (!increment.solved) {
      outcome.result = std::unexpected(IcpFailure::solve_failed);
      return outcome;
    }

    outcome.diagnostics.update_translation = increment.update_translation;
    outcome.diagnostics.update_rotation = increment.update_rotation;
    if (increment.update_translation > options_.max_update_translation ||
        increment.update_rotation > options_.max_update_rotation) {
      outcome.result = std::unexpected(IcpFailure::update_too_large);
      return outcome;
    }

    outcome.pose = increment.transform * outcome.pose;

    if (increment.update_translation < options_.min_update_translation &&
        increment.update_rotation < options_.min_update_rotation) {
      outcome.result = IcpSuccess{
          Converged{increment.update_translation, increment.update_rotation}};
      return outcome;
    }
  }

  // Every completed iteration was stable and well-constrained (otherwise we'd
  // have returned early), but the budget ran out before reaching tolerance.
  outcome.result = IcpSuccess{MaxIterations{}};
  return outcome;
}

ProjectiveIcpTracker::CorrespondenceSet
ProjectiveIcpTracker::find_correspondences(
    const VertexNormalMaps& live_maps, const SurfaceMaps& model_maps,
    const CameraIntrinsics& model_intrinsics,
    const Eigen::Matrix4f& model_world_to_camera,
    const Eigen::Matrix4f& camera_to_world) const {
  CorrespondenceSet correspondences;

  const ConstHostVertexNormalMapsView live_view = view(live_maps);
  const ConstHostSurfaceMapsView model_view = view(model_maps);
  const auto rotation = camera_to_world.block<3, 3>(0, 0);
  const auto translation = camera_to_world.block<3, 1>(0, 3);

  for (std::size_t y = 0; y < live_view.vertices.height; ++y) {
    for (std::size_t x = 0; x < live_view.vertices.width; ++x) {
      const Vec3f& live_vertex_sample = live_view.vertices.at(x, y);
      const Vec3f& live_normal_sample = live_view.normals.at(x, y);

      if (!all_finite(live_vertex_sample) || !all_finite(live_normal_sample)) {
        continue;
      }

      const Eigen::Vector3f live_vertex = to_eigen(live_vertex_sample);
      const Eigen::Vector3f live_normal = to_eigen(live_normal_sample);
      const Eigen::Vector3f source = rotation * live_vertex + translation;
      const Eigen::Vector4f source_in_model_camera =
          model_world_to_camera *
          Eigen::Vector4f{source.x(), source.y(), source.z(), 1.0F};
      if (!source_in_model_camera.allFinite() ||
          source_in_model_camera.z() <= 0.0F) {
        continue;
      }

      const Eigen::Vector2f pixel =
          model_intrinsics.project(source_in_model_camera.head<3>());
      const auto model_x = static_cast<int>(std::lround(pixel.x()));
      const auto model_y = static_cast<int>(std::lround(pixel.y()));
      if (model_x < 0 || model_y < 0 ||
          static_cast<std::size_t>(model_x) >= model_view.points.width ||
          static_cast<std::size_t>(model_y) >= model_view.points.height) {
        continue;
      }

      const Vec3f& model_vertex_sample = model_view.points.at(
          static_cast<std::size_t>(model_x), static_cast<std::size_t>(model_y));
      const Vec3f& model_normal_sample = model_view.normals.at(
          static_cast<std::size_t>(model_x), static_cast<std::size_t>(model_y));
      if (!all_finite(model_vertex_sample) ||
          !all_finite(model_normal_sample)) {
        continue;
      }

      const Eigen::Vector3f model_vertex = to_eigen(model_vertex_sample);
      const Eigen::Vector3f model_normal = to_eigen(model_normal_sample);
      const Eigen::Vector3f source_normal =
          (rotation * live_normal).normalized();
      const Eigen::Vector3f delta = source - model_vertex;
      const float distance = delta.norm();
      if (distance > options_.max_point_distance) {
        continue;
      }
      if (source_normal.dot(model_normal) < options_.min_normal_dot) {
        continue;
      }

      Eigen::Matrix<float, 6, 1> row;
      row.head<3>() = source.cross(model_normal);
      row.tail<3>() = model_normal;
      const float point_plane_residual =
          model_normal.dot(model_vertex - source);
      correspondences.normal_matrix += row * row.transpose();
      correspondences.normal_rhs += row * point_plane_residual;
      correspondences.distance_sum += distance;
      ++correspondences.count;
    }
  }

  return correspondences;
}

ProjectiveIcpTracker::SystemStability
ProjectiveIcpTracker::check_system_stability(
    const CorrespondenceSet& correspondences) const {
  if (correspondences.count < options_.min_correspondences) {
    return {};
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> solver{
      correspondences.normal_matrix};
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
    const CorrespondenceSet& correspondences) const {
  Increment increment;
  if (correspondences.count < options_.min_correspondences) {
    return increment;
  }

  const Eigen::Matrix<float, 6, 1> solution =
      correspondences.normal_matrix.ldlt().solve(correspondences.normal_rhs);
  if (!solution.allFinite()) {
    return increment;
  }

  const Eigen::Vector3f angle_axis = solution.head<3>();
  const Eigen::Vector3f translation = solution.tail<3>();
  const float angle = angle_axis.norm();
  Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
  if (angle > 1.0e-12F) {
    rotation = Eigen::AngleAxisf(angle, angle_axis / angle).toRotationMatrix();
  }

  increment.transform = make_transform_matrix(rotation, translation);
  increment.update_rotation = angle;
  increment.update_translation = translation.norm();
  increment.solved = true;
  return increment;
}

}  // namespace kinectfusion
