#include <kinectfusion/icp_optimizer.hpp>

#include <Eigen/Eigenvalues>

#include <cmath>
#include <utility>

namespace kinectfusion {

ProjectiveIcpResult ProjectiveIcpTracker::estimate_pose(
    const VertexNormalMaps& live_maps,
    const SurfaceMaps& model_maps,
    const CameraIntrinsics& model_intrinsics,
    const Eigen::Matrix4f& model_camera_to_world,
    Eigen::Matrix4f initial_camera_to_world) const {
  ProjectiveIcpResult result;
  result.pose = initial_camera_to_world;

  if (model_maps.points.width() != model_maps.normals.width() ||
      model_maps.points.height() != model_maps.normals.height()) {
    return result;
  }

  Eigen::Matrix4f pose = initial_camera_to_world;
  const Eigen::Matrix4f model_world_to_camera = model_camera_to_world.inverse();
  bool system_stable = false;
  for (unsigned int iteration = 0; iteration < iterations_; ++iteration) {
    const auto correspondences = find_correspondences(
        live_maps, model_maps, model_intrinsics, model_world_to_camera, pose);
    result.correspondences = correspondences.count;
    if (correspondences.count > 0) {
      result.mean_point_distance =
          correspondences.distance_sum / static_cast<float>(correspondences.count);
    }

    if (correspondences.count < min_correspondences_) {
      result.pose = pose;
      result.status = ProjectiveIcpStatus::too_few_correspondences;
      return result;
    }

    const auto stability = check_system_stability(correspondences);
    result.min_system_eigenvalue = stability.min_eigenvalue;
    result.condition_number = stability.condition_number;
    system_stable = stability.stable;
    if (!stability.stable) {
      result.pose = pose;
      result.status = ProjectiveIcpStatus::unconstrained_system;
      return result;
    }

    const auto increment = solve_increment(correspondences);
    if (!increment.solved) {
      result.pose = pose;
      result.status = ProjectiveIcpStatus::solve_failed;
      return result;
    }

    result.update_rotation = increment.update_rotation;
    result.update_translation = increment.update_translation;
    if (result.update_translation > max_update_translation_ ||
        result.update_rotation > max_update_rotation_) {
      result.pose = pose;
      result.status = ProjectiveIcpStatus::update_too_large;
      return result;
    }

    pose = increment.transform * pose;

    if (result.update_translation < min_update_translation_ &&
        result.update_rotation < min_update_rotation_) {
      break;
    }
  }

  result.pose = pose;
  result.converged = system_stable && result.correspondences >= min_correspondences_;
  if (result.converged) {
    result.status = ProjectiveIcpStatus::converged;
  } else if (result.correspondences < min_correspondences_) {
    result.status = ProjectiveIcpStatus::too_few_correspondences;
  } else {
    result.status = ProjectiveIcpStatus::unconstrained_system;
  }
  return result;
}

ProjectiveIcpTracker::CorrespondenceSet
ProjectiveIcpTracker::find_correspondences(
    const VertexNormalMaps& live_maps,
    const SurfaceMaps& model_maps,
    const CameraIntrinsics& model_intrinsics,
    const Eigen::Matrix4f& model_world_to_camera,
    const Eigen::Matrix4f& camera_to_world) const {
  CorrespondenceSet correspondences;

  const auto rotation = camera_to_world.block<3, 3>(0, 0);
  const auto translation = camera_to_world.block<3, 1>(0, 3);

  for (unsigned int y = 0; y < live_maps.vertices.height(); ++y) {
    for (unsigned int x = 0; x < live_maps.vertices.width(); ++x) {
      const Eigen::Vector3f& live_vertex = live_maps.vertices.at(x, y);
      const Eigen::Vector3f& live_normal = live_maps.normals.at(x, y);

      if (!live_vertex.allFinite() || !live_normal.allFinite()) {
        continue;
      }

      const Eigen::Vector3f source = rotation * live_vertex + translation;
      const Eigen::Vector4f source_in_model_camera =
          model_world_to_camera *
          Eigen::Vector4f{source.x(), source.y(), source.z(), 1.0F};
      if (!source_in_model_camera.allFinite() || source_in_model_camera.z() <= 0.0F) {
        continue;
      }

      const auto model_x = static_cast<int>(
          std::lround(
              (source_in_model_camera.x() * model_intrinsics.fx / source_in_model_camera.z()) +
              model_intrinsics.cx));
      const auto model_y = static_cast<int>(
          std::lround(
              (source_in_model_camera.y() * model_intrinsics.fy / source_in_model_camera.z()) +
              model_intrinsics.cy));
      if (model_x < 0 || model_y < 0 ||
          std::cmp_greater_equal(model_x, model_maps.points.width()) ||
          std::cmp_greater_equal(model_y, model_maps.points.height())) {
        continue;
      }

      const Eigen::Vector3f& model_vertex = model_maps.points.at(
          static_cast<unsigned int>(model_x), static_cast<unsigned int>(model_y));
      const Eigen::Vector3f& model_normal = model_maps.normals.at(
          static_cast<unsigned int>(model_x), static_cast<unsigned int>(model_y));
      if (!model_vertex.allFinite() || !model_normal.allFinite()) {
        continue;
      }

      const Eigen::Vector3f source_normal = (rotation * live_normal).normalized();
      const Eigen::Vector3f delta = source - model_vertex;
      const float distance = delta.norm();
      if (distance > max_point_distance_) {
        continue;
      }
      if (source_normal.dot(model_normal) < min_normal_dot_) {
        continue;
      }

      Eigen::Matrix<float, 6, 1> row;
      row.head<3>() = source.cross(model_normal);
      row.tail<3>() = model_normal;
      const float point_plane_residual = model_normal.dot(model_vertex - source);
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
  if (correspondences.count < min_correspondences_) {
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
      .stable = min_eigenvalue >= min_system_eigenvalue_ &&
                condition_number <= max_condition_number_,
      .min_eigenvalue = min_eigenvalue,
      .condition_number = condition_number};
}

ProjectiveIcpTracker::Increment ProjectiveIcpTracker::solve_increment(
    const CorrespondenceSet& correspondences) const {
  Increment increment;
  if (correspondences.count < min_correspondences_) {
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
