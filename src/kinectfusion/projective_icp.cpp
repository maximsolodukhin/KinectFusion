#include <kinectfusion/projective_icp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <stdexcept>

// Disable false warnings about Ceres in GCC
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace kinectfusion {
namespace {

struct PointToPlaneConstraint {
  PointToPlaneConstraint(Eigen::Vector3f source, Eigen::Vector3f target,
                         Eigen::Vector3f normal)
      : source_(std::move(source)),
        target_(std::move(target)),
        normal_(std::move(normal)) {}

  template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    // pose[0..2] is the angle-axis rotation, pose[3..5] the translation.
    const T* rotation = pose;
    const T* translation = pose + 3;
    const T source[3] = {T(source_.x()), T(source_.y()), T(source_.z())};
    const T target[3] = {T(target_.x()), T(target_.y()), T(target_.z())};
    const T normal[3] = {T(normal_.x()), T(normal_.y()), T(normal_.z())};

    T transformed[3];
    ceres::AngleAxisRotatePoint(rotation, source, transformed);
    transformed[0] += translation[0];
    transformed[1] += translation[1];
    transformed[2] += translation[2];

    residual[0] = (normal[0] * (transformed[0] - target[0])) +
                  (normal[1] * (transformed[1] - target[1])) +
                  (normal[2] * (transformed[2] - target[2]));
    return true;
  }

  static ceres::CostFunction* create(const Eigen::Vector3f& source,
                                     const Eigen::Vector3f& target,
                                     const Eigen::Vector3f& normal) {
    return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
        new PointToPlaneConstraint(source, target, normal));
  }

  Eigen::Vector3f source_;
  Eigen::Vector3f target_;
  Eigen::Vector3f normal_;
};


[[nodiscard]] Eigen::Matrix4f increment_to_matrix(
    const std::array<double, 6>& pose) {
  std::array<double, 9> rotation{};
  ceres::AngleAxisToRotationMatrix(pose.data(), rotation.data());

  return make_transform_matrix(
      Eigen::Map<const Eigen::Matrix3d>(rotation.data()).cast<float>(),
      Eigen::Map<const Eigen::Vector3d>(pose.data() + 3).cast<float>());
}

void validate_options(const ProjectiveIcpOptions& options) {
  if (options.iterations == 0) {
    throw std::invalid_argument("Projective ICP must run at least one iteration");
  }
  if (options.solver_iterations == 0) {
    throw std::invalid_argument(
        "Projective ICP solver must run at least one iteration");
  }
  if (options.max_point_distance <= 0.0F) {
    throw std::invalid_argument("Projective ICP max distance must be positive");
  }
  if (options.min_normal_dot < -1.0F || options.min_normal_dot > 1.0F) {
    throw std::invalid_argument("Projective ICP normal threshold is invalid");
  }
  if (options.max_update_translation <= 0.0F ||
      options.max_update_rotation <= 0.0F) {
    throw std::invalid_argument("Projective ICP max update must be positive");
  }
  if (options.min_system_eigenvalue <= 0.0F ||
      options.max_condition_number <= 0.0F) {
    throw std::invalid_argument("Projective ICP stability thresholds must be positive");
  }
}

}  // namespace

ProjectiveIcpTracker::ProjectiveIcpTracker(ProjectiveIcpOptions options)
    : options_(options) {
  validate_options(options_);
}

void ProjectiveIcpTracker::set_iterations(unsigned int iterations) {
  options_.iterations = iterations;
  validate_options(options_);
}

void ProjectiveIcpTracker::set_min_correspondences(
    std::size_t min_correspondences) {
  options_.min_correspondences = min_correspondences;
}

void ProjectiveIcpTracker::set_max_point_distance(float max_point_distance) {
  options_.max_point_distance = max_point_distance;
  validate_options(options_);
}

void ProjectiveIcpTracker::set_min_normal_dot(float min_normal_dot) {
  options_.min_normal_dot = min_normal_dot;
  validate_options(options_);
}

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
  for (unsigned int iteration = 0; iteration < options_.iterations;
       ++iteration) {
    const auto correspondences = find_correspondences(
        live_maps, model_maps, model_intrinsics, model_world_to_camera, pose);
    result.correspondences = correspondences.matches.size();
    if (!correspondences.matches.empty()) {
      result.mean_point_distance =
          correspondences.distance_sum /
          static_cast<float>(correspondences.matches.size());
    }

    if (correspondences.matches.size() < options_.min_correspondences) {
      result.pose = pose;
      result.status = ProjectiveIcpStatus::too_few_correspondences;
      return result;
    }

    const auto stability = check_system_stability(correspondences);
    result.min_system_eigenvalue = stability.min_eigenvalue;
    result.condition_number = stability.condition_number;
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
    if (result.update_translation > options_.max_update_translation ||
        result.update_rotation > options_.max_update_rotation) {
      result.pose = pose;
      result.status = ProjectiveIcpStatus::update_too_large;
      return result;
    }

    pose = increment.transform * pose;

    if (result.update_translation < options_.min_update_translation &&
        result.update_rotation < options_.min_update_rotation) {
      break;
    }
  }

  result.pose = pose;
  result.converged = result.correspondences >= options_.min_correspondences;
  result.status = result.converged ? ProjectiveIcpStatus::converged
                                   : ProjectiveIcpStatus::too_few_correspondences;
  return result;
}

ProjectiveIcpTracker::CorrespondenceSet ProjectiveIcpTracker::find_correspondences(
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
      if (!source_in_model_camera.allFinite() ||
          source_in_model_camera.z() <= 0.0F) {
        continue;
      }

      const auto model_x = static_cast<int>(std::lround(
          (source_in_model_camera.x() * model_intrinsics.fx /
           source_in_model_camera.z()) +
          model_intrinsics.cx));
      const auto model_y = static_cast<int>(std::lround(
          (source_in_model_camera.y() * model_intrinsics.fy /
           source_in_model_camera.z()) +
          model_intrinsics.cy));
      if (model_x < 0 || model_y < 0 ||
          std::cmp_greater_equal(model_x, model_maps.points.width()) ||
          std::cmp_greater_equal(model_y, model_maps.points.height())) {
        continue;
      }

      const Eigen::Vector3f& model_vertex =
          model_maps.points.at(static_cast<unsigned int>(model_x),
                               static_cast<unsigned int>(model_y));
      const Eigen::Vector3f& model_normal =
          model_maps.normals.at(static_cast<unsigned int>(model_x),
                                static_cast<unsigned int>(model_y));
      if (!model_vertex.allFinite() || !model_normal.allFinite()) {
        continue;
      }

      const Eigen::Vector3f source_normal = (rotation * live_normal).normalized();
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
      correspondences.normal_matrix += row * row.transpose();
      correspondences.distance_sum += distance;
      correspondences.matches.push_back(
          Correspondence{.source = source,
                         .target = model_vertex,
                         .normal = model_normal});
    }
  }

  return correspondences;
}

ProjectiveIcpTracker::SystemStability ProjectiveIcpTracker::check_system_stability(
    const CorrespondenceSet& correspondences) const {
  if (correspondences.matches.size() < options_.min_correspondences) {
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
  if (correspondences.matches.size() < options_.min_correspondences) {
    return increment;
  }


  std::array<double, 6> pose{};
  ceres::Problem problem;
  for (const auto& match : correspondences.matches) {
    problem.AddResidualBlock(
        PointToPlaneConstraint::create(match.source, match.target, match.normal),
        nullptr, pose.data());
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations =
      static_cast<int>(options_.solver_iterations);
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  if (summary.termination_type == ceres::FAILURE ||
      summary.termination_type == ceres::USER_FAILURE) {
    return increment;
  }

  const Eigen::Vector3d angle_axis{pose[0], pose[1], pose[2]};
  const Eigen::Vector3d translation{pose[3], pose[4], pose[5]};
  if (!angle_axis.allFinite() || !translation.allFinite()) {
    return increment;
  }

  increment.transform = increment_to_matrix(pose);
  increment.update_rotation = static_cast<float>(angle_axis.norm());
  increment.update_translation = static_cast<float>(translation.norm());
  increment.solved = true;
  return increment;
}

}  // namespace kinectfusion
