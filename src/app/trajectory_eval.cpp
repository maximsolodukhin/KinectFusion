#include "trajectory_eval.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>  // NOLINT(misc-include-cleaner): Eigen::umeyama
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace app {
namespace {

[[nodiscard]] double median_of(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::ranges::sort(values);
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values.at(middle);
  }
  return 0.5 * (values.at(middle - 1) + values.at(middle));
}

}  // namespace

AteEvaluator::AteEvaluator(const std::filesystem::path& groundtruth_path,
                           double max_dt)
    : max_dt_(max_dt) {
  std::ifstream file{groundtruth_path};
  if (!file) {
    return;
  }
  // TUM line: timestamp tx ty tz qx qy qz qw. ATE needs only the position.
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream stream{line};
    Sample sample;
    std::array<double, 4> orientation{};
    if (stream >> sample.stamp >> sample.position.x() >> sample.position.y() >>
        sample.position.z() >> orientation.at(0) >> orientation.at(1) >>
        orientation.at(2) >> orientation.at(3)) {
      ground_.emplace_back(sample);
    }
  }
  std::ranges::sort(ground_, {}, &Sample::stamp);
}

const AteEvaluator::Sample* AteEvaluator::associate(double stamp) const {
  const auto upper =
      std::ranges::lower_bound(ground_, stamp, {}, &Sample::stamp);
  const Sample* best = nullptr;
  double best_dt = max_dt_;
  for (auto candidate : {upper, upper == ground_.begin() ? upper : upper - 1}) {
    if (candidate == ground_.end()) {
      continue;
    }
    const double delta = std::abs(candidate->stamp - stamp);
    if (delta <= best_dt) {
      best_dt = delta;
      best = &*candidate;
    }
  }
  return best;
}

std::optional<AteStats> AteEvaluator::evaluate(
    const std::vector<std::pair<double, Eigen::Matrix4f>>& trajectory) const {
  std::vector<Eigen::Vector3d> estimated;
  std::vector<Eigen::Vector3d> ground;
  estimated.reserve(trajectory.size());
  ground.reserve(trajectory.size());

  for (const auto& [stamp, pose] : trajectory) {
    if (const Sample* match = associate(stamp)) {
      estimated.emplace_back(pose.block<3, 1>(0, 3).cast<double>());
      ground.emplace_back(match->position);
    }
  }

  const std::size_t count = estimated.size();
  if (count < kMinAssociations) {
    return std::nullopt;
  }

  Eigen::Matrix3Xd source(3, static_cast<Eigen::Index>(count));
  Eigen::Matrix3Xd target(3, static_cast<Eigen::Index>(count));
  for (std::size_t index = 0; index < count; ++index) {
    source.col(static_cast<Eigen::Index>(index)) = estimated.at(index);
    target.col(static_cast<Eigen::Index>(index)) = ground.at(index);
  }

  const Eigen::Matrix4d transform = Eigen::umeyama(source, target, false);
  const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
  const Eigen::Vector3d translation = transform.block<3, 1>(0, 3);

  std::vector<double> errors(count);
  double squared_sum = 0.0;
  double sum = 0.0;
  double max_error = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const Eigen::Vector3d aligned =
        (rotation * source.col(static_cast<Eigen::Index>(index))) + translation;
    const double error =
        (aligned - target.col(static_cast<Eigen::Index>(index))).norm();
    errors.at(index) = error;
    squared_sum += error * error;
    sum += error;
    max_error = std::max(max_error, error);
  }

  const auto denominator = static_cast<double>(count);
  return AteStats{.pairs = count,
                  .rmse = std::sqrt(squared_sum / denominator),
                  .mean = sum / denominator,
                  .median = median_of(std::move(errors)),
                  .max_error = max_error};
}

}  // namespace app
