#ifndef KINECTFUSION_SRC_APP_TRAJECTORY_EVAL_HPP
#define KINECTFUSION_SRC_APP_TRAJECTORY_EVAL_HPP

#include <Eigen/Core>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace app {

// Translational absolute trajectory error, in meters.
struct AteStats {
  std::size_t pairs{};
  double rmse{};
  double mean{};
  double median{};
  double max_error{};
};

// Absolute Trajectory Error against a TUM-format groundtruth file. Associates
// each estimated pose to the nearest groundtruth pose in time, aligns the two
// point sets with the closed-form rigid Umeyama solution (no scale), and
// reports the translational error. Mirrors scripts/evaluate_ate.py so the two
// agree.
class AteEvaluator {
 public:
  static constexpr double kDefaultMaxDt = 0.02;
  static constexpr std::size_t kMinAssociations = 3;

  // Reads the groundtruth once. A missing or malformed file leaves
  // has_groundtruth() false rather than throwing.
  explicit AteEvaluator(const std::filesystem::path& groundtruth_path,
                        double max_dt = kDefaultMaxDt);

  [[nodiscard]] bool has_groundtruth() const { return !ground_.empty(); }

  // nullopt when fewer than kMinAssociations timestamps associate.
  [[nodiscard]] std::optional<AteStats> evaluate(
      const std::vector<std::pair<double, Eigen::Matrix4f>>& trajectory) const;

 private:
  struct Sample {
    double stamp{};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  };

  // Nearest groundtruth position within max_dt of `stamp`, or nullptr.
  [[nodiscard]] const Sample* associate(double stamp) const;

  std::vector<Sample> ground_;  // ascending by stamp
  double max_dt_;
};

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_TRAJECTORY_EVAL_HPP */
