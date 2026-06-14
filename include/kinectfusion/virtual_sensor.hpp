#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/image_proc/read_png.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

class VirtualSensor {
 public:
  bool init(const std::filesystem::path& dataset_dir) {
    base_dir_ = dataset_dir;

    if (!read_file_list(base_dir_ / "depth.txt", depth_filenames_,
                        depth_timestamps_)) {
      return false;
    }
    if (!read_file_list(base_dir_ / "rgb.txt", color_filenames_,
                        color_timestamps_)) {
      return false;
    }
    if (!read_trajectory_file(base_dir_ / "groundtruth.txt", trajectory_,
                              trajectory_timestamps_)) {
      return false;
    }

    current_idx_ = -1;
    return !depth_filenames_.empty();
  }

  bool process_next_frame() {
    const auto next_idx = current_idx_ < 0 ? 0 : current_idx_ + increment_;
    if (next_idx < 0 ||
        static_cast<std::size_t>(next_idx) >= depth_filenames_.size()) {
      return false;
    }
    current_idx_ = next_idx;

    const auto depth_path = base_dir_ / depth_filenames_[current_idx_];
    depth_image_ =
        image_proc::read_png<image_proc::DepthImage>(depth_path.string());

    const auto color_idx = nearest_timestamp_index(
        color_timestamps_, depth_timestamps_[current_idx_]);
    if (color_idx.has_value()) {
      const auto color_path = base_dir_ / color_filenames_[*color_idx];
      color_image_ =
          image_proc::read_png<image_proc::ColorImage>(color_path.string());
    } else {
      color_image_ = {};
    }

    current_trajectory_ =
        nearest_trajectory(depth_timestamps_[current_idx_]).value_or(
            Eigen::Matrix4f::Identity());
    return true;
  }

  void set_increment(int increment) { increment_ = increment; }

  [[nodiscard]] int current_frame_index() const { return current_idx_; }

  [[nodiscard]] const image_proc::DepthImage& depth_image() const {
    return depth_image_;
  }

  [[nodiscard]] const image_proc::ColorImage& color_image() const {
    return color_image_;
  }

  [[nodiscard]] CameraIntrinsics color_intrinsics() const {
    return color_intrinsics_;
  }

  [[nodiscard]] CameraIntrinsics depth_intrinsics() const {
    return depth_intrinsics_;
  }

  [[nodiscard]] Eigen::Matrix4f color_extrinsics() const { return color_extrinsics_; }

  [[nodiscard]] Eigen::Matrix4f depth_extrinsics() const { return depth_extrinsics_; }

  [[nodiscard]] Eigen::Matrix4f trajectory() const { return current_trajectory_; }

 private:
  [[nodiscard]] std::optional<Eigen::Matrix4f> nearest_trajectory(
      double timestamp) const {
    const auto best = nearest_timestamp_index(trajectory_timestamps_, timestamp);
    if (!best.has_value()) {
      return std::nullopt;
    }
    return trajectory_[*best];
  }

  [[nodiscard]] static std::optional<std::size_t> nearest_timestamp_index(
      const std::vector<double>& timestamps, double timestamp) {
    if (timestamps.empty()) {
      return std::nullopt;
    }

    auto best = std::size_t{};
    auto best_distance = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < timestamps.size(); ++i) {
      const auto distance = std::abs(timestamps[i] - timestamp);
      if (distance < best_distance) {
        best = i;
        best_distance = distance;
      }
    }
    return best;
  }

  static bool read_file_list(const std::filesystem::path& filename,
                             std::vector<std::string>& paths,
                             std::vector<double>& timestamps) {
    std::ifstream file{filename};
    if (!file) {
      return false;
    }

    paths.clear();
    timestamps.clear();

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream input{line};
      double timestamp = 0.0;
      std::string path;
      if (input >> timestamp >> path) {
        timestamps.push_back(timestamp);
        paths.push_back(path);
      }
    }

    return !paths.empty();
  }

  static bool read_trajectory_file(const std::filesystem::path& filename,
                                   std::vector<Eigen::Matrix4f>& poses,
                                   std::vector<double>& timestamps) {
    std::ifstream file{filename};
    if (!file) {
      return false;
    }

    poses.clear();
    timestamps.clear();

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream input{line};
      double timestamp = 0.0;
      Eigen::Vector3f translation;
      Eigen::Quaternionf rotation;
      if (!(input >> timestamp >> translation.x() >> translation.y() >>
            translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >>
            rotation.w())) {
        continue;
      }

      if (rotation.norm() == 0.0F) {
        continue;
      }

      rotation.normalize();
      timestamps.push_back(timestamp);
      poses.push_back(make_transform_matrix(rotation.toRotationMatrix(), translation)
                          .inverse()
                          .eval());
    }

    return !poses.empty();
  }

  std::filesystem::path base_dir_;
  int current_idx_{-1};
  int increment_{1};

  image_proc::DepthImage depth_image_;
  image_proc::ColorImage color_image_;
  Eigen::Matrix4f current_trajectory_{Eigen::Matrix4f::Identity()};

  CameraIntrinsics color_intrinsics_{517.3F, 516.5F, 318.6F, 255.3F};
  CameraIntrinsics depth_intrinsics_{517.3F, 516.5F, 318.6F, 255.3F};
  Eigen::Matrix4f color_extrinsics_{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f depth_extrinsics_{Eigen::Matrix4f::Identity()};

  std::vector<std::string> depth_filenames_;
  std::vector<std::string> color_filenames_;
  std::vector<double> depth_timestamps_;
  std::vector<double> color_timestamps_;
  std::vector<Eigen::Matrix4f> trajectory_;
  std::vector<double> trajectory_timestamps_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP */
