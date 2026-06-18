#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/image_proc/read_png.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

// Reads a TUM RGB-D dataset frame by frame using the PNG decoder. Index files
// follow https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor {
 public:
  [[nodiscard]] bool init(const std::filesystem::path& dataset_dir) {
    base_dir_ = dataset_dir;
    if (!read_index(dataset_dir / "depth.txt", depth_files_)) {
      return false;
    }
    if (!read_index(dataset_dir / "rgb.txt", color_files_)) {
      return false;
    }
    // TUM RGB-D freiburg1 calibrated depth intrinsics.
    depth_intrinsics_ =
        CameraIntrinsics{.fx = 517.3F, .fy = 516.5F, .cx = 318.6F, .cy = 255.3F};
    current_index_ = -1;
    return !depth_files_.empty() && !color_files_.empty();
  }

  [[nodiscard]] bool process_next_frame() {
    ++current_index_;
    if (current_index_ < 0) {
      return false;
    }
    const std::size_t i = static_cast<std::size_t>(current_index_);
    if (i >= depth_files_.size() || i >= color_files_.size()) {
      return false;
    }
    depth_image_ = image_proc::read_png<image_proc::DepthImage>(
        (base_dir_ / depth_files_[i]).string());
    color_image_ = image_proc::read_png<image_proc::ColorImage>(
        (base_dir_ / color_files_[i]).string());
    return true;
  }

  [[nodiscard]] const image_proc::DepthImage& depth_image() const {
    return depth_image_;
  }
  [[nodiscard]] const image_proc::ColorImage& color_image() const {
    return color_image_;
  }
  [[nodiscard]] CameraIntrinsics depth_intrinsics() const {
    return depth_intrinsics_;
  }
  [[nodiscard]] int current_frame_index() const { return current_index_; }

 private:
  [[nodiscard]] static bool read_index(const std::filesystem::path& path,
                                       std::vector<std::string>& out) {
    std::ifstream file{path};
    if (!file) {
      return false;
    }
    out.clear();
    std::string line;
    for (int header = 0; header < 3 && std::getline(file, line); ++header) {
      // skip the three TUM header/comment lines
    }
    double timestamp = 0.0;
    std::string relative_path;
    while (file >> timestamp >> relative_path) {
      out.push_back(relative_path);
    }
    return true;
  }

  std::filesystem::path base_dir_;
  std::vector<std::string> depth_files_;
  std::vector<std::string> color_files_;
  image_proc::DepthImage depth_image_;
  image_proc::ColorImage color_image_;
  CameraIntrinsics depth_intrinsics_{};
  int current_index_{-1};
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP */
