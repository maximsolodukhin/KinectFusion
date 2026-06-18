#include <kinectfusion/virtual_sensor.hpp>

#include <fstream>

#include <kinectfusion/image_proc/read_png.hpp>

namespace kinectfusion {

bool VirtualSensor::init(const std::filesystem::path& dataset_dir) {
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

bool VirtualSensor::process_next_frame() {
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

const image_proc::DepthImage& VirtualSensor::depth_image() const {
  return depth_image_;
}

const image_proc::ColorImage& VirtualSensor::color_image() const {
  return color_image_;
}

CameraIntrinsics VirtualSensor::depth_intrinsics() const {
  return depth_intrinsics_;
}

int VirtualSensor::current_frame_index() const { return current_index_; }

bool VirtualSensor::read_index(const std::filesystem::path& path,
                               std::vector<std::string>& out) {
  std::ifstream file{path};
  if (!file) {
    return false;
  }
  out.clear();
  std::string line;
  for (int header = 0; header < 3 && std::getline(file, line); ++header) {
    // Skip the three TUM header/comment lines.
  }
  double timestamp = 0.0;
  std::string relative_path;
  while (file >> timestamp >> relative_path) {
    out.push_back(relative_path);
  }
  return true;
}

}  // namespace kinectfusion
