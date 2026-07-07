#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP

#include <filesystem>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <string>
#include <vector>

namespace kinectfusion {

// Reads a TUM RGB-D dataset frame by frame using the PNG decoder. Index files
// follow https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor {
 public:
  [[nodiscard]] bool init(const std::filesystem::path& dataset_dir);

  [[nodiscard]] bool process_next_frame();

  [[nodiscard]] const image_proc::DepthImage& depth_image() const;
  [[nodiscard]] const image_proc::ColorImage& color_image() const;
  [[nodiscard]] CameraIntrinsics depth_intrinsics() const;
  [[nodiscard]] int current_frame_index() const;

 private:
  [[nodiscard]] static bool read_index(const std::filesystem::path& path,
                                       std::vector<std::string>& out);

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
