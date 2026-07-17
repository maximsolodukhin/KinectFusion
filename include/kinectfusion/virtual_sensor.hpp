#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP

#include <cstddef>
#include <deque>
#include <filesystem>
#include <future>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <string>
#include <vector>

namespace kinectfusion {

// Reads a TUM RGB-D dataset frame by frame using the PNG decoder. Index files
// follow https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
// Frames are decoded ahead on background tasks so the caller's loop overlaps
// decode with processing; `preload` instead decodes the whole dataset into
// memory during init so the loop never touches the decoder.
class VirtualSensor {
 public:
  [[nodiscard]] bool init(const std::filesystem::path& dataset_dir,
                          bool preload = false);

  [[nodiscard]] bool process_next_frame();

  [[nodiscard]] const image_proc::DepthImage& depth_image() const;
  [[nodiscard]] const image_proc::ColorImage& color_image() const;
  [[nodiscard]] CameraIntrinsics depth_intrinsics() const;
  [[nodiscard]] int current_frame_index() const;

 private:
  struct FrameImages {
    image_proc::DepthImage depth;
    image_proc::ColorImage color;
  };

  static constexpr std::size_t kPrefetchDepth = 4;

  [[nodiscard]] static bool read_index(const std::filesystem::path& path,
                                       std::vector<std::string>& out);
  [[nodiscard]] FrameImages decode_frame(std::size_t index) const;
  void schedule_prefetch();
  void preload_all();

  std::filesystem::path base_dir_;
  std::vector<std::string> depth_files_;
  std::vector<std::string> color_files_;
  image_proc::DepthImage depth_image_;
  image_proc::ColorImage color_image_;
  CameraIntrinsics depth_intrinsics_{};
  int current_index_{-1};
  std::deque<std::future<FrameImages>> prefetched_;
  std::size_t next_frame_index_{0};
  std::vector<FrameImages> preloaded_;
  bool preload_{false};
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VIRTUAL_SENSOR_HPP */
