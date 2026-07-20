#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/image_proc/read_png.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/virtual_sensor.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kinectfusion {
namespace {

// TUM RGB-D calibrated depth intrinsics per camera family, selected from
// the dataset directory name; freiburg1 is the fallback.
constexpr CameraIntrinsics kTumFreiburg1Depth{
    .fx = 517.3F, .fy = 516.5F, .cx = 318.6F, .cy = 255.3F};
constexpr CameraIntrinsics kTumFreiburg2Depth{
    .fx = 520.9F, .fy = 521.0F, .cx = 325.1F, .cy = 249.7F};
constexpr CameraIntrinsics kTumFreiburg3Depth{
    .fx = 535.4F, .fy = 539.2F, .cx = 320.1F, .cy = 247.6F};

[[nodiscard]] CameraIntrinsics tum_depth_intrinsics(
    const std::filesystem::path& dataset_dir) {
  const std::string name = dataset_dir.string();
  if (name.contains("freiburg2")) {
    return kTumFreiburg2Depth;
  }
  if (name.contains("freiburg3")) {
    return kTumFreiburg3Depth;
  }
  return kTumFreiburg1Depth;
}

}  // namespace

bool VirtualSensor::init(const std::filesystem::path& dataset_dir,
                         bool preload) {
  base_dir_ = dataset_dir;
  if (!read_index(dataset_dir / "depth.txt", depth_files_,
                  &depth_timestamps_)) {
    return false;
  }
  if (!read_index(dataset_dir / "rgb.txt", color_files_, nullptr)) {
    return false;
  }

  depth_intrinsics_ = tum_depth_intrinsics(dataset_dir);
  current_index_ = -1;

  prefetched_.clear();
  next_frame_index_ = 0;

  preloaded_.clear();
  preload_ = preload;

  if (depth_files_.empty() || color_files_.empty()) {
    return false;
  }

  if (preload_) {
    preload_all();
  } else {
    schedule_prefetch();
  }
  return true;
}

void VirtualSensor::preload_all() {
  const std::size_t frame_count =
      std::min(depth_files_.size(), color_files_.size());
  preloaded_.resize(frame_count);
  const std::size_t workers = std::max<std::size_t>(
      1,
      std::min<std::size_t>(std::thread::hardware_concurrency(), frame_count));
  std::vector<std::future<void>> tasks;
  tasks.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    tasks.push_back(std::async(std::launch::async, [this, worker, workers,
                                                    frame_count] {
      for (std::size_t index = worker; index < frame_count; index += workers) {
        preloaded_[index] = decode_frame(index);
      }
    }));
  }
  for (auto& task : tasks) {
    task.get();
  }
}

VirtualSensor::FrameImages VirtualSensor::decode_frame(
    std::size_t index) const {
  using namespace image_proc;

  const std::filesystem::path dpath = base_dir_ / depth_files_[index];
  const std::filesystem::path cpath = base_dir_ / color_files_[index];

  return {.depth = read_png<DepthImage>(dpath.string()),
          .color = read_png<ColorImage>(cpath.string())};
}

// can be done less ugly with a thread pool, but this is good enough for now
void VirtualSensor::schedule_prefetch() {
  const std::size_t frame_count =
      std::min(depth_files_.size(), color_files_.size());
  while (prefetched_.size() < kPrefetchDepth &&
         next_frame_index_ < frame_count) {
    prefetched_.push_back(std::async(
        std::launch::async,
        [this, index = next_frame_index_] { return decode_frame(index); }));

    ++next_frame_index_;
  }
}

bool VirtualSensor::process_next_frame() {
  if (preload_) {
    if (next_frame_index_ >= preloaded_.size()) {
      return false;
    }

    depth_image_ = std::move(preloaded_[next_frame_index_].depth);
    color_image_ = std::move(preloaded_[next_frame_index_].color);

    ++next_frame_index_;
    ++current_index_;

    return true;
  }

  if (prefetched_.empty()) {
    return false;
  }
  FrameImages images = prefetched_.front().get();
  prefetched_.pop_front();
  schedule_prefetch();

  depth_image_ = std::move(images.depth);
  color_image_ = std::move(images.color);

  ++current_index_;
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

double VirtualSensor::current_timestamp() const {
  if (current_index_ < 0 ||
      static_cast<std::size_t>(current_index_) >= depth_timestamps_.size()) {
    return 0.0;
  }
  return depth_timestamps_[static_cast<std::size_t>(current_index_)];
}

bool VirtualSensor::read_index(const std::filesystem::path& path,
                               std::vector<std::string>& out,
                               std::vector<double>* timestamps) {
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
    if (timestamps != nullptr) {
      timestamps->push_back(timestamp);
    }
  }
  return true;
}

}  // namespace kinectfusion
