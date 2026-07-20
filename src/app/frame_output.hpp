#ifndef KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP
#define KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP

#include <cstddef>
#include <deque>
#include <filesystem>
#include <future>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/volume.hpp>
#include <string>
#include <vector>

#include "app_options.hpp"

namespace app {

class FrameOutput {
 public:
  explicit FrameOutput(const AppOptions& options);
  ~FrameOutput();

  FrameOutput(const FrameOutput&) = delete;
  FrameOutput& operator=(const FrameOutput&) = delete;
  FrameOutput(FrameOutput&&) = delete;
  FrameOutput& operator=(FrameOutput&&) = delete;

  // Writes the enabled artifacts (raycast PNG, point-cloud PLY) into
  // output_dir / subdirectory. Encoding and disk IO run on background
  // tasks that overlap the next frame's compute; errors surface on the
  // following write_frame call (or in the destructor).
  void write_frame(kinectfusion::SurfaceMaps maps, int frame_index,
                   const std::string& subdirectory = {});

  [[nodiscard]] bool writes_frames() const {
    return write_raycast_images_ || write_point_clouds_;
  }

  // Appends one row per comparison to output_dir/ablation_stats.csv; the
  // first write of a run truncates whatever a previous run left there.
  void append_ablation_stats(
      int frame_index,
      const std::vector<kinectfusion::PipelineComparison>& comparisons);

 private:
  [[nodiscard]] static std::string frame_prefix(int frame_index);
  static void write_raycast_image(const kinectfusion::SurfaceMaps& maps,
                                  const std::filesystem::path& dir,
                                  const std::string& prefix);

  static void write_raycast_point_cloud(const kinectfusion::SurfaceMaps& maps,
                                        const std::filesystem::path& dir,
                                        const std::string& prefix);
  void finish_pending_writes();

  // ~8 frames of PNG+PLY tasks in flight before the producer blocks.
  static constexpr std::size_t kMaxPendingWrites = 16;

  std::filesystem::path output_dir_;
  bool write_raycast_images_;
  bool write_point_clouds_;
  bool ablation_stats_started_{false};
  std::deque<std::future<void>> pending_writes_;
};

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP */
