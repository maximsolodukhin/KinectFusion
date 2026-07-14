#ifndef KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP
#define KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP

#include <filesystem>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/volume.hpp>
#include <string>
#include <vector>

#include "app_options.hpp"

namespace app {

class FrameOutput {
 public:
  explicit FrameOutput(const AppOptions& options);

  // Writes the enabled artifacts (raycast PNG, point-cloud PLY) into
  // output_dir / subdirectory.
  void write_frame(const kinectfusion::SurfaceMaps& maps, int frame_index,
                   const std::string& subdirectory = {}) const;

  // Appends one row per comparison to output_dir/ablation_stats.csv.
  void append_ablation_stats(
      int frame_index,
      const std::vector<kinectfusion::PipelineComparison>& comparisons) const;

 private:
  [[nodiscard]] static std::string frame_prefix(int frame_index);
  static void write_raycast_image(const kinectfusion::SurfaceMaps& maps,
                                  const std::filesystem::path& dir,
                                  const std::string& prefix);
  static void write_raycast_point_cloud(const kinectfusion::SurfaceMaps& maps,
                                        const std::filesystem::path& dir,
                                        const std::string& prefix);

  std::filesystem::path output_dir_;
  bool write_raycast_images_;
  bool write_point_clouds_;
};

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP */
