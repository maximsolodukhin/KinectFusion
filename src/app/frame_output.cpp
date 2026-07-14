#include "frame_output.hpp"

#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <kinectfusion/image_proc/write_png.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <stdexcept>
#include <string>

#include "app_options.hpp"
#include "logging.hpp"

namespace app {

FrameOutput::FrameOutput(const AppOptions& options)
    : output_dir_(options.output_dir),
      write_raycast_images_(options.write_raycast_images),
      write_point_clouds_(options.write_point_clouds) {}

std::string FrameOutput::frame_prefix(int frame_index) {
  return std::format("frame_{:06d}", frame_index);
}

void FrameOutput::write_raycast_image(const kinectfusion::SurfaceMaps& maps,
                                      const std::filesystem::path& dir,
                                      const std::string& prefix) {
  const auto path = dir / (prefix + "_raycast.png");
  kinectfusion::image_proc::write_png(maps.colors, path.string());
}

void FrameOutput::write_raycast_point_cloud(
    const kinectfusion::SurfaceMaps& maps, const std::filesystem::path& dir,
    const std::string& prefix) {
  const auto& points = maps.points.data();
  const auto& normals = maps.normals.data();
  const auto& colors = maps.colors.data();

  std::size_t vertex_count = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (kinectfusion::all_finite(points.at(i)) &&
        kinectfusion::all_finite(normals.at(i))) {
      ++vertex_count;
    }
  }

  const auto path = dir / (prefix + "_raycast_point_cloud.ply");
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"Failed to open point cloud output: " +
                             path.string()};
  }

  output << "ply\n"
         << "format ascii 1.0\n"
         << "element vertex " << vertex_count << '\n'
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "property float nx\n"
         << "property float ny\n"
         << "property float nz\n"
         << "property uchar red\n"
         << "property uchar green\n"
         << "property uchar blue\n"
         << "end_header\n";

  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& point = points.at(i);
    const auto& normal = normals.at(i);
    if (!kinectfusion::all_finite(point) || !kinectfusion::all_finite(normal)) {
      continue;
    }

    const auto color = kinectfusion::rgba_from_pixel(colors.at(i));
    output << point.x << ' ' << point.y << ' ' << point.z << ' ' << normal.x
           << ' ' << normal.y << ' ' << normal.z << ' '
           << static_cast<int>(color.x()) << ' ' << static_cast<int>(color.y())
           << ' ' << static_cast<int>(color.z()) << '\n';
  }
}

void FrameOutput::write_frame(const kinectfusion::SurfaceMaps& maps,
                              int frame_index,
                              const std::string& subdirectory) const {
  if (!write_raycast_images_ && !write_point_clouds_) {
    return;
  }

  const auto dir =
      subdirectory.empty() ? output_dir_ : output_dir_ / subdirectory;
  std::filesystem::create_directories(dir);
  const auto prefix = frame_prefix(frame_index);
  if (write_raycast_images_) {
    write_raycast_image(maps, dir, prefix);
  }
  if (write_point_clouds_) {
    write_raycast_point_cloud(maps, dir, prefix);
  }
}

void FrameOutput::append_ablation_stats(
    int frame_index,
    const std::vector<kinectfusion::PipelineComparison>& comparisons) const {
  if (comparisons.empty()) {
    return;
  }

  std::filesystem::create_directories(output_dir_);
  const auto path = output_dir_ / "ablation_stats.csv";
  const bool write_header = !std::filesystem::exists(path);
  std::ofstream output{path, std::ios::app};
  if (!output) {
    throw std::runtime_error{"Failed to open ablation stats output: " +
                             path.string()};
  }

  if (write_header) {
    output << "frame,pipeline,compared_voxels,volume_only_pipeline,"
              "volume_only_reference,max_distance_delta,mean_distance_delta,"
              "max_weight_delta,compared_pixels,surface_only_pipeline,"
              "surface_only_reference,max_point_distance,mean_point_distance,"
              "max_normal_angle,mean_normal_angle\n";
  }
  for (const kinectfusion::PipelineComparison& comparison : comparisons) {
    output << std::format("{},{:csv}\n", frame_index, comparison);
  }
}

}  // namespace app
