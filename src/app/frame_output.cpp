#include "frame_output.hpp"

#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <kinectfusion/image_proc/write_png.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/volume.hpp>
#include <stdexcept>
#include <string>

#include "app_options.hpp"

namespace app {
namespace {

[[nodiscard]] std::string frame_prefix(int frame_index) {
  return std::format("frame_{:06d}", frame_index);
}

void write_raycast_image(const kinectfusion::SurfaceMaps& maps,
                         const std::filesystem::path& dir,
                         const std::string& prefix) {
  const auto path = dir / (prefix + "_raycast.png");
  kinectfusion::image_proc::write_png(maps.colors, path.string());
}

void write_raycast_point_cloud(const kinectfusion::SurfaceMaps& maps,
                               const std::filesystem::path& dir,
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

}  // namespace

void write_outputs(const AppOptions& app_options,
                   const kinectfusion::SurfaceMaps& maps, int frame_index) {
  if (!app_options.write_raycast_images && !app_options.write_point_clouds) {
    return;
  }

  std::filesystem::create_directories(app_options.output_dir);
  const auto prefix = frame_prefix(frame_index);
  if (app_options.write_raycast_images) {
    write_raycast_image(maps, app_options.output_dir, prefix);
  }
  if (app_options.write_point_clouds) {
    write_raycast_point_cloud(maps, app_options.output_dir, prefix);
  }
}

}  // namespace app
