#include "frame_output.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <ios>
#include <kinectfusion/image_proc/write_png.hpp>
#include <kinectfusion/marching_cubes.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "app_options.hpp"
// Provides the std::formatter specializations behind "{:csv}".
#include "logging.hpp"  // IWYU pragma: keep

namespace app {

FrameOutput::FrameOutput(const AppOptions& options)
    : output_dir_(options.output_dir),
      write_raycast_images_(options.write_raycast_images),
      write_point_clouds_(options.write_point_clouds) {}

FrameOutput::~FrameOutput() {
  try {
    finish_pending_writes();
  } catch (const std::exception& error) {
    spdlog::error("Frame output write failed: {}", error.what());
  }
}

void FrameOutput::finish_pending_writes() {
  for (auto& pending : pending_writes_) {
    pending.get();
  }
  pending_writes_.clear();
}

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
  static_assert(std::endian::native == std::endian::little);
  const auto& points = maps.points.data();
  const auto& normals = maps.normals.data();
  const auto& colors = maps.colors.data();

  std::string vertices;
  constexpr std::size_t kVertexBytes =
      (2 * sizeof(kinectfusion::Vec3f)) + (3 * sizeof(std::uint8_t));
  vertices.reserve(points.size() * kVertexBytes);
  const auto append_vec3f = [&vertices](const kinectfusion::Vec3f& value) {
    const auto bytes =
        std::bit_cast<std::array<char, sizeof(kinectfusion::Vec3f)>>(value);
    vertices.append(bytes.data(), bytes.size());
  };

  std::size_t vertex_count = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& point = points.at(i);
    const auto& normal = normals.at(i);
    if (!kinectfusion::all_finite(point) || !kinectfusion::all_finite(normal)) {
      continue;
    }

    ++vertex_count;
    append_vec3f(point);
    append_vec3f(normal);
    const auto color = kinectfusion::rgba_from_pixel(colors.at(i));
    vertices.push_back(static_cast<char>(color.x()));
    vertices.push_back(static_cast<char>(color.y()));
    vertices.push_back(static_cast<char>(color.z()));
  }

  const auto path = dir / (prefix + "_raycast_point_cloud.ply");
  std::ofstream output{path, std::ios::binary};
  if (!output) {
    throw std::runtime_error{"Failed to open point cloud output: " +
                             path.string()};
  }

  output << "ply\n"
         << "format binary_little_endian 1.0\n"
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
  output.write(vertices.data(), static_cast<std::streamsize>(vertices.size()));
}

void FrameOutput::write_mesh(const kinectfusion::MarchingCubes::Mesh& mesh,
                             const std::string& subdirectory) const {
  static_assert(std::endian::native == std::endian::little);

  std::string vertices;
  constexpr std::size_t kVertexBytes =
      (2 * sizeof(kinectfusion::Vec3f)) + (3 * sizeof(std::uint8_t));
  vertices.reserve(mesh.positions.size() * kVertexBytes);
  const auto append_vec3f = [&vertices](const kinectfusion::Vec3f& value) {
    const auto bytes =
        std::bit_cast<std::array<char, sizeof(kinectfusion::Vec3f)>>(value);
    vertices.append(bytes.data(), bytes.size());
  };
  const auto append_channel = [&vertices](float value) {
    vertices.push_back(static_cast<char>(static_cast<std::uint8_t>(
        std::clamp(value, 0.0F, kinectfusion::kMaxColorChannelValueF))));
  };
  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    append_vec3f(mesh.positions[i]);
    append_vec3f(mesh.normals[i]);
    append_channel(mesh.colors[i].x);
    append_channel(mesh.colors[i].y);
    append_channel(mesh.colors[i].z);
  }

  std::string faces;
  constexpr std::size_t kFaceBytes =
      sizeof(std::uint8_t) + (3 * sizeof(std::uint32_t));
  faces.reserve(mesh.triangles.size() * kFaceBytes);
  for (const auto& triangle : mesh.triangles) {
    faces.push_back(static_cast<char>(3));
    const auto bytes =
        std::bit_cast<std::array<char, sizeof(triangle)>>(triangle);
    faces.append(bytes.data(), bytes.size());
  }

  const auto dir =
      subdirectory.empty() ? output_dir_ : output_dir_ / subdirectory;
  std::filesystem::create_directories(dir);
  const auto path = dir / "mesh.ply";
  std::ofstream output{path, std::ios::binary};
  if (!output) {
    throw std::runtime_error{"Failed to open mesh output: " + path.string()};
  }

  output << "ply\n"
         << "format binary_little_endian 1.0\n"
         << "element vertex " << mesh.positions.size() << '\n'
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "property float nx\n"
         << "property float ny\n"
         << "property float nz\n"
         << "property uchar red\n"
         << "property uchar green\n"
         << "property uchar blue\n"
         << "element face " << mesh.triangles.size() << '\n'
         << "property list uchar uint vertex_indices\n"
         << "end_header\n";
  output.write(vertices.data(), static_cast<std::streamsize>(vertices.size()));
  output.write(faces.data(), static_cast<std::streamsize>(faces.size()));
}

void FrameOutput::write_frame(kinectfusion::SurfaceMaps maps, int frame_index,
                              const std::string& subdirectory) {
  if (!writes_frames()) {
    return;
  }

  while (pending_writes_.size() >= kMaxPendingWrites) {
    pending_writes_.front().get();
    pending_writes_.pop_front();
  }
  auto dir = subdirectory.empty() ? output_dir_ : output_dir_ / subdirectory;
  std::filesystem::create_directories(dir);
  auto prefix = frame_prefix(frame_index);
  const auto shared =
      std::make_shared<const kinectfusion::SurfaceMaps>(std::move(maps));
  if (write_raycast_images_) {
    pending_writes_.push_back(std::async(
        std::launch::async,
        [shared, dir, prefix] { write_raycast_image(*shared, dir, prefix); }));
  }
  if (write_point_clouds_) {
    pending_writes_.push_back(
        std::async(std::launch::async, [shared, dir, prefix] {
          write_raycast_point_cloud(*shared, dir, prefix);
        }));
  }
}

void FrameOutput::append_ablation_stats(
    int frame_index,
    const std::vector<kinectfusion::PipelineComparison>& comparisons) {
  if (comparisons.empty()) {
    return;
  }

  std::filesystem::create_directories(output_dir_);
  const auto path = output_dir_ / "ablation_stats.csv";
  const bool write_header = !ablation_stats_started_;
  std::ofstream output{path, write_header ? std::ios::trunc : std::ios::app};
  if (!output) {
    throw std::runtime_error{"Failed to open ablation stats output: " +
                             path.string()};
  }
  ablation_stats_started_ = true;

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
